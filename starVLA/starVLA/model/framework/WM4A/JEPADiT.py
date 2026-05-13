# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
JEPADiT Framework.

Single-image V-JEPA 2.1 image-encoder branch + QwenGR00T-style flow-matching
action head for LeRobot/LIBERO training.

  image -> V-JEPA 2.1 image tokens -> GR00T cross-attention diffusion head

This framework intentionally matches the GR00T action head structure used by
QwenGR00T, replacing only the Qwen-VL encoder path with V-JEPA 2.1 image
tokens.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_repo_root = Path(__file__).resolve().parents[5]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import app.vjepa_2_1.models.vision_transformer as vit21
from deployment.model_server.tools.image_tools import to_pil_preserve
from src.utils.checkpoint_loader import robust_checkpoint_loader
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _clean_checkpoint_keys(state_dict):
    cleaned = {}
    for key, val in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        cleaned[new_key] = val
    return cleaned


def _resolve_encoder_arch(model_name: str) -> str:
    normalized = str(model_name).lower()
    if "gigantic" in normalized or "vitg_" in normalized and "_1_" in normalized:
        return "vit_gigantic_xformers"
    if "giant" in normalized or "vitg" in normalized:
        return "vit_giant_xformers"
    if "large" in normalized or "vitl" in normalized:
        return "vit_large"
    if "base" in normalized or "vitb" in normalized:
        return "vit_base"
    if normalized in vit21.__dict__:
        return normalized
    raise ValueError(f"Unsupported V-JEPA 2.1 model name: {model_name}")


@dataclass
class JEPADiTDefaultConfig:
    name: str = "JEPADiT"

    jepa: dict = field(default_factory=lambda: {
        "model_name": "vjepa2_1_vit_gigantic_384",
        "checkpoint_path": "",
        "checkpoint_key": "target_encoder",
        "img_size": 384,
        "patch_size": 16,
        "num_frames": 64,
        "tubelet_size": 2,
        "use_sdpa": True,
        "use_rope": True,
        "interpolate_rope": True,
        "img_temporal_dim_size": 1,
        "freeze_encoder": True,
        "image_views": 1,
        "multi_view_fusion": "concat_tokens",
    })

    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 7,
        "action_horizon": 8,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "inference_seed": 0,
        "num_target_vision_tokens": 32,
        "diffusion_model_cfg": {
            "cross_attention_dim": 1664,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
    })

    obs_image_size: Optional[list] = field(default_factory=lambda: [384, 384])


@FRAMEWORK_REGISTRY.register("JEPADiT")
class JEPA_DiT(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(JEPADiTDefaultConfig, config)

        jepa_cfg = self.config.framework.jepa

        encoder_arch = _resolve_encoder_arch(jepa_cfg.model_name)
        self.jepa_encoder = vit21.__dict__[encoder_arch](
            img_size=jepa_cfg.img_size,
            patch_size=jepa_cfg.patch_size,
            num_frames=jepa_cfg.num_frames,
            tubelet_size=jepa_cfg.tubelet_size,
            use_sdpa=jepa_cfg.use_sdpa,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=jepa_cfg.use_rope,
            img_temporal_dim_size=jepa_cfg.img_temporal_dim_size,
            interpolate_rope=jepa_cfg.interpolate_rope,
        )
        self._load_jepa_checkpoint(
            checkpoint_path=jepa_cfg.checkpoint_path,
            checkpoint_key=jepa_cfg.checkpoint_key,
        )

        # Match QwenGR00T's action head structure, but align cross-attention
        # to the raw V-JEPA token width at runtime.
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.jepa_encoder.embed_dim
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.image_size = tuple(getattr(self.config.framework, "obs_image_size", [jepa_cfg.img_size, jepa_cfg.img_size]))
        self.pixel_mean = (0.485, 0.456, 0.406)
        self.pixel_std = (0.229, 0.224, 0.225)

        if jepa_cfg.freeze_encoder:
            for param in self.jepa_encoder.parameters():
                param.requires_grad = False

    def _load_jepa_checkpoint(self, checkpoint_path: str, checkpoint_key: str) -> None:
        if not checkpoint_path:
            return

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"[JEPADiT] checkpoint not found, skip loading: {checkpoint_path}")
            return

        checkpoint = robust_checkpoint_loader(str(ckpt_path), map_location=torch.device("cpu"))
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unexpected checkpoint format at {checkpoint_path}")

        state_dict = None
        for key in [checkpoint_key, "ema_encoder", "target_encoder", "encoder"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            state_dict = checkpoint

        state_dict = _clean_checkpoint_keys(state_dict)
        msg = self.jepa_encoder.load_state_dict(state_dict, strict=False)
        print(f"[JEPADiT] loaded V-JEPA checkpoint from {checkpoint_path} with msg: {msg}")

    def _preprocess_images(self, batch_images: List) -> torch.Tensor:
        processed = []
        for image in batch_images:
            image = to_pil_preserve(image)
            image = image.convert("RGB")
            image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BICUBIC)
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, mean=self.pixel_mean, std=self.pixel_std)
            processed.append(tensor)
        return torch.stack(processed, dim=0).unsqueeze(2)  # [B, C, 1, H, W]

    def _normalize_image_views(self, examples: List[dict]) -> list[list]:
        expected_views = int(getattr(self.config.framework.jepa, "image_views", 1))
        normalized_views = []
        for example in examples:
            image = example["image"]
            views = image if isinstance(image, list) else [image]
            if len(views) < expected_views:
                raise ValueError(f"Expected at least {expected_views} image views, got {len(views)}")
            normalized_views.append(views[:expected_views])
        return normalized_views

    def _encode_image_batch(self, batch_images: List) -> torch.Tensor:
        device = next(self.action_model.parameters()).device
        encoder_dtype = next(self.jepa_encoder.parameters()).dtype
        action_dtype = next(self.action_model.parameters()).dtype
        image_tensor = self._preprocess_images(batch_images).to(device=device, dtype=encoder_dtype)
        encoder_requires_grad = any(p.requires_grad for p in self.jepa_encoder.parameters())
        use_autocast = device.type == "cuda" and encoder_dtype in {torch.float16, torch.bfloat16}
        with torch.set_grad_enabled(encoder_requires_grad):
            with torch.autocast(device_type=device.type, dtype=encoder_dtype, enabled=use_autocast):
                tokens = self.jepa_encoder(image_tensor)
        return tokens.to(dtype=action_dtype)

    def _extract_jepa_tokens(self, image_views: list[list]) -> torch.Tensor:
        fusion = str(getattr(self.config.framework.jepa, "multi_view_fusion", "concat_tokens"))
        if fusion != "concat_tokens":
            raise ValueError(f"Unsupported multi_view_fusion: {fusion}")

        num_views = len(image_views[0])
        view_tokens = []
        for view_idx in range(num_views):
            batch_images = [views[view_idx] for views in image_views]
            view_tokens.append(self._encode_image_batch(batch_images))
        return torch.cat(view_tokens, dim=1)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        image_views = self._normalize_image_views(examples)
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        vl_embs = self._extract_jepa_tokens(image_views)
        actions = torch.as_tensor(np.asarray(actions), device=vl_embs.device, dtype=vl_embs.dtype)
        actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

        repeated_diffusion_steps = self.config.framework.action_model.get("repeated_diffusion_steps", 4)
        vl_embs = vl_embs.repeat(repeated_diffusion_steps, 1, 1)
        actions_target = actions_target.repeat(repeated_diffusion_steps, 1, 1)

        state_repeated = None
        if state is not None:
            state_tensor = torch.as_tensor(np.asarray(state), device=vl_embs.device, dtype=vl_embs.dtype)
            state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1)

        action_loss = self.action_model(vl_embs, actions_target, state_repeated)
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        image_views = self._normalize_image_views(examples)
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        vl_embs = self._extract_jepa_tokens(image_views)
        state_tensor = (
            torch.as_tensor(np.asarray(state), device=vl_embs.device, dtype=vl_embs.dtype) if state is not None else None
        )
        action_dtype = next(self.action_model.parameters()).dtype
        use_autocast = vl_embs.device.type == "cuda" and action_dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=vl_embs.device.type, dtype=action_dtype, enabled=use_autocast):
            pred_actions = self.action_model.predict_action(vl_embs, state_tensor)
        return {"normalized_actions": pred_actions.float().detach().cpu().numpy()}
