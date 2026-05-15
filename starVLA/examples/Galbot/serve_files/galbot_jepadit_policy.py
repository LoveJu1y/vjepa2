"""OpenPI-compatible Galbot serve adapter for JEPADiT checkpoints."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from starVLA.model.framework.base_framework import baseframework

LOGGER = logging.getLogger(__name__)

SERVE_STATE_KEY = "observation/state"
SERVE_IMAGE_KEYS = {
    "observation/image": "head_left",
    "observation/left_wrist_image": "arm_left",
    "observation/wrist_image": "arm_right",
}


class JEPADiTGalbotPolicy:
    """Policy wrapper that speaks OpenPI Galbot wire schema.

    Input wire schema matches OpenPI's Galbot serve client:
      - observation/state: 23 dims, leg + head + left arm/gripper + right arm/gripper
      - observation/image: head-left RGB image
      - observation/left_wrist_image: left wrist RGB image
      - observation/wrist_image: right wrist RGB image

    Output schema is {"actions": [H, 26]} in the OpenPI full-body command layout:
    leg(5), head(2), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), chassis(3).
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        *,
        device: str = "cuda",
        use_bf16: bool = True,
        default_prompt: str | None = None,
        stats_path: str | Path | None = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.device = torch.device(device)
        self.default_prompt = default_prompt

        self.model = baseframework.from_pretrained(str(self.ckpt_path))
        if use_bf16:
            self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to(self.device).eval()

        self.action_stats = self._load_action_stats(stats_path)
        self.metadata = {
            "model": "JEPADiT",
            "checkpoint": str(self.ckpt_path),
            "action_horizon": 30,
            "action_dim": 26,
            "wire_schema": "openpi_galbot_fullbody",
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        flat_state = np.asarray(obs[SERVE_STATE_KEY], dtype=np.float32).reshape(-1)
        if flat_state.shape[0] != 23:
            raise ValueError(f"{SERVE_STATE_KEY} must have 23 dims, got {flat_state.shape}")

        state16 = self._extract_state16(flat_state)
        example = {
            "image": [
                self._decode_image(obs["observation/left_wrist_image"]),
                self._decode_image(obs["observation/wrist_image"]),
                self._decode_image(obs["observation/image"]),
            ],
            "lang": obs.get("prompt") or self.default_prompt or "",
            # The current JEPADiT Galbot configs were trained with include_state=false.
            # We keep state outside the model path and use it only for serve-time action conversion.
        }

        output = self.model.predict_action([example])
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized.ndim == 3:
            normalized = normalized[0]
        actions16 = self._unnormalize_action(normalized)
        actions26 = self._to_openpi_fullbody_actions(actions16, state16)
        return {"actions": actions26}

    def _load_action_stats(self, stats_path: str | Path | None) -> dict[str, np.ndarray]:
        if stats_path is None:
            run_dir = self._infer_run_dir(self.ckpt_path)
            stats_path = run_dir / "dataset_statistics.json"
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"dataset statistics not found: {stats_path}")

        stats = json.loads(stats_path.read_text())
        action = stats["new_embodiment"]["action"]
        low_key = "q01" if "q01" in action else "min"
        high_key = "q99" if "q99" in action else "max"
        return {
            "low": np.asarray(action[low_key], dtype=np.float32),
            "high": np.asarray(action[high_key], dtype=np.float32),
            "mask": np.asarray(action.get("mask", np.ones_like(action[low_key])), dtype=bool),
            "range_keys": (low_key, high_key),
        }

    @staticmethod
    def _infer_run_dir(ckpt_path: Path) -> Path:
        if ckpt_path.name == "pytorch_model.pt" and ckpt_path.parent.name == "final_model":
            return ckpt_path.parent.parent
        if ckpt_path.parent.name == "checkpoints":
            return ckpt_path.parent.parent
        return ckpt_path.parent

    @staticmethod
    def _decode_image(value: Any) -> Image.Image:
        arr = np.asarray(value)
        if arr.dtype.kind == "S":
            raw = arr.item() if arr.ndim else bytes(arr)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if arr.ndim == 0 and isinstance(arr.item(), (bytes, bytearray)):
            return Image.open(io.BytesIO(arr.item())).convert("RGB")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # add code for openloop test - znb
        # print(f"Decoded image with shape {arr.shape} and dtype {arr.dtype}")
        # arr = arr.transpose(1, 2, 0)
        # print(f"Decoded image with shape {arr.shape} and dtype {arr.dtype}")
        return Image.fromarray(arr).convert("RGB")

    @staticmethod
    def _extract_state16(flat_state: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                flat_state[7:14],
                flat_state[14:15],
                flat_state[15:22],
                flat_state[22:23],
            ],
            axis=0,
        ).astype(np.float32)

    def _unnormalize_action(self, normalized: np.ndarray) -> np.ndarray:
        normalized = np.clip(normalized, -1.0, 1.0)
        action_low = self.action_stats["low"]
        action_high = self.action_stats["high"]
        mask = self.action_stats["mask"]
        unnormalized = 0.5 * (normalized + 1.0) * (action_high - action_low + 1e-6) + action_low
        return np.where(mask, unnormalized, normalized).astype(np.float32)

    @staticmethod
    def _to_openpi_fullbody_actions(actions16: np.ndarray, state16: np.ndarray) -> np.ndarray:
        if actions16.ndim != 2 or actions16.shape[1] != 16:
            raise ValueError(f"Expected actions [H,16], got {actions16.shape}")

        out16 = actions16.copy()
        joint_idx = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])

        # Model outputs cumulative joint deltas relative to current state.
        # The robot client expects consecutive joint deltas for each served action step.
        absolute = actions16.copy()
        absolute[:, joint_idx] = state16[joint_idx] + actions16[:, joint_idx]
        shifted = np.empty_like(absolute)
        shifted[0] = state16
        shifted[1:] = absolute[:-1]
        out16[:, joint_idx] = absolute[:, joint_idx] - shifted[:, joint_idx]
        out16[:, [7, 15]] = actions16[:, [7, 15]]

        out26 = np.zeros((actions16.shape[0], 26), dtype=np.float32)
        out26[:, 7:14] = out16[:, 0:7]
        out26[:, 14] = out16[:, 7]
        out26[:, 15:22] = out16[:, 8:15]
        out26[:, 22] = out16[:, 15]
        return out26
