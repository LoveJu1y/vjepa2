"""OpenPI-compatible Galbot serve adapter for QwenGR00T checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from examples.Galbot.serve_files.galbot_serve_common import (
    SERVE_STATE_KEY,
    actions16_to_openpi_fullbody,
    decode_image,
    extract_state16_from_wire,
    load_state_action_stats,
    normalize_state,
    unnormalize_action,
)

LOGGER = logging.getLogger(__name__)


class QwenGR00TGalbotPolicy:
    """Policy wrapper that speaks the Galbot OpenPI wire schema."""

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

        self.state_stats, self.action_stats = load_state_action_stats(stats_path, self.ckpt_path)
        self.metadata = {
            "model": "QwenGR00T",
            "checkpoint": str(self.ckpt_path),
            "action_horizon": 30,
            "action_dim": 26,
            "wire_schema": "openpi_galbot_fullbody",
            "state_dim": 16,
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        state16 = extract_state16_from_wire(obs[SERVE_STATE_KEY])
        normalized_state = normalize_state(state16, self.state_stats)

        example = {
            "image": [
                decode_image(obs["observation/left_wrist_image"]),
                decode_image(obs["observation/wrist_image"]),
                decode_image(obs["observation/image"]),
            ],
            "lang": obs.get("prompt") or self.default_prompt or "",
            "state": normalized_state[None, :],
        }

        output = self.model.predict_action([example])
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized.ndim == 3:
            normalized = normalized[0]
        actions16 = unnormalize_action(normalized, self.action_stats)
        actions26 = actions16_to_openpi_fullbody(actions16, state16)
        return {"actions": actions26}
