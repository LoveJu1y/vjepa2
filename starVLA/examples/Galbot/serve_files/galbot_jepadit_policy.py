"""OpenPI-compatible Galbot serve adapter for JEPADiT checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.Galbot.serve_files.galbot_serve_common import (
    SERVE_STATE_KEY,
    actions16_to_openpi_fullbody,
    decode_image,
    extract_state16_from_wire,
    load_state_action_stats,
    normalize_state,
    unnormalize_action,
)
from starVLA.model.framework.base_framework import baseframework

LOGGER = logging.getLogger(__name__)


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

        self.state_stats, self.action_stats = load_state_action_stats(stats_path, self.ckpt_path)
        self.expect_state_input = bool(getattr(self.model.config.datasets.vla_data, "include_state", False))
        self.metadata = {
            "model": "JEPADiT",
            "checkpoint": str(self.ckpt_path),
            "action_horizon": 30,
            "action_dim": 26,
            "wire_schema": "openpi_galbot_fullbody",
            "state_dim": 16,
            "expects_state_input": self.expect_state_input,
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
        }
        if self.expect_state_input:
            example["state"] = normalized_state[None, :]

        output = self.model.predict_action([example])
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized.ndim == 3:
            normalized = normalized[0]
        actions16 = unnormalize_action(normalized, self.action_stats)
        actions26 = actions16_to_openpi_fullbody(actions16, state16)
        return {"actions": actions26}
