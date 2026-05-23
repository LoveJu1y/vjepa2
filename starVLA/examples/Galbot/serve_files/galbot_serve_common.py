"""Shared Galbot serve-side preprocessing / postprocessing helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SERVE_STATE_KEY = "observation/state"
SERVE_IMAGE_KEYS = {
    "observation/image": "head_left",
    "observation/left_wrist_image": "arm_left",
    "observation/wrist_image": "arm_right",
}


def infer_run_dir(ckpt_path: str | Path) -> Path:
    ckpt_path = Path(ckpt_path)
    if ckpt_path.name == "pytorch_model.pt" and ckpt_path.parent.name == "final_model":
        return ckpt_path.parent.parent
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def decode_image(value: Any) -> Image.Image:
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        raw = arr.item() if arr.ndim else bytes(arr)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if arr.ndim == 0 and isinstance(arr.item(), (bytes, bytearray)):
        return Image.open(io.BytesIO(arr.item())).convert("RGB")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def extract_state16_from_wire(flat_state: np.ndarray) -> np.ndarray:
    flat_state = np.asarray(flat_state, dtype=np.float32).reshape(-1)
    if flat_state.shape[0] != 23:
        raise ValueError(f"{SERVE_STATE_KEY} must have 23 dims, got {flat_state.shape}")
    return np.concatenate(
        [
            flat_state[7:14],
            flat_state[14:15],
            flat_state[15:22],
            flat_state[22:23],
        ],
        axis=0,
    ).astype(np.float32)


def load_state_action_stats(stats_path: str | Path | None, ckpt_path: str | Path) -> tuple[dict, dict]:
    if stats_path is None:
        stats_path = infer_run_dir(ckpt_path) / "dataset_statistics.json"
    stats_path = Path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"dataset statistics not found: {stats_path}")

    stats = json.loads(stats_path.read_text())
    root = stats["new_embodiment"]
    return root["state"], root["action"]


def normalize_state(state16: np.ndarray, state_stats: dict) -> np.ndarray:
    state16 = np.asarray(state16, dtype=np.float32)
    q01 = np.asarray(state_stats["q01"], dtype=np.float32)
    q99 = np.asarray(state_stats["q99"], dtype=np.float32)
    normalized = (state16 - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    static = np.isclose(q99 - q01, 0.0)
    normalized[static] = state16[static]
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def unnormalize_action(normalized: np.ndarray, action_stats: dict) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=np.float32)
    normalized = np.clip(normalized, -1.0, 1.0)
    low_key = "q01" if "q01" in action_stats else "min"
    high_key = "q99" if "q99" in action_stats else "max"
    action_low = np.asarray(action_stats[low_key], dtype=np.float32)
    action_high = np.asarray(action_stats[high_key], dtype=np.float32)
    mask = np.asarray(action_stats.get("mask", np.ones_like(action_low)), dtype=bool)
    unnormalized = 0.5 * (normalized + 1.0) * (action_high - action_low + 1e-6) + action_low
    return np.where(mask, unnormalized, normalized).astype(np.float32)


def actions16_to_openpi_fullbody(actions16: np.ndarray, state16: np.ndarray) -> np.ndarray:
    actions16 = np.asarray(actions16, dtype=np.float32)
    state16 = np.asarray(state16, dtype=np.float32)
    if actions16.ndim != 2 or actions16.shape[1] != 16:
        raise ValueError(f"Expected actions [H,16], got {actions16.shape}")

    out16 = actions16.copy()
    joint_idx = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])

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
