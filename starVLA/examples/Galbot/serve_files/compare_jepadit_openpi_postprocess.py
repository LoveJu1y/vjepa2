"""Offline compare JEPA vs OpenPI Galbot postprocess without starting servers.

This fixes a normalized model output chunk and a state vector, then runs:
  JEPA path: unnormalize -> JEPA serve conversion
  OpenPI path: Unnormalize -> AbsoluteActions -> GalbotServeOutputs

The goal is to check whether both servers would send the same final wire-format
``actions`` payload for the same model output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


STARVLA_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = STARVLA_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STARVLA_ROOT))

from examples.Galbot.serve_files.galbot_jepadit_policy import JEPADiTGalbotPolicy  # noqa: E402


DEFAULT_STATS = (
    "/share/project/zhangningboo/workspace/maomingming-openpi-validation-loss/openpi/assets/"
    "pi05_galbot_mmm_diewan/mmm_2015_2016_2017_2019_val_loss_filtered_processed/norm_stats.json"
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-json", default=DEFAULT_STATS)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--state-json", default=None, help="Optional JSON file for 16-dim training-layout state.")
    parser.add_argument("--normalized-actions-json", default=None, help="Optional JSON file for [H,16] normalized output.")
    return parser


def load_stats(stats_json: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(stats_json).read_text())
    norm_stats = payload.get("norm_stats", payload)
    if "state" not in norm_stats or "actions" not in norm_stats:
        raise ValueError(f"Expected norm_stats with state/actions in {stats_json}")
    return norm_stats["state"], norm_stats["actions"]


def make_state(args: argparse.Namespace) -> np.ndarray:
    if args.state_json is not None:
        state = np.asarray(json.loads(Path(args.state_json).read_text()), dtype=np.float32)
    else:
        state = np.asarray(
            [
                1.6913550, -1.3919681, -0.45830077, -2.0673983, -0.02564587, -0.4058082, -0.15021001, 0.1,
                -1.7761587, 1.2525198, 0.39128533, 2.1899745, 0.07758512, 0.28071702, 0.20730323, 0.1,
            ],
            dtype=np.float32,
        )
    state = state.reshape(-1)
    if state.shape[0] != 16:
        raise ValueError(f"Expected 16-dim training-layout state, got {state.shape}")
    return state


def make_normalized_actions(args: argparse.Namespace) -> np.ndarray:
    if args.normalized_actions_json is not None:
        actions = np.asarray(json.loads(Path(args.normalized_actions_json).read_text()), dtype=np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        actions = rng.uniform(low=-0.95, high=0.95, size=(args.horizon, args.action_dim)).astype(np.float32)

    if actions.shape != (args.horizon, args.action_dim):
        raise ValueError(f"Expected normalized actions {(args.horizon, args.action_dim)}, got {actions.shape}")
    return actions


def run_jepa_path(
    normalized_actions: np.ndarray,
    state16: np.ndarray,
    action_stats: dict[str, Any],
) -> np.ndarray:
    policy = object.__new__(JEPADiTGalbotPolicy)
    policy.action_stats = {
        "low": np.asarray(action_stats["q01"], dtype=np.float32),
        "high": np.asarray(action_stats["q99"], dtype=np.float32),
        "mask": np.ones(len(action_stats["q01"]), dtype=bool),
        "range_keys": ("q01", "q99"),
    }
    unnormalized = policy._unnormalize_action(normalized_actions)
    return policy._to_openpi_fullbody_actions(unnormalized, state16)


def run_openpi_path(
    normalized_actions: np.ndarray,
    state16: np.ndarray,
    state_stats: dict[str, Any],
    action_stats: dict[str, Any],
) -> np.ndarray:
    del state_stats
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)
    actions = (normalized_actions.astype(np.float32) + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    delta_mask = np.asarray([True] * 7 + [False] + [True] * 7 + [False], dtype=bool)
    dims = delta_mask.shape[0]

    # OpenPI AbsoluteActions: restore body dims to absolute with current state anchor.
    actions[:, :dims] += np.expand_dims(np.where(delta_mask, state16[:dims], 0.0), axis=0)

    # OpenPI GalbotServeOutputs for arm-only config.
    anchor = np.zeros(dims, dtype=np.float32)
    anchor[:dims] = np.where(delta_mask, state16[:dims], 0.0)
    shifted = np.empty_like(actions)
    shifted[0] = anchor
    shifted[1:] = actions[:-1]
    out = actions - shifted

    gripper_idx = np.where(~delta_mask)[0]
    if gripper_idx.size:
        out[:, gripper_idx] = actions[:, gripper_idx]

    # arm-only layout reorder to fixed 26-dim wire schema
    full = np.zeros((out.shape[0], 26), dtype=np.float32)
    full[:, 7:14] = out[:, 0:7]
    full[:, 14:15] = out[:, 7:8]
    full[:, 15:22] = out[:, 8:15]
    full[:, 22:23] = out[:, 15:16]
    return full


def summarize(arr: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def main() -> None:
    args = build_argparser().parse_args()
    state_stats, action_stats = load_stats(args.stats_json)
    state16 = make_state(args)
    normalized_actions = make_normalized_actions(args)

    jepa_actions = run_jepa_path(normalized_actions, state16, action_stats)
    openpi_actions = run_openpi_path(normalized_actions, state16, state_stats, action_stats)
    diff = jepa_actions - openpi_actions

    result = {
        "state16": state16.tolist(),
        "normalized_actions_summary": summarize(normalized_actions),
        "jepa_actions_summary": summarize(jepa_actions),
        "openpi_actions_summary": summarize(openpi_actions),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "first_jepa_action": jepa_actions[0].tolist(),
        "first_openpi_action": openpi_actions[0].tolist(),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
