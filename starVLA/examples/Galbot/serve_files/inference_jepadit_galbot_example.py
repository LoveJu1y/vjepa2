"""Minimal local inference example for a Galbot JEPADiT checkpoint.

This script constructs a fake Galbot real-robot observation and runs the same
policy adapter used by the websocket server. It is meant for checking that a
checkpoint can load and produce OpenPI-compatible full-body actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


STARVLA_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = STARVLA_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STARVLA_ROOT))

from examples.Galbot.serve_files.galbot_jepadit_policy import JEPADiTGalbotPolicy  # noqa: E402


DEFAULT_RUN_DIR = (
    "/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints/"
    "jepadit_galbot_g1_stack_bowl_3view_arms_delta_20k_bs16_chunk30"
)
DEFAULT_PROMPT = "Galbot_G1_stack_bowl_1"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--step", default="20000", help="Checkpoint step, or 'final'.")
    parser.add_argument("--ckpt-path", default=None, help="Override checkpoint path.")
    parser.add_argument("--stats-path", default=None, help="Override dataset_statistics.json path.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_dir)
    if args.ckpt_path is not None:
        ckpt_path = Path(args.ckpt_path)
    elif args.step == "final":
        ckpt_path = run_dir / "final_model" / "pytorch_model.pt"
    else:
        ckpt_path = run_dir / "checkpoints" / f"steps_{args.step}_pytorch_model.pt"

    stats_path = Path(args.stats_path) if args.stats_path is not None else run_dir / "dataset_statistics.json"
    return ckpt_path, stats_path


def make_fake_obs(prompt: str, image_size: int) -> dict:
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    state = np.zeros(23, dtype=np.float32)

    # Set grippers to a plausible open width in the real-robot 23-dim layout.
    state[14] = 0.10
    state[22] = 0.10

    return {
        "observation/state": state,
        "observation/image": image,
        "observation/left_wrist_image": image,
        "observation/wrist_image": image,
        "prompt": prompt,
    }


def main() -> None:
    args = build_argparser().parse_args()
    ckpt_path, stats_path = resolve_paths(args)

    print(f"ckpt_path: {ckpt_path}")
    print(f"stats_path: {stats_path}")
    print(f"prompt: {args.prompt}")

    policy = JEPADiTGalbotPolicy(
        ckpt_path,
        stats_path=stats_path,
        device=args.device,
        use_bf16=not args.no_bf16,
        default_prompt=args.prompt,
    )

    obs = make_fake_obs(args.prompt, args.image_size)
    start = time.perf_counter()
    output = policy.infer(obs)
    elapsed = time.perf_counter() - start

    actions = np.asarray(output["actions"])
    print(f"actions.shape: {actions.shape}")
    print(f"actions.dtype: {actions.dtype}")
    print(f"actions finite: {np.isfinite(actions).all()}")
    print(f"actions min/max: {actions.min():.6f} / {actions.max():.6f}")
    print(f"first action: {np.array2string(actions[0], precision=6, suppress_small=False)}")
    print(f"infer_seconds: {elapsed:.4f}")


if __name__ == "__main__":
    main()
