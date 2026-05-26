from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OPENPI_ROOT = Path("/share/project/zhangningboo/workspace/maomingming-openpi-validation-loss/openpi")
sys.path.insert(0, str(OPENPI_ROOT / "src"))

import openpi.training.config as opi_config  # noqa: E402
import openpi.training.data_loader as opi_loader  # noqa: E402
from openpi.shared import normalize as opi_normalize  # noqa: E402


STAR_REORDER = np.array([8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7])
JOINT_IDX = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14])


TASKS = {
    "book": {
        "config": "pi05_galbot_mmm_book",
        "data_root": "/share/project/zhangningboo/galbot_g1_dataset/2_converted/5_book_0430/5_book_0430_val_loss_filtered_processed",
        "repo_id": "5_book_0430_val_loss_filtered_processed",
        "stats": "/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_book_0430_filtered_processed/norm_stats.json",
    },
    "diewan": {
        "config": "pi05_galbot_mmm_diewan",
        "data_root": "/share/project/zhangningboo/galbot_g1_dataset/2_converted/4_diewan_0502/4_diewan_0502_val_loss_filtered_processed",
        "repo_id": "4_diewan_0502_val_loss_filtered_processed",
        "stats": "/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_diewan_0502_filtered_processed/norm_stats.json",
    },
    "stamp": {
        "config": "pi05_galbot_mmm_stamp",
        "data_root": "/share/project/zhangningboo/galbot_g1_dataset/2_converted/4_stamp_0503/4_stamp_0503_val_loss_filtered_processed",
        "repo_id": "4_stamp_0503_val_loss_filtered_processed",
        "stats": "/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_stamp_0503_filtered_processed/norm_stats.json",
    },
    "chouzhi": {
        "config": "pi05_galbot_mmm_chouzhi",
        "data_root": "/share/project/zhangningboo/galbot_g1_dataset/2_converted/2_chouzhi_0506/2_chouzhi_0506_val_loss_filtered_processed",
        "repo_id": "2_chouzhi_0506_val_loss_filtered_processed",
        "stats": "/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_chouzhi_0506_filtered_processed/norm_stats.json",
    },
    "sugar": {
        "config": "pi05_galbot_mmm_sugar",
        "data_root": "/share/project/zhangningboo/galbot_g1_dataset/2_converted/10_sugar_0507/10_sugar_0507_val_loss_filtered_processed",
        "repo_id": "10_sugar_0507_val_loss_filtered_processed",
        "stats": "/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_sugar_0507_filtered_processed/norm_stats.json",
    },
}

SAMPLE_INDICES = [0, 1, 5]


def load_stats(stats_path: str):
    raw = json.loads(Path(stats_path).read_text())
    root = raw.get("norm_stats", raw)
    return {
        "state": opi_normalize.NormStats(
            mean=np.asarray(root["state"]["mean"], dtype=np.float32),
            std=np.asarray(root["state"]["std"], dtype=np.float32),
            q01=np.asarray(root["state"]["q01"], dtype=np.float32),
            q99=np.asarray(root["state"]["q99"], dtype=np.float32),
        ),
        "actions": opi_normalize.NormStats(
            mean=np.asarray(root["actions"]["mean"], dtype=np.float32),
            std=np.asarray(root["actions"]["std"], dtype=np.float32),
            q01=np.asarray(root["actions"]["q01"], dtype=np.float32),
            q99=np.asarray(root["actions"]["q99"], dtype=np.float32),
        ),
    }


def star_prepare_from_parquet(parquet_path: Path, frame_index: int, horizon: int = 30):
    df = pd.read_parquet(parquet_path)
    right_arm = np.stack(df['observation.state.right_arm'].to_list()).astype(np.float32)
    right_gripper = np.stack(df['observation.state.right_gripper'].to_list()).astype(np.float32)
    left_arm = np.stack(df['observation.state.left_arm'].to_list()).astype(np.float32)
    left_gripper = np.stack(df['observation.state.left_gripper'].to_list()).astype(np.float32)
    raw = np.concatenate([right_arm, right_gripper, left_arm, left_gripper], axis=1).astype(np.float32)
    current = raw[frame_index]
    future = raw[frame_index + 1 : frame_index + 1 + horizon]
    state = current[STAR_REORDER].astype(np.float32)
    action = future[:, STAR_REORDER].astype(np.float32).copy()
    action[:, JOINT_IDX] -= state[JOINT_IDX]
    return state, action


def quant_norm(x: np.ndarray, stats: opi_normalize.NormStats):
    q01 = np.asarray(stats.q01, dtype=np.float32)
    q99 = np.asarray(stats.q99, dtype=np.float32)
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def summarize_diff(a: np.ndarray, b: np.ndarray):
    diff = np.abs(a - b)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
    }


def main():
    all_results = {}
    for task_name, spec in TASKS.items():
        cfg = opi_config.get_config(spec["config"])
        stats = load_stats(spec["stats"])
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        data_cfg = dataclasses.replace(
            data_cfg,
            data_root=spec["data_root"],
            repo_id=spec["repo_id"],
            norm_stats=stats,
        )
        raw_ds = opi_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
        norm = opi_loader._transforms.Normalize(stats, use_quantiles=data_cfg.use_quantile_norm)

        task_results = {}
        for sample_idx in SAMPLE_INDICES:
            raw_sample = raw_ds[sample_idx]
            episode_idx = int(np.asarray(raw_sample["episode_index"]).item())
            frame_idx = int(np.asarray(raw_sample["frame_index"]).item())
            parquet = sorted(Path(spec["data_root"]).glob(f"data/*/episode_{episode_idx:06d}.parquet"))[0]

            sample = raw_sample
            for t in data_cfg.repack_transforms.inputs:
                sample = t(sample)
            for t in data_cfg.data_transforms.inputs:
                sample = t(sample)
            opi_state_pre = np.asarray(sample["state"], dtype=np.float32)
            opi_action_pre = np.asarray(sample["actions"], dtype=np.float32)

            sample_norm = {"state": opi_state_pre.copy(), "actions": opi_action_pre.copy()}
            sample_norm = norm(sample_norm)
            opi_state_norm = np.asarray(sample_norm["state"], dtype=np.float32)
            opi_action_norm = np.asarray(sample_norm["actions"], dtype=np.float32)

            star_state_pre, star_action_pre = star_prepare_from_parquet(parquet, frame_idx, horizon=cfg.model.action_horizon)
            star_state_norm = quant_norm(star_state_pre, stats["state"])
            star_action_norm = quant_norm(star_action_pre, stats["actions"])

            task_results[str(sample_idx)] = {
                "identity": {
                    "episode_index": episode_idx,
                    "frame_index": frame_idx,
                    "parquet": str(parquet),
                },
                "pre_norm_state": summarize_diff(opi_state_pre, star_state_pre),
                "pre_norm_action": summarize_diff(opi_action_pre, star_action_pre),
                "post_norm_state": summarize_diff(opi_state_norm, star_state_norm),
                "post_norm_action": summarize_diff(opi_action_norm, star_action_norm),
                "opi_state_pre": opi_state_pre.tolist(),
                "star_state_pre": star_state_pre.tolist(),
                "opi_action0_pre": opi_action_pre[0].tolist(),
                "star_action0_pre": star_action_pre[0].tolist(),
                "opi_state_norm": opi_state_norm.tolist(),
                "star_state_norm": star_state_norm.tolist(),
                "opi_action0_norm": opi_action_norm[0].tolist(),
                "star_action0_norm": star_action_norm[0].tolist(),
            }
        all_results[task_name] = task_results

    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
