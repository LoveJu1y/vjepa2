from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
STARVLA_ROOT = REPO_ROOT / "starVLA"
OPENPI_ROOT = Path("/share/project/zhangningboo/workspace/maomingming-openpi-validation-loss/openpi")

sys.path.insert(0, str(STARVLA_ROOT))
sys.path.insert(0, str(OPENPI_ROOT / "src"))


@dataclass
class TaskSpec:
    name: str
    openpi_config: str
    openpi_data_root: str
    openpi_repo_id: str
    stats_path: str
    starvla_yaml: str


TASKS = [
    TaskSpec(
        name="book",
        openpi_config="pi05_galbot_mmm_book",
        openpi_data_root="/share/project/zhangningboo/galbot_g1_dataset/2_converted/5_book_0430/5_book_0430_val_loss_filtered_processed",
        openpi_repo_id="5_book_0430_val_loss_filtered_processed",
        stats_path="/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_book_0430_filtered_processed/norm_stats.json",
        starvla_yaml="/share/project/lvjing/vjepa2/starVLA/examples/Galbot/train_files/jepadit_train_galbot_book_arms_delta.yaml",
    ),
    TaskSpec(
        name="diewan",
        openpi_config="pi05_galbot_mmm_diewan",
        openpi_data_root="/share/project/zhangningboo/galbot_g1_dataset/2_converted/4_diewan_0502/4_diewan_0502_val_loss_filtered_processed",
        openpi_repo_id="4_diewan_0502_val_loss_filtered_processed",
        stats_path="/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_diewan_0502_filtered_processed/norm_stats.json",
        starvla_yaml="/share/project/lvjing/vjepa2/starVLA/examples/Galbot/train_files/jepadit_train_galbot_diewan_0502_arms_delta.yaml",
    ),
    TaskSpec(
        name="stamp",
        openpi_config="pi05_galbot_mmm_stamp",
        openpi_data_root="/share/project/zhangningboo/galbot_g1_dataset/2_converted/4_stamp_0503/4_stamp_0503_val_loss_filtered_processed",
        openpi_repo_id="4_stamp_0503_val_loss_filtered_processed",
        stats_path="/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_stamp_0503_filtered_processed/norm_stats.json",
        starvla_yaml="/share/project/lvjing/vjepa2/starVLA/examples/Galbot/train_files/jepadit_train_galbot_stamp_arms_delta.yaml",
    ),
    TaskSpec(
        name="chouzhi",
        openpi_config="pi05_galbot_mmm_chouzhi",
        openpi_data_root="/share/project/zhangningboo/galbot_g1_dataset/2_converted/2_chouzhi_0506/2_chouzhi_0506_val_loss_filtered_processed",
        openpi_repo_id="2_chouzhi_0506_val_loss_filtered_processed",
        stats_path="/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_chouzhi_0506_filtered_processed/norm_stats.json",
        starvla_yaml="/share/project/lvjing/vjepa2/starVLA/examples/Galbot/train_files/jepadit_train_galbot_chouzhi_arms_delta.yaml",
    ),
    TaskSpec(
        name="sugar",
        openpi_config="pi05_galbot_mmm_sugar",
        openpi_data_root="/share/project/zhangningboo/galbot_g1_dataset/2_converted/10_sugar_0507/10_sugar_0507_val_loss_filtered_processed",
        openpi_repo_id="10_sugar_0507_val_loss_filtered_processed",
        stats_path="/share/project/lvjing/vjepa2/5tasks_norm_states_filtered/5tasks_norm_states_filtered/2026_05_23_sugar_0507_filtered_processed/norm_stats.json",
        starvla_yaml="/share/project/lvjing/vjepa2/starVLA/examples/Galbot/train_files/jepadit_train_galbot_sugar_arms_delta.yaml",
    ),
]


def load_openpi_batch(task: TaskSpec):
    import dataclasses
    import openpi.training.config as opi_config
    import openpi.training.data_loader as opi_loader
    from openpi.shared import normalize as opi_normalize

    cfg = opi_config.get_config(task.openpi_config)
    raw_stats = json.loads(Path(task.stats_path).read_text())
    root = raw_stats.get("norm_stats", raw_stats)
    opi_stats = {
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
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    data_cfg = dataclasses.replace(
        data_cfg,
        data_root=task.openpi_data_root,
        repo_id=task.openpi_repo_id,
        norm_stats=opi_stats,
    )
    loader = opi_loader.create_torch_data_loader(
        data_cfg,
        model_config=cfg.model,
        action_horizon=cfg.model.action_horizon,
        batch_size=2,
        shuffle=False,
        num_batches=1,
        num_workers=0,
        seed=cfg.seed,
        framework="pytorch",
        gripper_columns=cfg.gripper_oversample_columns,
        gripper_oversample_factor=cfg.gripper_oversample_factor,
        gripper_motion_threshold=cfg.gripper_motion_threshold,
    )
    batch = next(iter(loader))
    obs, actions = batch
    return {
        "state": np.asarray(obs.state),
        "actions": np.asarray(actions),
    }


def load_starvla_batch(task: TaskSpec):
    import types
    import yaml
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    raw = yaml.safe_load(Path(task.starvla_yaml).read_text())
    data_cfg = types.SimpleNamespace(**raw["datasets"]["vla_data"])
    data_cfg.get = lambda key, default=None, _d=raw["datasets"]["vla_data"]: _d.get(key, default)
    dataset = get_vla_dataset(data_cfg=data_cfg)
    loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    state = np.asarray([sample["state"] for sample in batch], dtype=np.float32)
    actions = np.asarray([sample["action"] for sample in batch], dtype=np.float32)
    return {
        "state": state,
        "actions": actions,
    }


def summarize_diff(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a - b)
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
    }


def main():
    results = {}
    for task in TASKS[:1]:
        try:
            openpi_batch = load_openpi_batch(task)
            starvla_batch = load_starvla_batch(task)
            results[task.name] = {
                "state": summarize_diff(openpi_batch["state"], starvla_batch["state"]),
                "actions": summarize_diff(openpi_batch["actions"], starvla_batch["actions"]),
                "openpi_state_head": openpi_batch["state"][0, :16].tolist(),
                "starvla_state_head": starvla_batch["state"][0, 0, :16].tolist()
                if starvla_batch["state"].ndim == 3
                else starvla_batch["state"][0, :16].tolist(),
                "openpi_action_head": openpi_batch["actions"][0, 0, :16].tolist(),
                "starvla_action_head": starvla_batch["actions"][0, 0, :16].tolist(),
            }
        except Exception as exc:
            results[task.name] = {"error": repr(exc)}

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
