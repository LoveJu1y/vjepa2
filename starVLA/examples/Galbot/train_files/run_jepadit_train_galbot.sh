#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STARVLA_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PROJECT_ROOT=$(cd -- "${STARVLA_ROOT}/.." && pwd)
cd "${STARVLA_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${STARVLA_ROOT}:${PYTHONPATH:-}"
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_bond_0
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

ACCELERATE_BIN=${ACCELERATE_BIN:-/share/project/lvjing/miniconda3/envs/starVLA_tc/bin/accelerate}
WANDB_MODE=${WANDB_MODE:-offline}

Framework_name=JEPADiT
freeze_module_list='jepa_encoder'
config_yaml=${CONFIG_YAML:-./examples/Galbot/train_files/jepadit_train_galbot.yaml}
jepa_ckpt=${JEPA_CKPT:-/share/project/lvjing/models/vjepa2_1/vjepa2_1_vitG_384.pt}
galbot_data_root=${GALBOT_DATA_ROOT:-/share/project/lvjing/vjepa2/starVLA/playground/Datasets/GALBOT_G1_DIEWAN}
data_mix=${DATA_MIX:-galbot_stack_bowl}
run_root_dir=${RUN_ROOT_DIR:-./playground/Checkpoints}
run_id=${RUN_ID:-jepadit_galbot_g1_stack_bowl_3view}
num_processes=${NUM_PROCESSES:-8}
main_process_port=${MAIN_PROCESS_PORT:-29581}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

WANDB_MODE="${WANDB_MODE}" "${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --main_process_port "${main_process_port}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.jepa.checkpoint_path "${jepa_ckpt}" \
  --datasets.vla_data.data_root_dir "${galbot_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  "$@"
