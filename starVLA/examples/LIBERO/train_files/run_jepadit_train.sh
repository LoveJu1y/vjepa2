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

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=JEPADiT
freeze_module_list='jepa_encoder'
config_yaml=./examples/LIBERO/train_files/jepadit_train_libero.yaml
jepa_ckpt=/share/project/lvjing/models/vjepa2_1/vjepa2_1_vitG_384.pt
libero_data_root=/share/project/baishuanghao/data/libero_lerobot
data_mix=libero_10
run_root_dir=./playground/Checkpoints
run_id=0419_libero_jepadit_vitG384_32bts_25k
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

WANDB_MODE="${WANDB_MODE}" "${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.jepa.checkpoint_path "${jepa_ckpt}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
