#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CONFIG_YAML=${CONFIG_YAML:-./examples/Galbot/train_files/starvla_cotrain_galbot_chouzhi_qwengroot.yaml}
export GALBOT_DATA_ROOT=${GALBOT_DATA_ROOT:-/share/project/lvjing/vjepa2/starVLA/playground/Datasets/GALBOT_G1_CHOUZHI_0506_FILTERED}
export DATA_MIX=${DATA_MIX:-galbot_chouzhi_0506_filtered_arms_delta}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29704}
export WANDB_MODE=${WANDB_MODE:-online}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
export INCLUDE_STATE=${INCLUDE_STATE:-true}
STATE_TAG=$([[ "${INCLUDE_STATE,,}" == "true" || "${INCLUDE_STATE}" == "1" ]] && echo "withstate" || echo "nostate")
export RUN_ROOT_DIR=${RUN_ROOT_DIR:-./playground/Checkpoints_qwengroot_bs${PER_DEVICE_BATCH_SIZE}_step${MAX_TRAIN_STEPS}_${STATE_TAG}}
export RUN_ID=${RUN_ID:-qwengroot_galbot_chouzhi_0506_bs${PER_DEVICE_BATCH_SIZE}_step${MAX_TRAIN_STEPS}_ga2_lr1p5e5_${STATE_TAG}}
export WANDB_PROJECT=${WANDB_PROJECT:-starvla_galbot_qwengroot_bs${PER_DEVICE_BATCH_SIZE}_step${MAX_TRAIN_STEPS}_${STATE_TAG}}

bash "${SCRIPT_DIR}/run_qwengroot_train_galbot.sh" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --datasets.vla_data.include_state "${INCLUDE_STATE}" \
  --wandb_project "${WANDB_PROJECT}" \
  "$@"
