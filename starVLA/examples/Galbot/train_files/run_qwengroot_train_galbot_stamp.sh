#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export CONFIG_YAML=${CONFIG_YAML:-./examples/Galbot/train_files/starvla_cotrain_galbot_stamp_qwengroot.yaml}
export GALBOT_DATA_ROOT=${GALBOT_DATA_ROOT:-/share/project/lvjing/vjepa2/starVLA/playground/Datasets/GALBOT_G1_STAMP_0503_FILTERED}
export DATA_MIX=${DATA_MIX:-galbot_stamp_0503_filtered_arms_delta}
export RUN_ROOT_DIR=${RUN_ROOT_DIR:-./playground/Checkpoints_qwengroot_ga2_lr1p5e5}
export RUN_ID=${RUN_ID:-qwengroot_galbot_stamp_0503_bs8_5770_ga2_lr1p5e5}
export NUM_PROCESSES=${NUM_PROCESSES:-8}
export MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29703}
export WANDB_MODE=${WANDB_MODE:-online}

bash "${SCRIPT_DIR}/run_qwengroot_train_galbot.sh" "$@"
