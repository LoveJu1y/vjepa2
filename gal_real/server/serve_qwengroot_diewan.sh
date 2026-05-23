#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/share/project/lvjing/vjepa2}"
STARVLA_ROOT="${STARVLA_ROOT:-${REPO_ROOT}/starVLA}"
PYTHON_BIN="${PYTHON_BIN:-/share/project/lvjing/miniconda3/envs/starVLA_tc/bin/python}"

RUN_DIR="${RUN_DIR:-${STARVLA_ROOT}/playground/Checkpoints/qwengroot_galbot_diewan_0502}"
STEP="${STEP:-20000}"
PORT="${PORT:-6688}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Galbot_G1_stack_bowl_1}"

if [[ "${STEP}" == "final" ]]; then
  CKPT_PATH="${RUN_DIR}/final_model/pytorch_model.pt"
else
  CKPT_PATH="${RUN_DIR}/checkpoints/steps_${STEP}_pytorch_model.pt"
fi

STATS_PATH="${STATS_PATH:-${RUN_DIR}/dataset_statistics.json}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/gal_real/server/serve_qwengroot_policy.py" \
  --ckpt-path "${CKPT_PATH}" \
  --stats-path "${STATS_PATH}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  "$@"
