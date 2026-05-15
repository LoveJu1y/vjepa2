#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/share/project/lvjing/vjepa2}"
STARVLA_ROOT="${STARVLA_ROOT:-${REPO_ROOT}/starVLA}"
PYTHON_BIN="${PYTHON_BIN:-/share/project/lvjing/miniconda3/envs/starVLA_tc/bin/python}"

RUN_DIR="${RUN_DIR:-${STARVLA_ROOT}/playground/Checkpoints/jepadit_galbot_book_0430_3view_openpi_stats_20k_bs16_chunk30}"
STEP="${STEP:-20000}"
PORT="${PORT:-6688}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Galbot_G1_Push_the_book_to_the_edge_of_the_table_then_grab_it_and_place_it_on_the_bookshelf_2}"

if [[ "${STEP}" == "final" ]]; then
  CKPT_PATH="${RUN_DIR}/final_model/pytorch_model.pt"
else
  CKPT_PATH="${RUN_DIR}/checkpoints/steps_${STEP}_pytorch_model.pt"
fi

STATS_PATH="${STATS_PATH:-${RUN_DIR}/dataset_statistics.json}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/gal_real/server/serve_jepadit_policy.py" \
  --ckpt-path "${CKPT_PATH}" \
  --stats-path "${STATS_PATH}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  "$@"
