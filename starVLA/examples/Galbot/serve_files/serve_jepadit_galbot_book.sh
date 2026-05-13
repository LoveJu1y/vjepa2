#!/usr/bin/env bash
set -euo pipefail

cd /share/project/lvjing/vjepa2/starVLA

RUN_DIR="${RUN_DIR:-/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints/jepadit_galbot_book_0430_3view_arms_delta_20k_bs16_chunk30}"
STEP="${STEP:-20000}"
PORT="${PORT:-6688}"
DEVICE="${DEVICE:-cuda}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Galbot_G1_Push_the_book_to_the_edge_of_the_table_then_grab_it_and_place_it_on_the_bookshelf_2}"

if [[ "${STEP}" == "final" ]]; then
  CKPT_PATH="${RUN_DIR}/final_model/pytorch_model.pt"
else
  CKPT_PATH="${RUN_DIR}/checkpoints/steps_${STEP}_pytorch_model.pt"
fi
STATS_PATH="${RUN_DIR}/dataset_statistics.json"

echo "CKPT_PATH=${CKPT_PATH}"
echo "STATS_PATH=${STATS_PATH}"
echo "PORT=${PORT}"
echo "DEVICE=${DEVICE}"

exec /share/project/lvjing/miniconda3/envs/starVLA_tc/bin/python \
  examples/Galbot/serve_files/serve_jepadit_galbot.py \
  --ckpt-path "${CKPT_PATH}" \
  --stats-path "${STATS_PATH}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --port "${PORT}" \
  --device "${DEVICE}"

