#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STARVLA_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PROJECT_ROOT=$(cd -- "${STARVLA_ROOT}/.." && pwd)
cd "${STARVLA_ROOT}"

GPU_ID=${GPU_ID:-4}
PORT=${PORT:-25000}
NUM_TRIALS=${NUM_TRIALS:-50}
MAX_VIDEOS_TO_SAVE=${MAX_VIDEOS_TO_SAVE:-10}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-240}

STARVLA_PYTHON=${STARVLA_PYTHON:-/share/project/lvjing/miniconda3/envs/starVLA_tc/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/share/project/lvjing/miniconda3/envs/libero/bin/python}
LIBERO_HOME=${LIBERO_HOME:-/share/project/baishuanghao/code/LIBERO}
LIBERO_DATASETS_DIR=${LIBERO_DATASETS_DIR:-/share/project/baishuanghao/data}
CKPT_PATH=${CKPT_PATH:-/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints/0418_libero_jepadit_vitG384/checkpoints/steps_20000_pytorch_model.pt}

RUN_DIR=$(cd -- "$(dirname -- "${CKPT_PATH}")/.." && pwd)
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-${RUN_DIR}/results/libero_10}
LIBERO_CFG_DIR=${LIBERO_CFG_DIR:-/tmp/libero_cfg_bsh}
XVFB_ERROR_LOG=${XVFB_ERROR_LOG:-${VIDEO_OUT_PATH}/xvfb.log}
RUN_LOG=${RUN_LOG:-${VIDEO_OUT_PATH}/run.log}
SERVER_LOG=${SERVER_LOG:-${VIDEO_OUT_PATH}/server.log}
EVAL_LOG=${EVAL_LOG:-${VIDEO_OUT_PATH}/eval.log}

mkdir -p "${LIBERO_CFG_DIR}" "${VIDEO_OUT_PATH}"
rm -f "${RUN_LOG}" "${SERVER_LOG}" "${EVAL_LOG}" "${XVFB_ERROR_LOG}"
cat > "${LIBERO_CFG_DIR}/config.yaml" <<YAML
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/bddl_files
init_states: ${LIBERO_HOME}/libero/libero/init_files
assets: ${LIBERO_HOME}/libero/libero/assets
datasets: ${LIBERO_DATASETS_DIR}
YAML

export PYTHONPATH="${STARVLA_ROOT}:${PROJECT_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${RUN_LOG}"
}

log "[config] gpu=${GPU_ID} port=${PORT} trials=${NUM_TRIALS} max_videos=${MAX_VIDEOS_TO_SAVE}"
log "[config] ckpt=${CKPT_PATH}"
log "[config] output=${VIDEO_OUT_PATH}"
log "[config] server_log=${SERVER_LOG}"
log "[config] eval_log=${EVAL_LOG}"
log "[config] xvfb_log=${XVFB_ERROR_LOG}"

log "[1/2] Starting policy server on GPU ${GPU_ID}, port ${PORT}"
env -u DEBUG \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  stdbuf -oL -eL "${STARVLA_PYTHON}" \
  deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    --use_bf16 > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
log "[pid] server_pid=${SERVER_PID}"

log "[wait] Waiting for server to become ready"
for _ in $(seq 1 $((SERVER_READY_TIMEOUT / 2))); do
  if PORT="${PORT}" "${STARVLA_PYTHON}" - <<'PY' >/dev/null 2>&1
import os
import socket

port = int(os.environ["PORT"])
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
finally:
    s.close()
PY
  then
    break
  fi
  sleep 2
done

if ! PORT="${PORT}" "${STARVLA_PYTHON}" - <<'PY' >/dev/null 2>&1
import os
import socket

port = int(os.environ["PORT"])
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
finally:
    s.close()
PY
then
  log "[error] Server did not become ready on port ${PORT}"
  exit 1
fi

log "[ready] Server is ready on port ${PORT}"
log "[2/2] Running LIBERO-10 evaluation"
env -u DEBUG \
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_CONFIG_PATH="${LIBERO_CFG_DIR}" \
  MUJOCO_GL=glx \
  LIBGL_ALWAYS_SOFTWARE=1 \
  xvfb-run -a -e "${XVFB_ERROR_LOG}" -s '-screen 0 1024x768x24 +extension GLX +render -noreset' \
  stdbuf -oL -eL "${LIBERO_PYTHON}" \
  examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.max-videos-to-save "${MAX_VIDEOS_TO_SAVE}" \
    --args.video-out-path "${VIDEO_OUT_PATH}" 2>&1 | tee -a "${EVAL_LOG}"
