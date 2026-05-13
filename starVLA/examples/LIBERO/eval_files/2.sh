#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STARVLA_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PROJECT_ROOT=$(cd -- "${STARVLA_ROOT}/.." && pwd)
cd "${STARVLA_ROOT}"

GPU_ID=${GPU_ID:-4}
PORT=${PORT:-25010}
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
SERVER_LOG=${SERVER_LOG:-${VIDEO_OUT_PATH}/server.log}
EVAL_LOG=${EVAL_LOG:-${VIDEO_OUT_PATH}/eval.log}

mkdir -p "${LIBERO_CFG_DIR}" "${VIDEO_OUT_PATH}"

cat > "${LIBERO_CFG_DIR}/config.yaml" <<YAML
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/bddl_files
init_states: ${LIBERO_HOME}/libero/libero/init_files
assets: ${LIBERO_HOME}/libero/libero/assets
datasets: ${LIBERO_DATASETS_DIR}
YAML

export PYTHONPATH="${STARVLA_ROOT}:${PROJECT_ROOT}:${LIBERO_HOME}:${PYTHONPATH:-}"

print_debug_info() {
  echo "========== DEBUG INFO =========="
  echo "STARVLA_ROOT=${STARVLA_ROOT}"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "GPU_ID=${GPU_ID}"
  echo "PORT=${PORT}"
  echo "STARVLA_PYTHON=${STARVLA_PYTHON}"
  echo "LIBERO_PYTHON=${LIBERO_PYTHON}"
  echo "LIBERO_HOME=${LIBERO_HOME}"
  echo "LIBERO_DATASETS_DIR=${LIBERO_DATASETS_DIR}"
  echo "CKPT_PATH=${CKPT_PATH}"
  echo "VIDEO_OUT_PATH=${VIDEO_OUT_PATH}"
  echo "LIBERO_CFG_DIR=${LIBERO_CFG_DIR}"
  echo "SERVER_LOG=${SERVER_LOG}"
  echo "EVAL_LOG=${EVAL_LOG}"
  echo "XVFB_ERROR_LOG=${XVFB_ERROR_LOG}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "================================"
}

show_logs_on_error() {
  local exit_code=$?
  if [[ $exit_code -ne 0 ]]; then
    echo
    echo "================ ERROR DETECTED ================"
    echo "[server log tail]"
    tail -n 200 "${SERVER_LOG}" 2>/dev/null || true
    echo
    echo "[eval log tail]"
    tail -n 200 "${EVAL_LOG}" 2>/dev/null || true
    echo
    echo "[xvfb log tail]"
    tail -n 200 "${XVFB_ERROR_LOG}" 2>/dev/null || true
    echo "==============================================="
  fi
  return $exit_code
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap show_logs_on_error EXIT
trap cleanup EXIT

print_debug_info

echo "[0/3] Basic checks"
if [[ ! -x "${STARVLA_PYTHON}" ]]; then
  echo "STARVLA_PYTHON not executable: ${STARVLA_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO_PYTHON not executable: ${LIBERO_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME not found: ${LIBERO_HOME}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_DATASETS_DIR}" ]]; then
  echo "LIBERO_DATASETS_DIR not found: ${LIBERO_DATASETS_DIR}" >&2
  exit 1
fi

if ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"; then
  echo "Port ${PORT} is already in use." >&2
  ss -ltnp | grep ":${PORT}" || true
  exit 1
fi

echo "[1/3] Starting policy server on GPU ${GPU_ID}, port ${PORT}"
env -u DEBUG \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${STARVLA_PYTHON}" \
  deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${PORT}" \
    --use_bf16 \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "SERVER_PID=${SERVER_PID}"
echo "Server log -> ${SERVER_LOG}"

echo "[wait] Waiting for websocket server to become ready"
READY=0
for _ in $(seq 1 $((SERVER_READY_TIMEOUT / 2))); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server process exited early." >&2
    tail -n 200 "${SERVER_LOG}" >&2 || true
    exit 1
  fi

  if PORT="${PORT}" "${STARVLA_PYTHON}" - <<'PY' >/dev/null 2>&1
import asyncio
import os
import websockets

async def main():
    uri = f"ws://127.0.0.1:{os.environ['PORT']}"
    async with websockets.connect(uri, open_timeout=2, close_timeout=2):
        pass

asyncio.run(main())
PY
  then
    READY=1
    break
  fi

  sleep 2
done

if [[ "${READY}" != "1" ]]; then
  echo "Server did not become ready on port ${PORT}" >&2
  ss -ltnp | grep ":${PORT}" || true
  tail -n 200 "${SERVER_LOG}" >&2 || true
  exit 1
fi

echo "[ok] WebSocket server is ready on port ${PORT}"
ss -ltnp | grep ":${PORT}" || true

echo "[2/3] Running LIBERO-10 evaluation"
env -u DEBUG \
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_CONFIG_PATH="${LIBERO_CFG_DIR}" \
  MUJOCO_GL=glx \
  LIBGL_ALWAYS_SOFTWARE=1 \
  xvfb-run -a -e "${XVFB_ERROR_LOG}" -s '-screen 0 1024x768x24 +extension GLX +render -noreset' \
  "${LIBERO_PYTHON}" \
  examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.task-suite-name libero_10 \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.max-videos-to-save "${MAX_VIDEOS_TO_SAVE}" \
    --args.video-out-path "${VIDEO_OUT_PATH}" \
  2>&1 | tee "${EVAL_LOG}"

echo "[3/3] Finished"
echo "Videos / results -> ${VIDEO_OUT_PATH}"
echo "Server log       -> ${SERVER_LOG}"
echo "Eval log         -> ${EVAL_LOG}"
echo "Xvfb log         -> ${XVFB_ERROR_LOG}"