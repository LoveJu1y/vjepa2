#!/usr/bin/env bash
set -euo pipefail

# Download missing EK100 videos from Bristol source by video_id list.
#
# Expected list format:
#   one video_id per line, e.g. P01_101
#
# Default local layout (file_format=2 compatible):
#   <dataset_root>/<split>/<PID>/<VIDEO_ID>.MP4
# where dataset_root defaults to:
#   /share/project/galbot-Hotel-Model/ego-data/epic_kitchens/3h91syskeag572hl6tvuovwv4d/videos
#
# Default source URL:
#   https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos

LIST_FILE="/share/project/lvjing/vjepa2/doc/missing_ek100_train_video_ids.txt"
DATASET_ROOT="/share/project/galbot-Hotel-Model/ego-data/epic_kitchens/3h91syskeag572hl6tvuovwv4d/videos"
BASE_URL="https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos"
SPLIT="train"
PROXY_URL=""
MAX_CONCURRENCY=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --list-file PATH           Missing video_id list file (default: ${LIST_FILE})
  --dataset-root PATH        Local dataset root (default: ${DATASET_ROOT})
  --base-url URL             Remote base url (default: ${BASE_URL})
  --split NAME               Split folder for url/path: train|test (default: ${SPLIT})
  --proxy URL                Proxy url, e.g. http://10.8.36.23:2080
  --max-concurrency N        Parallel downloads (default: ${MAX_CONCURRENCY})
  -h, --help                 Show this help

Examples:
  $(basename "$0") --proxy http://10.8.36.23:2080
  $(basename "$0") --split train --max-concurrency 4
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list-file)
      LIST_FILE="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --proxy)
      PROXY_URL="$2"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$LIST_FILE" ]]; then
  echo "List file not found: $LIST_FILE"
  exit 1
fi

if [[ -n "$PROXY_URL" ]]; then
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  echo "Using proxy: $PROXY_URL"
fi

if ! command -v aria2c >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "Need aria2c or wget in PATH."
  exit 1
fi

download_one() {
  local vid="$1"
  local pid="${vid%%_*}"
  local local_dir="${DATASET_ROOT}/${SPLIT}/${pid}"
  local local_file="${local_dir}/${vid}.MP4"
  local url="${BASE_URL}/${SPLIT}/${pid}/${vid}.MP4"

  mkdir -p "$local_dir"
  if [[ -f "$local_file" ]]; then
    echo "[SKIP] ${vid} exists"
    return 0
  fi

  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --allow-overwrite=false \
      --auto-file-renaming=false \
      --continue=true \
      --max-tries=8 \
      --retry-wait=3 \
      --timeout=60 \
      --connect-timeout=20 \
      --summary-interval=0 \
      -x 8 -s 8 -k 1M \
      --dir "$local_dir" \
      --out "${vid}.MP4" \
      "$url" >/dev/null
  else
    wget -c -t 8 -T 60 -O "$local_file" "$url" >/dev/null
  fi

  if [[ -f "$local_file" ]]; then
    echo "[OK]   ${vid}"
  else
    echo "[FAIL] ${vid}"
    return 1
  fi
}

export -f download_one
export DATASET_ROOT BASE_URL SPLIT

echo "list_file      : $LIST_FILE"
echo "dataset_root   : $DATASET_ROOT"
echo "base_url       : $BASE_URL"
echo "split          : $SPLIT"
echo "max_concurrency: $MAX_CONCURRENCY"

# shellcheck disable=SC2013
if [[ "$MAX_CONCURRENCY" -le 1 ]]; then
  while IFS= read -r vid; do
    [[ -z "$vid" ]] && continue
    download_one "$vid" || true
  done < "$LIST_FILE"
else
  # xargs parallelism for faster completion
  grep -v '^[[:space:]]*$' "$LIST_FILE" | xargs -I{} -P "$MAX_CONCURRENCY" bash -lc 'download_one "$@"' _ {}
fi

echo "Done."
