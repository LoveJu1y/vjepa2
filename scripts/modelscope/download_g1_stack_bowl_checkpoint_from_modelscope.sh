#!/usr/bin/env bash
set -euo pipefail

TOKEN="ms-26ef5124-69c8-4aa9-85ba-a0c2a8e7c7de"
PATH_IN_REPO="jepadit_galbot_g1_stack_bowl_3view_arms_delta_20k_bs16_chunk30"
LOCAL_DIR="/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <modelscope_repo_id> [local_dir] [path_in_repo]"
  echo "Example: $0 your_name/jepadit_galbot_g1_stack_bowl"
  exit 2
fi

REPO_ID="$1"
if [[ $# -ge 2 ]]; then
  LOCAL_DIR="$2"
fi
if [[ $# -ge 3 ]]; then
  PATH_IN_REPO="$3"
fi

mkdir -p "$LOCAL_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate base
else
  echo "conda was not found in PATH; continuing with current Python environment." >&2
fi

python - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("modelscope") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "modelscope"])
PY

python - "$REPO_ID" "$LOCAL_DIR" "$PATH_IN_REPO" "$TOKEN" <<'PY'
import sys
from pathlib import Path

from modelscope import snapshot_download

repo_id, local_dir, path_in_repo, token = sys.argv[1:5]
local_dir = Path(local_dir)
allow_patterns = [
    f"{path_in_repo}/**",
    f"{path_in_repo}/*",
]

print(f"Downloading modelscope://{repo_id}/{path_in_repo} to {local_dir}")
downloaded = snapshot_download(
    repo_id,
    local_dir=str(local_dir),
    allow_patterns=allow_patterns,
    token=token,
)
print(downloaded)
PY
