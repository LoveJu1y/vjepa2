#!/usr/bin/env bash
set -euo pipefail

TOKEN="ms-26ef5124-69c8-4aa9-85ba-a0c2a8e7c7de"
CHECKPOINT_DIR="/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints/jepadit_galbot_g1_stack_bowl_3view_arms_delta_20k_bs16_chunk30"
PATH_IN_REPO="jepadit_galbot_g1_stack_bowl_3view_arms_delta_20k_bs16_chunk30"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <modelscope_repo_id> [path_in_repo]"
  echo "Example: $0 your_name/jepadit_galbot_g1_stack_bowl"
  exit 2
fi

REPO_ID="$1"
if [[ $# -ge 2 ]]; then
  PATH_IN_REPO="$2"
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 1
fi

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

python - "$REPO_ID" "$PATH_IN_REPO" "$CHECKPOINT_DIR" "$TOKEN" <<'PY'
import sys
from pathlib import Path

from modelscope.hub.api import HubApi

repo_id, path_in_repo, checkpoint_dir, token = sys.argv[1:5]
checkpoint_dir = Path(checkpoint_dir)

api = HubApi()
api.login(access_token=token)

try:
    api.create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="model",
        visibility="private",
        exist_ok=True,
    )
except Exception as exc:
    print(f"create_repo skipped or failed: {exc}")

print(f"Uploading {checkpoint_dir} to modelscope://{repo_id}/{path_in_repo}")
info = api.upload_folder(
    repo_id=repo_id,
    folder_path=str(checkpoint_dir),
    path_in_repo=path_in_repo,
    token=token,
    repo_type="model",
    commit_message=f"Upload {checkpoint_dir.name}",
)
print(info)
PY
