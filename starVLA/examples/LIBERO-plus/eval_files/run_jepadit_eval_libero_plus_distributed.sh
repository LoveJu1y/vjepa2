#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"

REPO_ROOT="/share/project/lvjing/vjepa2/starVLA"
STAR_PYTHON="${STAR_PYTHON:-/share/project/lvjing/miniconda3/envs/starVLA_tc/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/share/project/lvjing/miniconda3/envs/libero-plus/bin/python}"
LIBERO_HOME="${LIBERO_HOME:-/share/project/lvjing/LIBERO-plus}"

CKPT_PATH="${CKPT_PATH:-/share/project/lvjing/vjepa2/starVLA/playground/Checkpoints/0419_libero_jepadit_vitG384_32bts_25k/checkpoints/steps_25000_pytorch_model.pt}"
SUITE="${SUITE:-libero_10}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
SERVERS_PER_GPU="${SERVERS_PER_GPU:-2}"
WORKERS_PER_SERVER="${WORKERS_PER_SERVER:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
MAX_VIDEOS_TO_SAVE="${MAX_VIDEOS_TO_SAVE:-2}"
ASSIGNMENT_SEED="${ASSIGNMENT_SEED:-20260422}"
BASE_PORT="${BASE_PORT:-27000}"
LIBERO_DATASETS_PATH="${LIBERO_DATASETS_PATH:-/share/project/baishuanghao/data}"

CKPT_DIR="$(cd "$(dirname "${CKPT_PATH}")/.." && pwd)"
CKPT_STEM="$(basename "${CKPT_PATH}" .pt)"
RUN_NAME="${RUN_NAME:-${SUITE}_${CKPT_STEM}_8gpu_2server_8worker_seed${ASSIGNMENT_SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CKPT_DIR}/libero_plus_eval/${RUN_NAME}}"
CONFIG_DIR="${OUTPUT_ROOT}/libero_plus_config"
CONFIG_PATH="${CONFIG_DIR}/config.yaml"
ASSIGNMENT_JSON="${OUTPUT_ROOT}/assignment.json"
AGGREGATED_JSON="${OUTPUT_ROOT}/aggregate/metrics.json"

prepare_dirs() {
  mkdir -p "${OUTPUT_ROOT}" "${CONFIG_DIR}" "${OUTPUT_ROOT}/aggregate"
}

write_libero_config() {
  cat > "${CONFIG_PATH}" <<EOF
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/bddl_files
init_states: ${LIBERO_HOME}/libero/libero/init_files
assets: ${LIBERO_HOME}/libero/libero/assets
datasets: ${LIBERO_DATASETS_PATH}
EOF
}

generate_assignment() {
  LIBERO_HOME="${LIBERO_HOME}" \
  LIBERO_CONFIG_PATH="${CONFIG_DIR}" \
  PYTHONPATH="${REPO_ROOT}:${LIBERO_HOME}" \
  SUITE="${SUITE}" \
  GPU_IDS="${GPU_IDS}" \
  SERVERS_PER_GPU="${SERVERS_PER_GPU}" \
  WORKERS_PER_SERVER="${WORKERS_PER_SERVER}" \
  BASE_PORT="${BASE_PORT}" \
  ASSIGNMENT_SEED="${ASSIGNMENT_SEED}" \
  ASSIGNMENT_JSON="${ASSIGNMENT_JSON}" \
  "${LIBERO_PYTHON}" - <<'PY'
import json
import os
import random
from pathlib import Path

from libero.libero import benchmark

suite = os.environ["SUITE"]
gpu_ids = [int(x) for x in os.environ["GPU_IDS"].split(",") if x.strip()]
servers_per_gpu = int(os.environ["SERVERS_PER_GPU"])
workers_per_server = int(os.environ["WORKERS_PER_SERVER"])
base_port = int(os.environ["BASE_PORT"])
seed = int(os.environ["ASSIGNMENT_SEED"])
assignment_json = Path(os.environ["ASSIGNMENT_JSON"])

task_suite = benchmark.get_benchmark_dict()[suite]()
num_tasks = task_suite.n_tasks
task_ids = list(range(num_tasks))
rng = random.Random(seed)
rng.shuffle(task_ids)

total_servers = len(gpu_ids) * servers_per_gpu
total_workers = total_servers * workers_per_server

chunk_sizes = [num_tasks // total_workers] * total_workers
for idx in range(num_tasks % total_workers):
    chunk_sizes[idx] += 1

chunks = []
cursor = 0
for size in chunk_sizes:
    chunks.append(task_ids[cursor:cursor + size])
    cursor += size

servers = []
workers = []
worker_global_idx = 0
for gpu_idx, gpu_id in enumerate(gpu_ids):
    for server_slot in range(servers_per_gpu):
        port = base_port + gpu_id * 10 + server_slot
        server_name = f"gpu{gpu_id}_server{server_slot}"
        server_dir = assignment_json.parent / server_name
        servers.append(
            {
                "gpu_id": gpu_id,
                "server_slot": server_slot,
                "server_name": server_name,
                "port": port,
                "server_dir": str(server_dir),
            }
        )
        for worker_slot in range(workers_per_server):
            task_orders = chunks[worker_global_idx]
            worker_name = f"{server_name}_worker{worker_slot:02d}"
            worker_dir = server_dir / f"worker{worker_slot:02d}"
            workers.append(
                {
                    "worker_global_idx": worker_global_idx,
                    "worker_slot": worker_slot,
                    "worker_name": worker_name,
                    "gpu_id": gpu_id,
                    "server_slot": server_slot,
                    "server_name": server_name,
                    "port": port,
                    "task_orders": task_orders,
                    "worker_dir": str(worker_dir),
                }
            )
            worker_global_idx += 1

assignment = {
    "suite": suite,
    "seed": seed,
    "num_tasks": num_tasks,
    "gpu_ids": gpu_ids,
    "servers_per_gpu": servers_per_gpu,
    "workers_per_server": workers_per_server,
    "total_servers": total_servers,
    "total_workers": total_workers,
    "servers": servers,
    "workers": workers,
}

assignment_json.parent.mkdir(parents=True, exist_ok=True)
with open(assignment_json, "w", encoding="utf-8") as f:
    json.dump(assignment, f, indent=2, ensure_ascii=False)

print(assignment_json)
PY
}

launch_servers() {
  REPO_ROOT="${REPO_ROOT}" \
  STAR_PYTHON="${STAR_PYTHON}" \
  CKPT_PATH="${CKPT_PATH}" \
  ASSIGNMENT_JSON="${ASSIGNMENT_JSON}" \
  "${STAR_PYTHON}" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

repo_root = os.environ["REPO_ROOT"]
star_python = os.environ["STAR_PYTHON"]
ckpt_path = os.environ["CKPT_PATH"]
assignment = json.load(open(os.environ["ASSIGNMENT_JSON"], "r", encoding="utf-8"))

for server in assignment["servers"]:
    server_dir = Path(server["server_dir"])
    server_dir.mkdir(parents=True, exist_ok=True)
    log_path = server_dir / "server.log"
    pid_path = server_dir / "server.pid"
    cmd = (
        f"cd {repo_root} && "
        f"exec nohup env -u DEBUG "
        f"PYTHONPATH={repo_root}:/share/project/lvjing/vjepa2 "
        f"CUDA_VISIBLE_DEVICES={server['gpu_id']} "
        f"{star_python} deployment/model_server/server_policy.py "
        f"--ckpt_path {ckpt_path} "
        f"--port {server['port']} "
        f"--use_bf16 "
        f"> {log_path} 2>&1 < /dev/null"
    )
    proc = subprocess.Popen(["bash", "-lc", cmd], start_new_session=True)
    pid = str(proc.pid)
    pid_path.write_text(pid + "\n", encoding="utf-8")
    print(f"launched {server['server_name']} pid={pid} port={server['port']}")
PY

  REPO_ROOT="${REPO_ROOT}" \
  ASSIGNMENT_JSON="${ASSIGNMENT_JSON}" \
  "${STAR_PYTHON}" - <<'PY'
import json
import os
import time
from pathlib import Path

assignment = json.load(open(os.environ["ASSIGNMENT_JSON"], "r", encoding="utf-8"))

for server in assignment["servers"]:
    server_dir = Path(server["server_dir"])
    log_path = server_dir / "server.log"
    pid_path = server_dir / "server.pid"
    deadline = time.time() + 900
    while time.time() < deadline:
        pid = pid_path.read_text(encoding="utf-8").strip()
        if not Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"{server['server_name']} died before becoming ready")
        if log_path.exists() and "server running ..." in log_path.read_text(encoding="utf-8", errors="ignore"):
            print(f"ready {server['server_name']} port={server['port']}")
            break
        time.sleep(5)
    else:
        raise TimeoutError(f"Timed out waiting for {server['server_name']} to become ready")
PY
}

launch_workers() {
  REPO_ROOT="${REPO_ROOT}" \
  LIBERO_PYTHON="${LIBERO_PYTHON}" \
  LIBERO_HOME="${LIBERO_HOME}" \
  CONFIG_DIR="${CONFIG_DIR}" \
  CKPT_PATH="${CKPT_PATH}" \
  SUITE="${SUITE}" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  MAX_VIDEOS_TO_SAVE="${MAX_VIDEOS_TO_SAVE}" \
  ASSIGNMENT_JSON="${ASSIGNMENT_JSON}" \
  "${STAR_PYTHON}" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

repo_root = os.environ["REPO_ROOT"]
libero_python = os.environ["LIBERO_PYTHON"]
libero_home = os.environ["LIBERO_HOME"]
config_dir = os.environ["CONFIG_DIR"]
ckpt_path = os.environ["CKPT_PATH"]
suite = os.environ["SUITE"]
num_trials = int(os.environ["NUM_TRIALS_PER_TASK"])
max_videos = int(os.environ["MAX_VIDEOS_TO_SAVE"])
assignment = json.load(open(os.environ["ASSIGNMENT_JSON"], "r", encoding="utf-8"))

for worker in assignment["workers"]:
    worker_dir = Path(worker["worker_dir"])
    worker_dir.mkdir(parents=True, exist_ok=True)
    task_orders = ",".join(str(x) for x in worker["task_orders"])
    if not task_orders:
        continue
    log_path = worker_dir / "eval.log"
    xvfb_path = worker_dir / "xvfb.log"
    pid_path = worker_dir / "eval.pid"
    video_dir = worker_dir / suite
    metrics_log_dir = worker_dir / "logs"
    cmd = (
        f"cd {repo_root} && "
        f"exec nohup env -u DEBUG "
        f"LIBERO_HOME={libero_home} "
        f"LIBERO_CONFIG_PATH={config_dir} "
        f"PYTHONPATH={repo_root}:{libero_home} "
        f"MUJOCO_GL=glx "
        f"LIBGL_ALWAYS_SOFTWARE=1 "
        f"xvfb-run -a -e {xvfb_path} -s '-screen 0 1024x768x24 +extension GLX +render -noreset' "
        f"{libero_python} examples/LIBERO-plus/eval_files/eval_libero.py "
        f"--args.pretrained-path {ckpt_path} "
        f"--args.host 127.0.0.1 "
        f"--args.port {worker['port']} "
        f"--args.task-suite-name {suite} "
        f"--args.num-trials-per-task {num_trials} "
        f"--args.video-out-path {video_dir} "
        f"--args.log-path {metrics_log_dir} "
        f"--args.max-videos-to-save {max_videos} "
        f"--args.task-orders {task_orders} "
        f"> {log_path} 2>&1 < /dev/null"
    )
    proc = subprocess.Popen(["bash", "-lc", cmd], start_new_session=True)
    pid = str(proc.pid)
    pid_path.write_text(pid + "\n", encoding="utf-8")
    print(f"launched {worker['worker_name']} pid={pid} tasks={len(worker['task_orders'])}")
PY
}

status_run() {
  ASSIGNMENT_JSON="${ASSIGNMENT_JSON}" \
  "${STAR_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

assignment = json.load(open(os.environ["ASSIGNMENT_JSON"], "r", encoding="utf-8"))
done = 0
for worker in assignment["workers"]:
    metrics = Path(worker["worker_dir"]) / assignment["suite"] / "metrics.json"
    if metrics.exists():
        done += 1
print(f"workers_done={done}/{len(assignment['workers'])}")
PY
}

aggregate_run() {
  REPO_ROOT="${REPO_ROOT}" \
  ASSIGNMENT_JSON="${ASSIGNMENT_JSON}" \
  AGGREGATED_JSON="${AGGREGATED_JSON}" \
  "${STAR_PYTHON}" - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

repo_root = os.environ["REPO_ROOT"]
assignment = json.load(open(os.environ["ASSIGNMENT_JSON"], "r", encoding="utf-8"))
metrics_paths = []
for worker in assignment["workers"]:
    metrics = Path(worker["worker_dir"]) / assignment["suite"] / "metrics.json"
    if worker["task_orders"] and not metrics.exists():
        raise FileNotFoundError(f"Missing worker metrics: {metrics}")
    if metrics.exists():
        metrics_paths.append(str(metrics))

output_path = Path(os.environ["AGGREGATED_JSON"])
output_path.parent.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable,
    f"{repo_root}/examples/LIBERO-plus/eval_files/aggregate_parallel_metrics.py",
    "--inputs",
    *metrics_paths,
    "--output",
    str(output_path),
]
subprocess.check_call(cmd)
print(output_path)
PY
}

stop_run() {
  ASSIGNMENT_JSON="${ASSIGNMENT_JSON}" \
  "${STAR_PYTHON}" - <<'PY'
import json
import os
import signal
from pathlib import Path

assignment = json.load(open(os.environ["ASSIGNMENT_JSON"], "r", encoding="utf-8"))
pid_files = []
for server in assignment["servers"]:
    pid_files.append(Path(server["server_dir"]) / "server.pid")
for worker in assignment["workers"]:
    pid_files.append(Path(worker["worker_dir"]) / "eval.pid")

for pid_file in pid_files:
    if not pid_file.exists():
        continue
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
        print(f"stopped {pid}")
    except ProcessLookupError:
        pass
PY
}

prepare_dirs
write_libero_config

case "${MODE}" in
  launch)
    generate_assignment
    launch_servers
    launch_workers
    echo "launched run at ${OUTPUT_ROOT}"
    ;;
  status)
    status_run
    ;;
  aggregate)
    aggregate_run
    ;;
  stop)
    stop_run
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
