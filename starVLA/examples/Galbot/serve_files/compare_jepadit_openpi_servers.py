"""Compare JEPA and OpenPI Galbot websocket server responses on one observation.

This is a protocol / output-compatibility check, not a quality benchmark.
It compares:
  - metadata keys / values
  - response top-level keys
  - actions shape / dtype / finiteness
  - action numeric differences when shapes match
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


STARVLA_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = STARVLA_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STARVLA_ROOT))

from gal_real.model_agent.openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402


DEFAULT_PROMPT = "Galbot_G1_stack_bowl_1"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jepa-host", default="127.0.0.1")
    parser.add_argument("--jepa-port", type=int, required=True)
    parser.add_argument("--openpi-host", default="127.0.0.1")
    parser.add_argument("--openpi-port", type=int, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-json", default=None, help="Optional JSON file containing a 23-dim state array.")
    parser.add_argument("--dump-json", default=None, help="Optional path to dump comparison result JSON.")
    return parser


def make_fake_obs(prompt: str, image_size: int, state_json: str | None) -> dict[str, Any]:
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    if state_json is not None:
        state = np.asarray(json.loads(Path(state_json).read_text()), dtype=np.float32).reshape(-1)
    else:
        state = np.zeros(23, dtype=np.float32)
        state[14] = 0.10
        state[22] = 0.10

    if state.shape[0] != 23:
        raise ValueError(f"Expected 23-dim state, got {state.shape}")

    return {
        "observation/state": state,
        "observation/image": image,
        "observation/left_wrist_image": image,
        "observation/wrist_image": image,
        "prompt": prompt,
    }


def summarize_array(arr: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(arr).all()),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def compare_metadata(jepa_meta: dict[str, Any], openpi_meta: dict[str, Any]) -> dict[str, Any]:
    jepa_keys = set(jepa_meta.keys())
    openpi_keys = set(openpi_meta.keys())
    shared = sorted(jepa_keys & openpi_keys)

    differing_values = {}
    for key in shared:
        jv = jepa_meta[key]
        ov = openpi_meta[key]
        if jv != ov:
            differing_values[key] = {"jepa": jv, "openpi": ov}

    return {
        "jepa_only_keys": sorted(jepa_keys - openpi_keys),
        "openpi_only_keys": sorted(openpi_keys - jepa_keys),
        "differing_values": differing_values,
    }


def compare_responses(jepa_resp: dict[str, Any], openpi_resp: dict[str, Any]) -> dict[str, Any]:
    jepa_keys = set(jepa_resp.keys())
    openpi_keys = set(openpi_resp.keys())
    result: dict[str, Any] = {
        "jepa_only_keys": sorted(jepa_keys - openpi_keys),
        "openpi_only_keys": sorted(openpi_keys - jepa_keys),
    }

    if "actions" not in jepa_resp or "actions" not in openpi_resp:
        result["actions_present"] = {"jepa": "actions" in jepa_resp, "openpi": "actions" in openpi_resp}
        return result

    jepa_actions = np.asarray(jepa_resp["actions"], dtype=np.float32)
    openpi_actions = np.asarray(openpi_resp["actions"], dtype=np.float32)
    result["jepa_actions"] = summarize_array(jepa_actions)
    result["openpi_actions"] = summarize_array(openpi_actions)

    if jepa_actions.shape == openpi_actions.shape:
        diff = jepa_actions - openpi_actions
        result["action_diff"] = {
            "max_abs": float(np.max(np.abs(diff))),
            "mean_abs": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        }
        result["first_jepa_action"] = jepa_actions[0].tolist() if jepa_actions.ndim >= 2 else jepa_actions.tolist()
        result["first_openpi_action"] = (
            openpi_actions[0].tolist() if openpi_actions.ndim >= 2 else openpi_actions.tolist()
        )
    else:
        result["action_shape_match"] = False

    return result


def main() -> None:
    args = build_argparser().parse_args()
    obs = make_fake_obs(args.prompt, args.image_size, args.state_json)

    jepa_client = WebsocketClientPolicy(host=args.jepa_host, port=args.jepa_port)
    openpi_client = WebsocketClientPolicy(host=args.openpi_host, port=args.openpi_port)

    jepa_meta = jepa_client.get_server_metadata()
    openpi_meta = openpi_client.get_server_metadata()
    jepa_resp = jepa_client.infer(obs)
    openpi_resp = openpi_client.infer(obs)

    result = {
        "request": {
            "prompt": args.prompt,
            "image_size": args.image_size,
            "state": np.asarray(obs["observation/state"], dtype=np.float32).tolist(),
        },
        "metadata_compare": compare_metadata(jepa_meta, openpi_meta),
        "response_compare": compare_responses(jepa_resp, openpi_resp),
        "jepa_metadata": jepa_meta,
        "openpi_metadata": openpi_meta,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
