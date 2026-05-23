"""OpenPI-compatible websocket server for Galbot QwenGR00T policies."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

GAL_REAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GAL_REAL_ROOT.parent
STARVLA_ROOT = REPO_ROOT / "starVLA"
sys.path.insert(0, str(GAL_REAL_ROOT))
sys.path.insert(0, str(STARVLA_ROOT))

from gal_real.server.serve_jepadit_policy import OpenPICompatiblePolicyServer  # noqa: E402
from examples.Galbot.serve_files.galbot_qwengroot_policy import QwenGR00TGalbotPolicy  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True, help="Path to a QwenGR00T pytorch_model.pt checkpoint.")
    parser.add_argument("--stats-path", default=None, help="Optional dataset_statistics.json path.")
    parser.add_argument("--default-prompt", default=None, help="Prompt used when obs does not contain 'prompt'.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6688)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    policy = QwenGR00TGalbotPolicy(
        args.ckpt_path,
        stats_path=args.stats_path,
        device=args.device,
        use_bf16=not args.no_bf16,
        default_prompt=args.default_prompt,
    )

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "protocol": "openpi_websocket_direct",
            "server": "gal_real_qwengroot",
        }
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    logging.info("Serving QwenGR00T Galbot policy on %s:%s (%s, %s)", args.host, args.port, hostname, local_ip)

    OpenPICompatiblePolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
