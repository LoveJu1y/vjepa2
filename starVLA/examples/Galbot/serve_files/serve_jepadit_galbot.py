"""Serve a JEPADiT Galbot checkpoint with the OpenPI websocket protocol."""

from __future__ import annotations

import argparse
import logging
import socket

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from examples.Galbot.serve_files.galbot_jepadit_policy import JEPADiTGalbotPolicy


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True, help="Path to steps_*.pt or final_model/pytorch_model.pt")
    parser.add_argument("--stats-path", default=None, help="Optional dataset_statistics.json path")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--port", type=int, default=6688)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--idle-timeout", type=int, default=-1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    policy = JEPADiTGalbotPolicy(
        args.ckpt_path,
        stats_path=args.stats_path,
        device=args.device,
        use_bf16=not args.no_bf16,
        default_prompt=args.default_prompt,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating JEPADiT Galbot server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
