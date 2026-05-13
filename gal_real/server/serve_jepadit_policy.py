"""OpenPI-compatible websocket server for Galbot JEPADiT policies.

This server intentionally matches the protocol used by gal_real's
WebsocketClientPolicy: send metadata once on connection, then return the
policy output dict directly for each observation.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
from pathlib import Path
import socket
import sys
import time
import traceback

import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames


GAL_REAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = GAL_REAL_ROOT.parent
STARVLA_ROOT = REPO_ROOT / "starVLA"

sys.path.insert(0, str(GAL_REAL_ROOT))
sys.path.insert(0, str(STARVLA_ROOT))

from model_agent.openpi_client import msgpack_numpy  # noqa: E402
from examples.Galbot.serve_files.galbot_jepadit_policy import JEPADiTGalbotPolicy  # noqa: E402


LOGGER = logging.getLogger(__name__)


class OpenPICompatiblePolicyServer:
    def __init__(
        self,
        policy: JEPADiTGalbotPolicy,
        *,
        host: str = "0.0.0.0",
        port: int = 6688,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        LOGGER.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_start = time.monotonic()
                result = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                result["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    result["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(result))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                LOGGER.exception("JEPADiT inference server error")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-path", required=True, help="Path to a JEPADiT pytorch_model.pt checkpoint.")
    parser.add_argument("--stats-path", default=None, help="Optional dataset_statistics.json path.")
    parser.add_argument("--default-prompt", default=None, help="Prompt used when obs does not contain 'prompt'.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6688)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-bf16", action="store_true")
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

    metadata = dict(policy.metadata)
    metadata.update(
        {
            "protocol": "openpi_websocket_direct",
            "server": "gal_real_jepadit",
        }
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    LOGGER.info("Serving JEPADiT Galbot policy on %s:%s (%s, %s)", args.host, args.port, hostname, local_ip)

    OpenPICompatiblePolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
