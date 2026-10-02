"""`python -m assistant_edge` / `assistant-edge`: run the edge agent with one body."""

import argparse
import asyncio
import os

from assistant_edge.agent import AgentOptions, EdgeAgent
from assistant_edge.bodies import available, load_body
from assistant_link.console import DEFAULT_DEV_TOKEN, DEV_TOKEN_ENV
from assistant_link.server import DEFAULT_PORT, EDGE_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_edge")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--body", default="console", help=f"one of: {', '.join(available())}")
    parser.add_argument("--url", default=f"ws://127.0.0.1:{DEFAULT_PORT}{EDGE_PATH}")
    parser.add_argument("--token", default=os.environ.get(DEV_TOKEN_ENV, DEFAULT_DEV_TOKEN))
    parser.add_argument(
        "--energy-trigger-dbfs",
        type=float,
        default=None,
        help="open a mic window when the level stays above this many dBFS (default: off)",
    )
    args = parser.parse_args(argv)
    body = load_body(args.body)
    options = AgentOptions(
        device_id=args.device_id,
        url=args.url,
        token=args.token,
        energy_trigger_dbfs=args.energy_trigger_dbfs,
    )
    return asyncio.run(EdgeAgent(body, options).run())
