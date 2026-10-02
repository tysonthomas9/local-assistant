"""`python -m assistant_brain` / `assistant-brain`: run the brain.

    python -m assistant_brain --config-dir config --profile ci --engine basic

It serves EdgeLink (`[net].edgelink_bind`) and the loopback admin endpoint
(`[net].admin_bind`), routes each edge's input to its DialogManager and answers with the
chosen turn engine: `echo` (the basic engine's echo mode, no models) or `basic` (text replies
from the LLM at `[llm].base_url` through the priority gate). `--host`/`--port` and
`--admin-port` override the binds (port 0 picks a free port); `--set section.key=value`
overrides any config key. The first line printed is
`LISTENING url=ws://... admin=http://... engine=...`; the event lines that follow are listed
in `assistant_brain.console`. Logs go to stderr.
"""

import argparse
import asyncio
import contextlib
import os
import signal
from pathlib import Path

from assistant_brain.adapters.llm import LlmClient
from assistant_brain.adapters.priority_gate import PriorityGate
from assistant_brain.admin import AdminServer
from assistant_brain.bus import EventBus
from assistant_brain.console import emit
from assistant_brain.engine.basic import BasicTurnEngine
from assistant_brain.router import Router
from assistant_brain.sessions import SessionManager
from assistant_brain.turnlog import TurnLog
from assistant_core.config import AssistantConfig, load_config, parse_cli_overrides
from assistant_core.log import configure_logging
from assistant_link.auth import DevTokenVerifier
from assistant_link.console import DEFAULT_DEV_TOKEN, DEV_TOKEN_ENV
from assistant_link.server import LinkServer


def split_bind(bind: str) -> tuple[str, int]:
    host, _, port = bind.rpartition(":")
    return host.strip("[]"), int(port)


async def serve(config: AssistantConfig, config_dir: Path, engine_mode: str, token: str) -> int:
    bus = EventBus()
    turns = TurnLog()
    router = Router(config_dir, config.brain.default_assistant)
    llm: LlmClient | None = None
    if engine_mode == "basic":
        gate = PriorityGate(config.llm.max_concurrency, config.llm.reserved_voice_slots)
        llm = LlmClient(config.llm, gate)
    engine = BasicTurnEngine("echo" if engine_mode == "echo" else "basic", llm)
    sessions = SessionManager(
        bus=bus, router=router, engine=engine, turns=turns, follow_up_s=config.brain.follow_up_s
    )
    host, port = split_bind(config.net.edgelink_bind)
    link = LinkServer(sessions, DevTokenVerifier(token), host=host, port=port)
    admin_host, admin_port = split_bind(config.net.admin_bind)
    admin = AdminServer(
        host=admin_host,
        port=admin_port,
        engine_name=engine.name,
        sessions=sessions,
        turns=turns,
        bus=bus,
        llm=llm,
    )
    await link.start()
    await admin.start()
    llm_fields = {"llm": config.llm.base_url, "model": config.llm.model} if llm else {}
    emit("LISTENING", url=link.url, admin=admin.url, engine=engine.name, **llm_fields)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    serving = asyncio.create_task(link.serve_forever(), name="edgelink")
    waiting = asyncio.create_task(stop.wait(), name="stop")
    await asyncio.wait({serving, waiting}, return_when=asyncio.FIRST_COMPLETED)
    for task in (serving, waiting):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    await admin.close()
    await sessions.close()
    await link.close()
    await engine.aclose()
    if llm is not None:
        await llm.aclose()
    emit("STOPPED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_brain")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--profile", default=None, help="config/profiles/<profile>.toml")
    parser.add_argument(
        "--engine",
        choices=["echo", "basic"],
        default=None,
        help="echo (no models) or basic (LLM text replies); default: [engine].impl",
    )
    parser.add_argument("--host", default=None, help="EdgeLink host (default: [net])")
    parser.add_argument("--port", type=int, default=None, help="EdgeLink port (0: any free)")
    parser.add_argument("--admin-port", type=int, default=None, help="admin port (0: any free)")
    parser.add_argument("--token", default=os.environ.get(DEV_TOKEN_ENV, DEFAULT_DEV_TOKEN))
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args(argv)

    overrides = parse_cli_overrides(args.set)
    config = load_config(args.config_dir, profile=args.profile, cli_overrides=overrides)
    binds: dict[str, str] = {}
    if args.host is not None or args.port is not None:
        host, port = split_bind(config.net.edgelink_bind)
        binds["edgelink_bind"] = f"{args.host or host}:{port if args.port is None else args.port}"
    if args.admin_port is not None:
        binds["admin_bind"] = f"{split_bind(config.net.admin_bind)[0]}:{args.admin_port}"
    if binds:
        overrides = {**overrides, "net": {**overrides.get("net", {}), **binds}}
        config = load_config(args.config_dir, profile=args.profile, cli_overrides=overrides)
    engine_mode = args.engine or config.engine.impl
    if engine_mode == "realtime":
        parser.error("the realtime engine arrives in phase 2; use --engine basic or echo")
    configure_logging(
        service="brain",
        level=config.observability.log_level,
        fmt=config.observability.log_format,
    )
    with contextlib.suppress(KeyboardInterrupt):
        return asyncio.run(serve(config, args.config_dir, engine_mode, args.token))
    return 0
