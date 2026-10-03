"""``python -m gadget run|pair|info``.

* ``run``: keep the gadget linked to Muse and serve the ``/turn`` bridge, in
  one process.
* ``pair``: the SDK's own ``musegadget pair`` (Bluetooth LE setup with the
  Muse app). Needs the host's BlueZ over D-Bus; used only on the PC.
* ``info``: the SDK's ``musegadget info`` (node id, BLE name, paired or not).

State lives in ``$MUSEGADGET_STATE_DIR`` (a mounted volume, never the image).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from musegadget import cli as upstream_cli
from musegadget import config, identity
from musegadget.executor import Account

from gadget import bridge, chat, restrict
from gadget.link import RobotService

log = logging.getLogger("gadget")


def turn_options() -> chat.TurnOptions:
    session_id = os.environ.get("MUSE_SESSION_ID", chat.DEFAULT_SESSION_ID)
    style_hint = chat.STYLE_NOTE if os.environ.get("MUSE_STYLE_HINT_ON") == "1" else chat.DEFAULT_STYLE_HINT
    return chat.TurnOptions(session_id=session_id.strip() or None, style_hint=style_hint)


def current_account() -> Account:
    """This process's account; a uid without a passwd entry (``--user``) still works."""
    try:
        return Account.current()
    except KeyError:
        return Account("muse", os.getuid(), os.getgid(), str(config.state_dir()))


def is_paired() -> bool:
    return config.load_json(config.PAIRING_FILE) is not None


async def run(host: str, port: int) -> None:
    try:
        sdk_token = config.sdk_token()
    except ValueError:
        log.warning("the saved SDK token is not valid; running without it")
        sdk_token = None
    service = RobotService(
        identity=identity.load_or_create(),
        executor=restrict.RestrictedExecutor(current_account()),
        sdk_token=sdk_token,
    )
    web = bridge.Bridge(lambda: service.link, is_paired, turn_options())
    server = await web.serve(host, port)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, service.stop)
    log.info("gadget %s up as %r; commands: %s", "paired" if is_paired() else "not paired",
             service.display_name, ", ".join(sorted(restrict.command_specs())))
    try:
        await service.run()
    finally:
        server.close()
        await server.wait_closed()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gadget")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="link to Muse and serve the /turn bridge")
    run_p.add_argument("--host", default=os.environ.get("MUSE_BRIDGE_HOST", bridge.DEFAULT_HOST))
    run_p.add_argument("--port", type=int,
                       default=int(os.environ.get("MUSE_BRIDGE_PORT", bridge.DEFAULT_PORT)))
    pair_p = sub.add_parser("pair", help="Bluetooth LE setup with the Muse app (PC only)")
    pair_p.add_argument("--timeout", type=int, default=upstream_cli.DEFAULT_SETUP_WINDOW_S)
    sub.add_parser("info", help="node id, BLE name, paired or not")
    args = parser.parse_args(argv)

    if args.command == "pair":
        return upstream_cli.main(["pair", "--timeout", str(args.timeout)])
    if args.command == "info":
        return upstream_cli.main(["info"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(args.host, args.port))
    except ValueError as exc:
        print(f"gadget: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
