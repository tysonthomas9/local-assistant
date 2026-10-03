"""Run the reachy-mini daemon WITH media, every socket of it on loopback only.

    python -m assistant_robot_reachy.daemon [reachy-mini-daemon options...]

Upstream defaults that would reach the LAN, and what this launcher changes (upstream code is
unchanged; two of its functions are wrapped before its normal entry point runs):

- the media server's WebRTC signalling server (`webrtcsink`, port 8443) listens on 0.0.0.0:
  its `signalling-server-host` is set to 127.0.0.1 (local SDK clients use ws://127.0.0.1:8443);
- the daemon announces itself over mDNS (UDP 5353 on every interface): the registration is
  skipped, since nothing on the LAN may use this daemon;
- the HTTP API: pass `--fastapi-host 127.0.0.1` (the default for the Lite anyway).

It also keeps the robot safe when nothing looks after it (two more wraps, upstream unchanged):

- the motor watchdog (`assistant_robot_reachy.watchdog`) runs in this process: when the edge
  agent's heartbeat stops (the agent died, froze or stopped, or its link dropped), a robot
  whose motors are on goes to rest (`goto_sleep`) and its motors are turned off;
- stopping the daemon (SIGTERM, SIGINT, and SIGHUP too, as when the SSH session that started
  it drops) puts a robot whose motors are on to rest first (upstream's `reset_to_sleep`,
  which ends with the motors off), even with `--no-goto-sleep-on-stop`; a robot already at
  rest (motors off) is not moved.

On a macOS edge host the tests start it inside "Reachy Edge.app" (scripts/edge_app_run.sh),
the process macOS holds responsible for its camera and microphone use.
"""

import signal
import sys
from typing import Any

from assistant_robot_reachy import watchdog

SIGNALLING_HOST = "127.0.0.1"


def _patch_loopback() -> None:
    from reachy_mini.daemon.app import main as daemon_main
    from reachy_mini.media import media_server

    configure = media_server.GstMediaServer._configure_webrtc

    def configure_on_loopback(self: Any, pipeline: Any) -> Any:
        sink = configure(self, pipeline)
        sink.set_property("signalling-server-host", SIGNALLING_HOST)
        return sink

    media_server.GstMediaServer._configure_webrtc = configure_on_loopback

    class NoMdns:
        """Skips the LAN announcement (the daemon is for this machine only)."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def register(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def unregister(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __getattr__(self, name: str) -> object:
            return lambda *args, **kwargs: None

    daemon_main.MdnsServiceRegistration = NoMdns


def _patch_rest_on_stop() -> None:
    from reachy_mini.daemon.daemon import Daemon
    from reachy_mini.io.protocol import MotorControlMode

    stop = Daemon.stop

    async def stop_at_rest(self: Any, goto_sleep_on_stop: bool = True) -> Any:
        try:
            on = self.backend.get_motor_control_mode() != MotorControlMode.Disabled
        except Exception:
            on = False  # no backend (or no robot): nothing to rest
        if on and not goto_sleep_on_stop:
            print("WATCHDOG daemon-stop motors=on: the robot goes to rest first", flush=True)
        return await stop(self, goto_sleep_on_stop=goto_sleep_on_stop or on)

    Daemon.stop = stop_at_rest


def _api_base(argv: list[str]) -> str:
    port = "8000"
    for i, arg in enumerate(argv):
        if arg == "--fastapi-port" and i + 1 < len(argv):
            port = argv[i + 1]
        elif arg.startswith("--fastapi-port="):
            port = arg.partition("=")[2]
    return f"http://127.0.0.1:{port}"


def main() -> int:
    _patch_loopback()
    _patch_rest_on_stop()
    from reachy_mini.daemon.app.main import main as daemon_main

    # SIGHUP (the SSH session gone) stops the daemon the way SIGTERM does: cleanly, at rest.
    signal.signal(signal.SIGHUP, lambda *_: signal.raise_signal(signal.SIGTERM))
    watchdog.start(_api_base(sys.argv[1:]))
    sys.argv[0] = "reachy-mini-daemon"
    daemon_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
