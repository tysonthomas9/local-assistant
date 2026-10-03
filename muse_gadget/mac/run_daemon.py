"""Run reachy-mini-daemon (pinned, unmodified) with every socket on loopback.

    python run_daemon.py --fastapi-host 127.0.0.1 [daemon options...]

Upstream defaults that reach the LAN, and what is changed (upstream functions are wrapped,
not edited; same approach as local_backend/run_daemon.py):
- the WebRTC signalling server (`webrtcsink`, port 8443) listens on 0.0.0.0: set to 127.0.0.1;
- the daemon announces itself over mDNS: the registration is skipped.
The HTTP API is put on 127.0.0.1 by passing --fastapi-host 127.0.0.1 (run_poc.sh does).
"""

from __future__ import annotations

import signal
import sys
from typing import Any


def _patch_loopback() -> None:
    from reachy_mini.daemon.app import main as daemon_main
    from reachy_mini.media import media_server

    configure = media_server.GstMediaServer._configure_webrtc

    def configure_on_loopback(self: Any, pipeline: Any) -> Any:
        sink = configure(self, pipeline)
        sink.set_property("signalling-server-host", "127.0.0.1")
        return sink

    media_server.GstMediaServer._configure_webrtc = configure_on_loopback

    class NoMdns:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __getattr__(self, name: str) -> object:
            return lambda *args, **kwargs: None

    if hasattr(daemon_main, "MdnsServiceRegistration"):
        daemon_main.MdnsServiceRegistration = NoMdns


def main() -> int:
    _patch_loopback()
    from reachy_mini.daemon.app.main import main as daemon_main

    # SIGHUP (the SSH session gone) stops the daemon like SIGTERM: cleanly, robot to sleep.
    signal.signal(signal.SIGHUP, lambda *_: signal.raise_signal(signal.SIGTERM))
    sys.argv[0] = "reachy-mini-daemon"
    daemon_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
