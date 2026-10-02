"""Run the reachy-mini daemon WITH media, every socket of it on loopback only.

    python -m assistant_robot_reachy.daemon [reachy-mini-daemon options...]

Upstream defaults that would reach the LAN, and what this launcher changes (upstream code is
unchanged; two of its functions are wrapped before its normal entry point runs):

- the media server's WebRTC signalling server (`webrtcsink`, port 8443) listens on 0.0.0.0:
  its `signalling-server-host` is set to 127.0.0.1 (local SDK clients use ws://127.0.0.1:8443);
- the daemon announces itself over mDNS (UDP 5353 on every interface): the registration is
  skipped, since nothing on the LAN may use this daemon;
- the HTTP API: pass `--fastapi-host 127.0.0.1` (the default for the Lite anyway).

On a macOS edge host the tests start it inside "Reachy Edge.app" (scripts/edge_app_run.sh),
the process macOS holds responsible for its camera and microphone use.
"""

import sys
from typing import Any

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


def main() -> int:
    _patch_loopback()
    from reachy_mini.daemon.app.main import main as daemon_main

    sys.argv[0] = "reachy-mini-daemon"
    daemon_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
