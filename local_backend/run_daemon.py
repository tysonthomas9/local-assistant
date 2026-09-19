"""Run reachy-mini-daemon with its WebRTC signalling server bound to 127.0.0.1.

The daemon's media server creates a GStreamer `webrtcsink` with `run-signalling-server=True`
and leaves `signalling-server-host` at its default of 0.0.0.0, so anyone on the LAN can reach
port 8443 and negotiate the camera + microphone stream. Local SDK clients use
ws://127.0.0.1:8443 (reachy_mini/media/central_signaling_relay.py), so loopback is enough on
this USB-tethered Lite robot. Set REACHY_SIGNALLING_HOST=0.0.0.0 to restore LAN access.

Upstream code is unchanged: this wraps GstMediaServer._configure_webrtc, then calls the
normal daemon entry point with the same arguments.
"""

import os
import sys

from reachy_mini.media import media_server

HOST = os.environ.get("REACHY_SIGNALLING_HOST", "127.0.0.1")
_configure_webrtc = media_server.GstMediaServer._configure_webrtc


def _configure_webrtc_on_host(self, pipeline):
    sink = _configure_webrtc(self, pipeline)
    sink.set_property("signalling-server-host", HOST)
    return sink


media_server.GstMediaServer._configure_webrtc = _configure_webrtc_on_host

from reachy_mini.daemon.app.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
