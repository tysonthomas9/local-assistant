"""Log non-loopback DNS lookups and connections made from Python code, with the code that made them.

Only Python-level sockets are visible (sys.addaudithook): connections opened by native libraries
(e.g. onnxruntime's C++ telemetry, GStreamer, llama.cpp) are NOT logged. Use
`strace -f -e trace=connect` for those.

Loaded automatically when this directory is on PYTHONPATH (start_local_backend.sh does that with
REACHY_NET_AUDIT=1). Writes to local_backend/logs/net_audit.log.
"""

import os
import sys
import traceback
from datetime import datetime

_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "net_audit.log")
_LOCAL = ("127.", "::1", "localhost", "0.0.0.0", "::ffff:127.")


def _write(kind, detail):
    stack = "".join(traceback.format_stack(limit=25)[:-2])
    with open(_LOG, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} pid={os.getpid()} {kind} {detail}\n{stack}\n")


def _hook(event, args):
    try:
        if event == "socket.getaddrinfo":
            host = args[0]
            if host and not str(host).startswith(_LOCAL):
                _write("DNS", f"{host}:{args[1]}")
        elif event == "socket.connect":
            addr = args[1]
            if isinstance(addr, tuple) and addr and not str(addr[0]).startswith(_LOCAL):
                _write("CONNECT", f"{addr}")
    except Exception:
        pass


sys.addaudithook(_hook)
