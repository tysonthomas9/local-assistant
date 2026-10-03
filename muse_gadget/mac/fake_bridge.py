"""A stand-in for the Muse bridge: answers every /turn with "you said ..." (for hardware tests).

    python fake_bridge.py [--port 48080]

Same contract as the real bridge (see muse_bridge.py), bound to 127.0.0.1 only. It logs only
the length of each turn, never its text.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class EchoBridge(BaseHTTPRequestHandler):
    server_version = "fake-muse-bridge"

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"paired": True, "linked": True, "fake": True})
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        if path != "/turn":
            self._send(404, {"error": "not_found"})
            return
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            self._send(415, {"error": "unsupported_media_type"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
        except (ValueError, AttributeError):
            self._send(400, {"error": "bad_request"})
            return
        if not isinstance(text, str) or not text.strip():
            self._send(400, {"error": "bad_request"})
            return
        sys.stderr.write(f"fake-bridge: turn of {len(text)} chars\n")
        reply = f"You said: {text.strip()}"
        if "stream=1" in query.split("&"):   # one sentence, then the end
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"text": reply}).encode() + b"\n" + b'{"done": true}\n')
            self.close_connection = True
            return
        self._send(200, {"reply": reply})

    def log_message(self, format: str, *args: object) -> None:  # quiet: no request lines
        pass


def serve(port: int = 48080) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), EchoBridge)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=48080)
    args = parser.parse_args()
    server = serve(args.port)
    print(f"fake-bridge: listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
