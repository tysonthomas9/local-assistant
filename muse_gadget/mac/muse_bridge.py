"""Client for the Muse gadget bridge on the Mac's loopback (contract in the PR context note).

POST /turn {"text": ...} -> 200 {"reply": ...}; 503 {"error": "not_paired"|"link_down"};
415 for non-JSON; 504 when Muse takes longer than 60 s. GET /health -> {"paired", "linked"}.
POST /turn?stream=1 -> NDJSON {"text": sentence} lines as Muse writes, then {"done": true[, "error"]}.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

DEFAULT_URL = "http://127.0.0.1:48080"


class BridgeError(Exception):
    """The bridge didn't produce a reply; `code` is a short reason (not_paired, link_down, ...)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class BridgeClient:
    def __init__(self, base_url: str | None = None, timeout_s: float = 70.0) -> None:
        self.base_url = (base_url or os.environ.get("MUSE_BRIDGE_URL") or DEFAULT_URL).rstrip("/")
        self.timeout_s = timeout_s
        self._stream_lock = threading.Lock()
        self._stream_sock: Optional[socket.socket] = None
        self._dropped = False

    def turn(self, text: str) -> str:
        """Send one user turn; return Muse's reply text (blocking; call it in a thread)."""
        data = self._post_turn(text, "/turn", lambda resp: json.loads(resp.read() or b"{}"))
        reply = data.get("reply") if isinstance(data, dict) else None
        if not isinstance(reply, str):
            raise BridgeError("bad_reply")
        return reply

    def turn_stream(self, text: str, on_sentence: Callable[[str], None]) -> str:
        """Like turn(), but call on_sentence with each sentence as soon as Muse has written it.

        Returns the whole reply. Raises BridgeError only if nothing was handed over; a turn
        that runs out after some text just ends (the text so far is the reply). drop_stream()
        (from another thread) ends it early: what was handed over so far is returned."""
        sentences: list[str] = []
        error = ""
        url = urllib.parse.urlsplit(self.base_url)
        conn = http.client.HTTPConnection(url.hostname or "127.0.0.1", url.port or 80, timeout=self.timeout_s)
        with self._stream_lock:
            self._dropped = False
        try:
            try:
                conn.request("POST", (url.path or "") + "/turn?stream=1", json.dumps({"text": text}).encode(),
                             {"Content-Type": "application/json"})
                with self._stream_lock:
                    self._stream_sock = conn.sock
                    if self._dropped:  # dropped while connecting: hang up so the bridge stops the turn
                        conn.sock.shutdown(socket.SHUT_RDWR)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                raise BridgeError("dropped" if self._dropped else "unreachable", type(e).__name__) from None
            if resp.status != 200:
                try:
                    code = str(json.loads(resp.read() or b"{}").get("error") or resp.status)
                except Exception:
                    code = str(resp.status)
                raise BridgeError("timeout" if resp.status == 504 else code, f"HTTP {resp.status}")
            try:
                for raw in resp:
                    try:
                        line = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(line, dict):
                        continue
                    piece = line.get("text") if "text" in line else line.get("reply")   # one-shot form
                    if isinstance(piece, str) and piece.strip():
                        sentences.append(piece.strip())
                        on_sentence(piece.strip())
                    if isinstance(line.get("error"), str):
                        error = line["error"]
            except (OSError, http.client.HTTPException, ValueError):
                if not self._dropped:
                    error = error or "unreachable"
        finally:
            with self._stream_lock:
                self._stream_sock = None
            conn.close()
        if error and not sentences:
            raise BridgeError(error)
        return " ".join(sentences)

    def drop_stream(self) -> None:
        """End the turn_stream() in progress (a barge-in): the bridge sees us leave and stops the turn."""
        with self._stream_lock:
            sock, self._dropped = self._stream_sock, True
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _post_turn(self, text: str, path: str, read):
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            self.base_url + path, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return read(resp)
        except urllib.error.HTTPError as e:
            try:
                code = str(json.loads(e.read() or b"{}").get("error") or e.code)
            except Exception:
                code = str(e.code)
            if e.code == 504:
                code = "timeout"
            raise BridgeError(code, f"HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise BridgeError("unreachable", str(getattr(e, "reason", e))) from None

    def health(self) -> dict:
        with urllib.request.urlopen(self.base_url + "/health", timeout=5) as resp:
            return json.loads(resp.read() or b"{}")


SPOKEN_ERRORS = {
    "not_paired": "Muse isn't paired with me yet.",
    "link_down": "I can't reach Muse right now.",
    "timeout": "Muse is taking too long to answer.",
    "unreachable": "The Muse bridge isn't running.",
}


def spoken_error(err: BridgeError) -> str:
    return SPOKEN_ERRORS.get(err.code, "Something went wrong talking to Muse.")
