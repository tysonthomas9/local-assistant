"""Client for the Muse gadget bridge on the Mac's loopback (contract in the PR context note).

POST /turn {"text": ...} -> 200 {"reply": ...}; 503 {"error": "not_paired"|"link_down"};
415 for non-JSON; 504 when Muse takes longer than 60 s. GET /health -> {"paired", "linked"}.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

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

    def turn(self, text: str) -> str:
        """Send one user turn; return Muse's reply text (blocking; call it in a thread)."""
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            self.base_url + "/turn", data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read() or b"{}")
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
        reply = data.get("reply") if isinstance(data, dict) else None
        if not isinstance(reply, str):
            raise BridgeError("bad_reply")
        return reply

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
