"""Loopback HTTP bridge between the robot's conversation app and Muse.

* ``POST /turn`` with JSON ``{"text": "..."}`` -> 200 ``{"reply": "..."}``.
  Errors: 415 for a body that isn't JSON (audio included: speech-to-text is
  local), 400 for JSON without text, 503 ``{"error": "not_paired"}`` or
  ``{"error": "link_down"}``, 504 ``{"error": "timeout"}`` after 60 s with no
  reply text (with some text, the text so far comes back as a 200), 502
  ``{"error": "muse_error"}`` if Muse refuses the turn.
* ``POST /turn?stream=1`` (same body) streams the reply as NDJSON while Muse writes it:
  one ``{"text": "<sentence>"}`` line per sentence as soon as it ends, then
  ``{"done": true}`` (or ``{"done": true, "error": "timeout"}`` when the turn ran out
  after some text). An error before any sentence gets the same status and JSON body as
  the one-shot form.
* ``GET /health`` -> 200 ``{"paired": bool, "linked": bool}``.

The bridge listens on loopback only. Inside a container, where the published
port must reach the container's own interface, it may bind ``0.0.0.0`` only
when ``MUSE_BRIDGE_IN_CONTAINER=1``; the launcher then publishes the port on
the host's ``127.0.0.1`` only. Turns run one at a time.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
from typing import Callable

from gadget import chat

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 48080
MAX_BODY = 16 * 1024
MAX_TEXT = 4000
HEADER_TIMEOUT_S = 10
# chat.turn has its own deadline (TurnOptions.timeout_s) and returns the text received so far
# when it hits it. This outer guard only catches a turn that hangs past that, so it must be later.
TURN_MARGIN_S = 5.0
IN_CONTAINER_ENV = "MUSE_BRIDGE_IN_CONTAINER"
_REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed",
            413: "Payload Too Large", 415: "Unsupported Media Type", 502: "Bad Gateway",
            503: "Service Unavailable", 504: "Gateway Timeout"}


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind_host(host: str, in_container: bool | None = None) -> None:
    """Refuse any non-loopback address, except 0.0.0.0 inside a container."""
    if in_container is None:
        in_container = os.environ.get(IN_CONTAINER_ENV) == "1"
    if is_loopback(host) or (in_container and host == "0.0.0.0"):
        return
    raise ValueError(f"the bridge listens on loopback only, not {host!r}")


class Bridge:
    def __init__(
        self,
        get_link: Callable[[], chat.Link | None],
        is_paired: Callable[[], bool],
        options: chat.TurnOptions = chat.TurnOptions(),
    ) -> None:
        self._get_link = get_link
        self._is_paired = is_paired
        self._options = options
        self._turn_lock = asyncio.Lock()

    async def serve(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                    in_container: bool | None = None) -> asyncio.AbstractServer:
        check_bind_host(host, in_container)
        server = await asyncio.start_server(self._handle, host, port, limit=MAX_BODY)
        log.info("bridge listening on %s:%d", host, port)
        return server

    async def handle(self, method: str, path: str, headers: dict, body: bytes,
                     emit: chat.OnSentence | None = None) -> tuple[int, dict]:
        """Route one request; returns ``(status, JSON body)``.

        With ``emit`` and ``?stream=1``, reply sentences go to ``emit`` as they arrive."""
        path, _, query = path.partition("?")
        stream = emit is not None and "stream=1" in query.split("&")
        if path == "/health":
            if method != "GET":
                return 405, {"error": "method_not_allowed"}
            return 200, {"paired": self._is_paired(), "linked": self._get_link() is not None}
        if path != "/turn":
            return 404, {"error": "not_found"}
        if method != "POST":
            return 405, {"error": "method_not_allowed"}
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return 415, {"error": "unsupported_media_type"}
        try:
            request = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return 415, {"error": "unsupported_media_type"}
        text = request.get("text") if isinstance(request, dict) else None
        if not isinstance(text, str) or not text.strip():
            return 400, {"error": "bad_request"}
        if len(text) > MAX_TEXT:
            return 413, {"error": "too_long"}
        if not self._is_paired():
            return 503, {"error": "not_paired"}
        try:
            reply = await asyncio.wait_for(self._turn(text, emit if stream else None),
                                           self._options.timeout_s + TURN_MARGIN_S)
        except asyncio.TimeoutError:
            return 504, {"error": "timeout"}
        except chat.TurnError as exc:
            return exc.status, {"error": exc.code}
        except (ConnectionError, OSError) as exc:
            log.warning("turn failed: %s", type(exc).__name__)
            return 503, {"error": "link_down"}
        return 200, {"reply": reply}

    async def _turn(self, text: str, on_sentence: chat.OnSentence | None = None) -> str:
        async with self._turn_lock:
            link = self._get_link()
            if link is None:
                raise chat.TurnError("link_down", 503)
            return await chat.turn(link, text, self._options, on_sentence=on_sentence)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, payload = 400, {"error": "bad_request"}
        streaming = False

        async def emit(sentence: str) -> None:   # ?stream=1: headers on the first sentence
            nonlocal streaming
            if not streaming:
                streaming = True
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
                             b"Cache-Control: no-store\r\nConnection: close\r\n\r\n")
            writer.write(json.dumps({"text": sentence}).encode() + b"\n")
            await writer.drain()

        try:
            method, path, headers, body = await asyncio.wait_for(_read_request(reader), HEADER_TIMEOUT_S)
            status, payload = await self.handle(method, path, headers, body, emit)
        except _TooLarge:
            status, payload = 413, {"error": "too_long"}
        except (ValueError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            pass
        except Exception:
            log.exception("bridge request failed")
            status, payload = 502, {"error": "muse_error"}
        if streaming:   # the reply so far went out as sentences: just end the stream
            end = {"done": True}
            if status != 200 and isinstance(payload.get("error"), str):
                end["error"] = payload["error"]
            try:
                writer.write(json.dumps(end).encode() + b"\n")
                await writer.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                writer.close()
            return
        data = json.dumps(payload).encode()
        head = (f"HTTP/1.1 {status} {_REASONS.get(status, 'Error')}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n")
        try:
            writer.write(head.encode() + data)
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()


class _TooLarge(Exception):
    pass


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict, bytes]:
    request_line = (await reader.readline()).decode("latin-1").strip()
    parts = request_line.split()
    if len(parts) != 3 or not parts[2].startswith("HTTP/"):
        raise ValueError("bad request line")
    headers: dict = {}
    while True:
        line = (await reader.readline()).decode("latin-1")
        if line in ("\r\n", "\n", ""):
            break
        name, sep, value = line.partition(":")
        if not sep:
            raise ValueError("bad header")
        headers[name.strip().lower()] = value.strip()
        if len(headers) > 50:
            raise ValueError("too many headers")
    length = int(headers.get("content-length") or 0)
    if length < 0:
        raise ValueError("bad length")
    if length > MAX_BODY:
        raise _TooLarge()
    body = await reader.readexactly(length) if length else b""
    return parts[0].upper(), parts[1], headers, body
