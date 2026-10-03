"""Robot-tools endpoint: the Muse gadget calls a few of Pollen's own app tools through it.

MuseHandler serves this inside the app's process, on the Mac's 127.0.0.1 only, and only while
it runs (which is only while run_poc.sh holds the hw-run lock). The gadget's Podman container
reaches it as host.containers.internal (Podman's gvproxy forwards that to the Mac's loopback).

    POST /tool   Authorization: Bearer <run secret>
                 {"tool": "dance", "args": {"move": "simple_nod"}}
    200 {"ok": true|false, "result": {...Pollen tool result...}}
    401 {"error": "unauthorized"}   missing or wrong secret (checked first)
    403 {"error": "tool_not_allowed"}
    503 {"error": "asleep"}         the robot session isn't up (or is going down)

Only the tools in ALLOWED_TOOLS: Pollen's recorded emotions and dances, stopping them, its
move_head directions, the SDK face tracker on/off, and a few status topics. No volume, camera,
sleep, memory or web tools. Calls go through the app's BackgroundToolManager, the same way the
Hugging Face backend dispatches a model's tool call (see MuseHandler._dispatch_robot_tool).

The secret is per run: run_poc.sh writes it to a 0600 file on the Mac (MUSE_ROBOT_TOOLS_SECRET_FILE,
one line `MUSE_ROBOT_TOOLS_SECRET=<hex>`), gives the same file to the gadget container as its
--env-file, and deletes it at cleanup. Without it the endpoint isn't started.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import logging
import os
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 48081
PORT_ENV = "MUSE_ROBOT_TOOLS_PORT"
SECRET_FILE_ENV = "MUSE_ROBOT_TOOLS_SECRET_FILE"
SECRET_KEY = "MUSE_ROBOT_TOOLS_SECRET"
MIN_SECRET_LEN = 32
MAX_BODY = 8 * 1024
HEADER_TIMEOUT_S = 10.0
TOOL_TIMEOUT_S = 10.0

ALLOWED_TOOLS = frozenset({
    "play_emotion", "dance", "stop_dance", "stop_emotion", "move_head", "head_tracking", "robot_status",
})
# No wifi (IP address), account or installed-apps topics: the answer goes to Muse's cloud.
STATUS_TOPICS = frozenset({"name", "software", "imu"})

_REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
            405: "Method Not Allowed", 413: "Payload Too Large", 415: "Unsupported Media Type",
            500: "Internal Server Error", 503: "Service Unavailable", 504: "Gateway Timeout"}

Dispatch = Callable[[str, dict], Awaitable[dict]]


def read_secret(path: str | None = None) -> str | None:
    """The run's secret from MUSE_ROBOT_TOOLS_SECRET_FILE, or None (endpoint off)."""
    path = path if path is not None else os.environ.get(SECRET_FILE_ENV)
    if not path:
        return None
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            for line in f:
                key, sep, value = line.strip().partition("=")
                if sep and key == SECRET_KEY and len(value) >= MIN_SECRET_LEN:
                    return value
    except OSError as e:
        logger.warning("robot tools: can't read the secret file (%s); endpoint off", type(e).__name__)
        return None
    logger.warning("robot tools: no usable secret in the secret file; endpoint off")
    return None


def port_from_env() -> int:
    return int(os.environ.get(PORT_ENV) or DEFAULT_PORT)


def _check_loopback(host: str) -> None:
    try:
        ok = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"robot tools listen on loopback only, not {host!r}")


def _summary(result: Any) -> str:
    """A short, text-free summary of a tool result for the log."""
    if not isinstance(result, dict):
        return type(result).__name__
    if result.get("error"):
        return "error: " + str(result["error"])[:120]
    return json.dumps(result, default=str)[:160]


class RobotToolsServer:
    def __init__(self, secret: str, dispatch: Dispatch, is_awake: Callable[[], bool],
                 timeout_s: float = TOOL_TIMEOUT_S) -> None:
        if not secret or len(secret) < MIN_SECRET_LEN:
            raise ValueError("robot tools need a secret of at least 32 characters")
        self._secret = secret.encode()
        self._dispatch = dispatch
        self._is_awake = is_awake
        self._timeout_s = timeout_s
        self._lock = asyncio.Lock()   # one tool call at a time

    async def serve(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> asyncio.AbstractServer:
        _check_loopback(host)
        server = await asyncio.start_server(self._handle, host, port, limit=MAX_BODY)
        logger.info("robot tools: listening on %s:%d", host, port)
        return server

    def _authorized(self, headers: dict) -> bool:
        scheme, _, token = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), self._secret)

    async def handle(self, method: str, path: str, headers: dict, body: bytes) -> tuple[int, dict]:
        if not self._authorized(headers):
            logger.warning("robot tools: refused a request without the run's secret")
            return 401, {"error": "unauthorized"}
        if path.split("?", 1)[0] != "/tool":
            return 404, {"error": "not_found"}
        if method != "POST":
            return 405, {"error": "method_not_allowed"}
        if headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return 415, {"error": "unsupported_media_type"}
        try:
            request = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return 400, {"error": "bad_request"}
        if not isinstance(request, dict):
            return 400, {"error": "bad_request"}
        tool, args = request.get("tool"), request.get("args", {})
        if not isinstance(args, dict):
            return 400, {"error": "bad_request"}
        if tool not in ALLOWED_TOOLS:
            logger.warning("robot tools: refused tool %r", str(tool)[:40])
            return 403, {"error": "tool_not_allowed"}
        if tool == "robot_status" and args.get("topic") not in STATUS_TOPICS:
            return 403, {"error": "tool_not_allowed"}
        if not self._is_awake():
            logger.info("robot tool %s -> asleep", tool)
            return 503, {"error": "asleep"}
        async with self._lock:
            try:
                result = await asyncio.wait_for(self._dispatch(tool, args), self._timeout_s)
            except asyncio.TimeoutError:
                logger.warning("robot tool %s -> timeout", tool)
                return 504, {"error": "timeout"}
        logger.info("robot tool %s -> %s", tool, _summary(result))
        ok = isinstance(result, dict) and not result.get("error")
        return 200, {"ok": ok, "result": result}

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, payload = 400, {"error": "bad_request"}
        try:
            method, path, headers, body = await asyncio.wait_for(_read_request(reader), HEADER_TIMEOUT_S)
            status, payload = await self.handle(method, path, headers, body)
        except _TooLarge:
            status, payload = 413, {"error": "too_long"}
        except (ValueError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            pass
        except Exception:
            logger.exception("robot tools: request failed")
            status, payload = 500, {"error": "internal"}
        data = json.dumps(payload, default=str).encode()
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
    parts = (await reader.readline()).decode("latin-1").split()
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
