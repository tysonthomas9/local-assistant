"""The brain side of EdgeLink: a WebSocket server that edges dial into.

    server = LinkServer(handler, DevTokenVerifier("dev-token"), host="127.0.0.1", port=8770)
    await server.start()

Per connection: the bearer token is checked through the pluggable `TokenVerifier`, the first
message must be `hello` (major-version mismatch -> close 4001, bad token -> close 4003), the
handler builds the `welcome`, and then JSON and binary frames are dispatched to the handler.
Keepalive: a WebSocket ping every 5 s; a peer that has not answered for 15 s is dead (the
ping is sent 5 s after the last pong and gets 10 s to be answered).

`python -m assistant_link.server --console` runs a real server that prints what it receives
and sends what is typed on stdin (see `assistant_link.console`).
"""

import asyncio
import contextlib
import ipaddress
import logging
import ssl as ssl_module
import uuid
from typing import Final, Protocol

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidMessage

from assistant_contracts.frames import Frame, opus_negotiated
from assistant_contracts.messages import Envelope, Error, Hello, Welcome, dump_message
from assistant_contracts.version import CloseCode, is_compatible
from assistant_link.auth import TokenVerifier, bearer_token
from assistant_link.connection import Connection, VersionMismatch, decode_text

EDGE_PATH: Final = "/edge/v1"
DEFAULT_PORT: Final = 8770
PING_INTERVAL_S: Final = 5.0
PING_TIMEOUT_S: Final = 10.0
"""Ping 5 s after the last pong, dead 10 s later: 15 s without a reply in total."""
CLOSE_TIMEOUT_S: Final = 2.0
"""How long a closing handshake may take before the TCP connection is dropped."""
HELLO_TIMEOUT_S: Final = 10.0
POLICY_VIOLATION: Final = 1008


class _QuietEarlyHangups(logging.Filter):
    """Drop websockets' traceback for a peer that hung up before sending any HTTP request.

    Port probes and edges that give up mid-connect do this; it is not a server error.
    Every other handshake failure is still logged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        hung_up = isinstance(exc, InvalidMessage) and isinstance(exc.__cause__, EOFError)
        return not (record.getMessage() == "opening handshake failed" and hung_up)


WS_LOGGER: Final = logging.getLogger("assistant_link.server.websockets")
WS_LOGGER.addFilter(_QuietEarlyHangups())


class LinkHandler(Protocol):
    """What the brain plugs into the server (the console, later the SessionManager)."""

    async def on_connect(self, conn: Connection, hello: Hello) -> Welcome:
        """A device passed auth and version checks. Return the `welcome` to send.

        `conn.session_id` is the new session's id; use it as `welcome.session_id`.
        """
        ...

    async def on_message(self, conn: Connection, message: Envelope) -> None: ...

    async def on_frame(self, conn: Connection, frame: Frame) -> None: ...

    async def on_disconnect(self, conn: Connection, code: int | None, reason: str) -> None: ...

    async def on_refused(self, device_id: str | None, code: str, detail: str) -> None:
        """A connection or one incoming item was refused (auth, version, bad frame, ...)."""
        ...


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class LinkServer:
    def __init__(
        self,
        handler: LinkHandler,
        verifier: TokenVerifier,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        ssl: ssl_module.SSLContext | None = None,
        ping_interval: float = PING_INTERVAL_S,
        ping_timeout: float = PING_TIMEOUT_S,
        hello_timeout: float = HELLO_TIMEOUT_S,
    ) -> None:
        if ssl is None and not is_loopback(host):
            raise ValueError(f"plain ws:// is allowed only on loopback, not on {host!r}")
        self.handler = handler
        self.verifier = verifier
        self.host = host
        self.port = port
        self.ssl = ssl
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.hello_timeout = hello_timeout
        self.connections: dict[str, Connection] = {}
        self._server: Server | None = None

    @property
    def url(self) -> str:
        scheme = "wss" if self.ssl is not None else "ws"
        return f"{scheme}://{self.host}:{self.port}{EDGE_PATH}"

    async def start(self) -> None:
        self._server = await serve(
            self._serve_one,
            self.host,
            self.port,
            ssl=self.ssl,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            close_timeout=CLOSE_TIMEOUT_S,
            max_size=2**22,
            logger=WS_LOGGER,
        )
        if self.port == 0:
            self.port = next(iter(self._server.sockets)).getsockname()[1]

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def serve_forever(self) -> None:
        assert self._server is not None, "call start() first"
        await self._server.serve_forever()

    async def _refuse(
        self, ws: ServerConnection, device_id: str | None, close: int, code: str, detail: str
    ) -> None:
        await self.handler.on_refused(device_id, code, detail)
        with contextlib.suppress(ConnectionClosed):
            await ws.send(dump_message(Error(code=code, message=detail)))
        await ws.close(close, detail[:120])

    async def _read_hello(self, ws: ServerConnection) -> Hello | None:
        try:
            raw = await asyncio.wait_for(ws.recv(), self.hello_timeout)
        except ConnectionClosed:
            return None  # the peer left before saying hello: nothing to refuse or log
        except TimeoutError:
            await self._refuse(ws, None, POLICY_VIOLATION, "no_hello", "no hello in time")
            return None
        if not isinstance(raw, str):
            await self._refuse(ws, None, POLICY_VIOLATION, "no_hello", "expected hello")
            return None
        try:
            message = decode_text(raw, sender="edge")
        except VersionMismatch as exc:
            await self._refuse(ws, None, CloseCode.VERSION_MISMATCH, "version_mismatch", str(exc))
            return None
        except ValueError as exc:
            await self._refuse(ws, None, POLICY_VIOLATION, "no_hello", str(exc))
            return None
        if not isinstance(message, Hello):
            detail = f"expected hello, got {message.type}"
            await self._refuse(ws, None, POLICY_VIOLATION, "no_hello", detail)
            return None
        if not is_compatible(message.proto):
            detail = f"hello.proto {message.proto!r} has another major version"
            await self._refuse(
                ws, message.device_id, CloseCode.VERSION_MISMATCH, "version_mismatch", detail
            )
            return None
        return message

    async def _serve_one(self, ws: ServerConnection) -> None:
        request = ws.request
        if request is None or request.path != EDGE_PATH:
            await self._refuse(ws, None, POLICY_VIOLATION, "bad_path", f"use {EDGE_PATH}")
            return
        token = bearer_token(request.headers.get("Authorization"))
        if token is None:
            await self._refuse(ws, None, CloseCode.AUTH_REFUSED, "auth", "missing bearer token")
            return
        hello = await self._read_hello(ws)
        if hello is None:
            return
        if not await self.verifier.verify(token, hello.device_id):
            await self._refuse(ws, hello.device_id, CloseCode.AUTH_REFUSED, "auth", "bad token")
            return

        conn = Connection(
            ws,
            side="brain",
            device_id=hello.device_id,
            session_id=f"s-{uuid.uuid4().hex[:12]}",
            write_stall_s=self.ping_interval + self.ping_timeout,
        )
        welcome = await self.handler.on_connect(conn, hello)
        conn.codec.opus = opus_negotiated(hello.body.capabilities, welcome.audio.opus)
        previous = self.connections.get(hello.device_id)
        self.connections[hello.device_id] = conn
        if previous is not None:
            await previous.close(1000, "replaced by a new connection")
        conn.start()
        await conn.send(welcome)
        try:
            await conn.receive_loop(
                lambda m: self.handler.on_message(conn, m),
                lambda f: self.handler.on_frame(conn, f),
                lambda code, detail: self.handler.on_refused(conn.device_id, code, detail),
            )
        finally:
            await conn.stop_writer()
            if self.connections.get(conn.device_id) is conn:
                del self.connections[conn.device_id]
            await ws.wait_closed()
            await self.handler.on_disconnect(conn, ws.close_code, ws.close_reason or "")


if __name__ == "__main__":
    from assistant_link.console import server_main

    raise SystemExit(server_main())
