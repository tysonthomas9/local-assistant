"""The edge side of EdgeLink: dials the brain, says hello, and reconnects when the link drops.

    client = LinkClient("ws://127.0.0.1:8770/edge/v1", token, make_hello, handler)
    code = await client.run()   # returns only when the brain refused us for good

After a drop the client waits `backoff_delay(attempt)` (0.5 s doubling to 10 s, with jitter)
and dials again; a successful `welcome` resets the backoff. A 4001 (version) or 4003 (auth)
close is final: retrying cannot fix it, so `run` returns that code.

`ssl` and `pin_fingerprint` are the hooks for TLS with certificate pinning (task S7).
`python -m assistant_link.client --console` runs a real client that prints what it receives
and sends what is typed on stdin (see `assistant_link.console`).
"""

import asyncio
import contextlib
import hashlib
import random
import ssl as ssl_module
from collections.abc import Callable
from typing import Final, Protocol
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from assistant_contracts.frames import Frame, opus_negotiated
from assistant_contracts.messages import Envelope, Error, Hello, Welcome, dump_message
from assistant_contracts.version import CloseCode
from assistant_link.backoff import backoff_delay
from assistant_link.connection import Connection, VersionMismatch, decode_text
from assistant_link.server import CLOSE_TIMEOUT_S, PING_INTERVAL_S, PING_TIMEOUT_S, is_loopback

FATAL_CLOSE_CODES: Final = frozenset({CloseCode.VERSION_MISMATCH, CloseCode.AUTH_REFUSED})
WELCOME_TIMEOUT_S: Final = 10.0
OPEN_TIMEOUT_S: Final = 5.0


class EdgeHandler(Protocol):
    """What the edge plugs into the client (the console, later the edge agent)."""

    async def on_welcome(self, conn: Connection, welcome: Welcome) -> None: ...

    async def on_message(self, conn: Connection, message: Envelope) -> None: ...

    async def on_frame(self, conn: Connection, frame: Frame) -> None: ...

    async def on_disconnect(self, code: int | None, reason: str) -> None: ...

    async def on_refused(self, code: str, detail: str) -> None:
        """Something was refused: an item the brain sent us, or our hello (brain `error`)."""
        ...

    async def on_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        """The link is down; the next dial happens in `delay_s`."""
        ...


def cert_fingerprint(der: bytes) -> str:
    """SHA-256 of a DER certificate, lowercase hex (the value pinned by edges and mDNS TXT)."""
    return hashlib.sha256(der).hexdigest()


class PinMismatch(Exception):
    """The brain's certificate does not match the pinned fingerprint."""


class LinkClient:
    def __init__(
        self,
        url: str,
        token: str,
        hello: Callable[[], Hello],
        handler: EdgeHandler,
        *,
        ssl: ssl_module.SSLContext | None = None,
        pin_fingerprint: Callable[[str], bool] | None = None,
        ping_interval: float = PING_INTERVAL_S,
        ping_timeout: float = PING_TIMEOUT_S,
        rng: random.Random | None = None,
    ) -> None:
        parts = urlsplit(url)
        if parts.scheme == "ws" and not is_loopback(parts.hostname or ""):
            raise ValueError(f"plain ws:// is allowed only on loopback, not {parts.hostname!r}")
        self.url = url
        self.token = token
        self.make_hello = hello
        self.handler = handler
        self.ssl = ssl
        self.pin_fingerprint = pin_fingerprint
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.rng = rng or random.Random()
        self.connection: Connection | None = None
        self._stopping = False
        self._ws: ClientConnection | None = None

    async def stop(self) -> None:
        self._stopping = True
        if self._ws is not None:
            await self._ws.close()

    async def run(self) -> int:
        """Stay connected until stopped (returns 1000) or refused for good (returns 4001/4003)."""
        attempt = 0
        while not self._stopping:
            try:
                code, reason, welcomed = await self._session()
            except (OSError, InvalidHandshake, InvalidURI, TimeoutError, PinMismatch) as exc:
                code, reason, welcomed = None, f"{type(exc).__name__}: {exc}", False
            if self._stopping:
                break
            if code in FATAL_CLOSE_CODES:
                assert code is not None
                return code
            attempt = 1 if welcomed else attempt + 1
            delay = backoff_delay(attempt, self.rng)
            await self.handler.on_retry(attempt, delay, reason)
            await asyncio.sleep(delay)
        return 1000

    def _check_pin(self, ws: ClientConnection) -> None:
        if self.pin_fingerprint is None:
            return
        ssl_object = ws.transport.get_extra_info("ssl_object")
        der = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
        if der is None or not self.pin_fingerprint(cert_fingerprint(der)):
            raise PinMismatch("the brain certificate does not match the pinned fingerprint")

    async def _session(self) -> tuple[int | None, str, bool]:
        """One connection: handshake, dispatch until closed. Returns (code, reason, welcomed)."""
        hello = self.make_hello()
        async with connect(
            self.url,
            additional_headers={"Authorization": f"Bearer {self.token}"},
            ssl=self.ssl,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            open_timeout=OPEN_TIMEOUT_S,
            close_timeout=CLOSE_TIMEOUT_S,
            max_size=2**22,
        ) as ws:
            self._ws = ws
            try:
                self._check_pin(ws)
                await ws.send(dump_message(hello))
                welcome = await self._await_welcome(ws)
                if welcome is None:
                    await ws.wait_closed()
                    return await self._closed(ws, welcomed=False)
                conn = Connection(
                    ws,
                    side="edge",
                    device_id=hello.device_id,
                    session_id=welcome.session_id,
                    opus=opus_negotiated(hello.body.capabilities, welcome.audio.opus),
                )
                self.connection = conn
                conn.start()
                await self.handler.on_welcome(conn, welcome)
                try:
                    await conn.receive_loop(
                        lambda m: self.handler.on_message(conn, m),
                        lambda f: self.handler.on_frame(conn, f),
                        self.handler.on_refused,
                    )
                finally:
                    self.connection = None
                    await conn.stop_writer()
                await ws.wait_closed()
                return await self._closed(ws, welcomed=True)
            finally:
                self._ws = None

    async def _closed(
        self, ws: ClientConnection, *, welcomed: bool
    ) -> tuple[int | None, str, bool]:
        code, reason = ws.close_code, ws.close_reason or ""
        await self.handler.on_disconnect(code, reason)
        return code, reason or f"closed with {code}", welcomed

    async def _await_welcome(self, ws: ClientConnection) -> Welcome | None:
        """The brain's first message: `welcome`, or `error` before it closes the link."""
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), WELCOME_TIMEOUT_S)
            except ConnectionClosed:
                return None
            if not isinstance(raw, str):
                continue
            try:
                message = decode_text(raw, sender="brain")
            except (VersionMismatch, ValueError) as exc:
                await self.handler.on_refused("bad_welcome", str(exc))
                with contextlib.suppress(ConnectionClosed):
                    await ws.close(CloseCode.VERSION_MISMATCH, "brain speaks another version")
                return None
            if isinstance(message, Welcome):
                return message
            if isinstance(message, Error):
                await self.handler.on_refused(message.code, message.message)


if __name__ == "__main__":
    from assistant_link.console import client_main

    raise SystemExit(client_main())
