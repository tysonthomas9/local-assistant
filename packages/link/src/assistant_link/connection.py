"""One EdgeLink connection: JSON vs binary dispatch and a bounded per-connection send queue.

Both the brain-side server and the edge-side client wrap their websocket in a `Connection`.
Outgoing messages and frames go through a bounded queue drained by one writer task, so a slow
peer pushes back on the producer (`send` waits while the queue is full; `send_nowait` raises
`SendQueueFull`) instead of growing memory without bound. Incoming text frames are decoded
as messages and binary frames through the per-connection `FrameCodec`; protocol errors are
answered with an `error` message, and a major-version mismatch closes the link with 4001.
"""

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Final

from pydantic import ValidationError
from websockets.asyncio.connection import Connection as WebSocket
from websockets.exceptions import ConnectionClosed

from assistant_contracts.frames import Frame, FrameCodec, FrameError, FrameKindNotNegotiated
from assistant_contracts.messages import Envelope, Error, dump_message, parse_message
from assistant_contracts.version import PROTOCOL_MAJOR, CloseCode
from assistant_link.directions import Side, frame_allowed_from, message_allowed_from

SEND_QUEUE_SIZE: Final = 256
"""Queued outgoing items per connection (about 5 s of 20 ms audio frames)."""

MessageHandler = Callable[[Envelope], Awaitable[None]]
FrameHandler = Callable[[Frame], Awaitable[None]]
RefusalHandler = Callable[[str, str], Awaitable[None]]
"""Called with (error code, detail) whenever an incoming item is refused."""


class SendQueueFull(Exception):
    """`send_nowait` found the connection's send queue full (the peer is not keeping up)."""


class WrongDirection(ValueError):
    """This side may not send (or receive) that message type or frame kind."""


class VersionMismatch(Exception):
    """The peer speaks another major protocol version."""


def other(side: Side) -> Side:
    return "brain" if side == "edge" else "edge"


def check_major(data: dict[str, object]) -> None:
    """Raise `VersionMismatch` if a decoded JSON message carries another major `v`."""
    v = data.get("v")
    if v != PROTOCOL_MAJOR:
        raise VersionMismatch(f"message has v={v!r}, this side speaks v={PROTOCOL_MAJOR}")


def decode_text(raw: str | bytes, *, sender: Side) -> Envelope:
    """Decode one JSON text frame sent by `sender`.

    Raises `VersionMismatch`, `WrongDirection` or `ValueError` (bad JSON or fields).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError("a message must be a JSON object")
    check_major(data)
    try:
        message = parse_message(data)
    except ValidationError as exc:
        raise ValueError(f"invalid message: {exc.errors()[0]['msg']}") from None
    if not message_allowed_from(sender, message.type):
        raise WrongDirection(f"{message.type!r} may not be sent by the {sender}")
    return message


class Connection:
    """A live EdgeLink connection seen from `side` ("brain" on the server, "edge" on a client)."""

    def __init__(
        self,
        ws: WebSocket,
        *,
        side: Side,
        device_id: str,
        session_id: str,
        opus: bool = False,
        queue_size: int = SEND_QUEUE_SIZE,
    ) -> None:
        self.ws = ws
        self.side: Side = side
        self.peer: Side = other(side)
        self.device_id = device_id
        self.session_id = session_id
        self.codec = FrameCodec(opus=opus)
        self._queue: asyncio.Queue[str | bytes | None] = asyncio.Queue(maxsize=queue_size)
        self._writer: asyncio.Task[None] | None = None

    @property
    def opus(self) -> bool:
        """0x05 Opus frames were negotiated on this connection."""
        return self.codec.opus

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._writer is None:
            self._writer = asyncio.create_task(self._drain(), name=f"link-writer:{self.device_id}")

    async def _drain(self) -> None:
        while (item := await self._queue.get()) is not None:
            try:
                await self.ws.send(item)
            except ConnectionClosed:
                return

    # ------------------------------------------------------------ sending

    def _encode_message(self, message: Envelope) -> str:
        if not message_allowed_from(self.side, message.type):
            raise WrongDirection(f"the {self.side} may not send {message.type!r}")
        return dump_message(message)

    def _encode_frame(self, frame: Frame) -> bytes:
        if not frame_allowed_from(self.side, frame.kind):
            raise WrongDirection(f"the {self.side} may not send frame {frame.kind.name}")
        return self.codec.encode(frame)

    async def send(self, message: Envelope) -> None:
        """Queue a message; waits while the send queue is full (backpressure)."""
        await self._queue.put(self._encode_message(message))

    def send_nowait(self, message: Envelope) -> None:
        try:
            self._queue.put_nowait(self._encode_message(message))
        except asyncio.QueueFull:
            raise SendQueueFull(f"send queue to {self.device_id} is full") from None

    async def send_frame(self, frame: Frame) -> None:
        """Queue a binary frame. Raises `FrameKindNotNegotiated` for 0x05 without Opus."""
        await self._queue.put(self._encode_frame(frame))

    def send_frame_nowait(self, frame: Frame) -> None:
        try:
            self._queue.put_nowait(self._encode_frame(frame))
        except asyncio.QueueFull:
            raise SendQueueFull(f"send queue to {self.device_id} is full") from None

    async def send_raw(self, data: str | bytes) -> None:
        """Queue raw bytes or text unchecked (debug consoles only: tests peer validation)."""
        await self._queue.put(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(None)
        await self.ws.close(code, reason)
        await self.stop_writer()

    async def stop_writer(self) -> None:
        if self._writer is not None:
            self._writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer

    # ------------------------------------------------------------ receiving

    async def _refuse(self, code: str, detail: str, on_refused: RefusalHandler | None) -> None:
        if on_refused is not None:
            await on_refused(code, detail)
        with contextlib.suppress(ConnectionClosed):
            await self.ws.send(dump_message(Error(code=code, message=detail)))

    async def receive_loop(
        self,
        on_message: MessageHandler,
        on_frame: FrameHandler,
        on_refused: RefusalHandler | None = None,
    ) -> None:
        """Dispatch incoming items until the connection closes.

        Text frames become messages, binary frames go through the codec. A bad item is
        answered with an `error` message and the link stays up; a major-version mismatch
        closes it with 4001.
        """
        try:
            async for raw in self.ws:
                if isinstance(raw, str):
                    try:
                        message = decode_text(raw, sender=self.peer)
                    except VersionMismatch as exc:
                        await self._refuse("version_mismatch", str(exc), on_refused)
                        await self.ws.close(CloseCode.VERSION_MISMATCH, "protocol major mismatch")
                        return
                    except WrongDirection as exc:
                        await self._refuse("wrong_direction", str(exc), on_refused)
                        continue
                    except ValueError as exc:
                        await self._refuse("bad_message", str(exc), on_refused)
                        continue
                    await on_message(message)
                else:
                    try:
                        frame = self.codec.decode(raw)
                    except FrameKindNotNegotiated as exc:
                        await self._refuse("frame_refused", str(exc), on_refused)
                        continue
                    except FrameError as exc:
                        await self._refuse("bad_frame", str(exc), on_refused)
                        continue
                    if not frame_allowed_from(self.peer, frame.kind):
                        detail = f"frame {frame.kind.name} may not be sent by the {self.peer}"
                        await self._refuse("wrong_direction", detail, on_refused)
                        continue
                    await on_frame(frame)
        except ConnectionClosed:
            return
