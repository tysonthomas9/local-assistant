"""Console modes: a real EdgeLink server and a real client driven by stdin, printing to stdout.

These are debugging tools (and the peers the e2e features drive), not test doubles: they run
the production `LinkServer` / `LinkClient` over a real WebSocket.

    python -m assistant_link.server --console [--host 127.0.0.1] [--port 8770] [--accept-opus]
    python -m assistant_link.client --console --device-id desk [--url ...] [--opus] [--proto 1.0]

Every output line is `TAG key=value ... [json]`; the JSON (if any) starts at the first `{`.
Server tags: LISTENING, CONNECTED, RECV, FRAME, SENT, SENT-FRAME, REFUSED, DISCONNECTED,
STREAMED, JPEG, CONSOLE-ERROR. A received 0x01 mic frame's FRAME line carries its level
(`dbfs=`); 0x03 chunks of one slot are joined and, once a whole JPEG has arrived, printed as
`JPEG device= slot= bytes= width= height=`.
Client tags: WELCOME, RECV, FRAME, SENT, SENT-FRAME, FRAME-REFUSED, REFUSED, CLOSED, RETRY,
GAVE-UP, CONSOLE-ERROR. Both: FLOOD-STARTED, FLOOD-DONE, FLOOD-STOPPED.
DISCONNECTED and CLOSED carry `stalled=true` when the write-stall watchdog dropped the link.

Server stdin commands:            Client stdin commands:
  send <device> <type> [json]       send <type> [json]
  frame <device> <kind> [k=v ...]   frame <kind> [bytes=N] [stream=N] [seq=N] [unchecked]
  flood <device> <kind> <n> [k=v]   flood <kind> <n> [bytes=N] [stream=N]
  raw <device> <text>               raw <text>
  close <device> <code> [reason]    quit
  stream <device> <stream_id> <clip> [rate=N] [text=... (rest of line)]
  quit
`<kind>` is a name (mic_pcm, out_pcm, jpeg_chunk, sound_clip, opus) or a number (0x01).
Frame payloads are a deterministic byte pattern; both sides print its CRC-32.
`stream` sends real speech the way the brain does: `speak.begin`, the clip as 20 ms 0x02 frames
(as fast as the link takes them), `speak.end`. `<clip>` is `tone:<hz>:<seconds>` or the path
of a 16-bit mono WAV file.
"""

import argparse
import asyncio
import contextlib
import json
import os
import sys
import threading
import time
import wave
import zlib
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from typing import Any

from assistant_contracts.capabilities import Capabilities
from assistant_contracts.frames import Frame, FrameError, FrameKind, encode_frame, opus_negotiated
from assistant_contracts.messages import (
    BodyInfo,
    Envelope,
    Hello,
    Welcome,
    WelcomeAudio,
    dump_message,
    parse_message,
)
from assistant_contracts.version import PROTOCOL_VERSION
from assistant_core.jpeg import jpeg_size
from assistant_core.levels import dbfs, tone
from assistant_link.auth import DevTokenVerifier
from assistant_link.client import LinkClient
from assistant_link.connection import Connection, LinkClosed, SendQueueFull, WrongDirection
from assistant_link.directions import frame_allowed_from
from assistant_link.server import DEFAULT_PORT, EDGE_PATH, LinkServer

DEV_TOKEN_ENV = "ASSISTANT_LINK_TOKEN"
DEFAULT_DEV_TOKEN = "dev-token"


def emit(tag: str, payload: Any = None, **fields: object) -> None:
    parts = [tag, *(f"{k}={v}" for k, v in fields.items())]
    if payload is not None:
        parts.append(payload if isinstance(payload, str) else json.dumps(payload))
    sys.stdout.write(" ".join(parts) + "\n")
    sys.stdout.flush()


def parse_kind(text: str) -> FrameKind:
    try:
        return FrameKind(int(text, 0))
    except ValueError:
        pass
    try:
        return FrameKind[text.upper()]
    except KeyError:
        raise ValueError(f"unknown frame kind {text!r}") from None


def pattern_payload(kind: FrameKind, size: int) -> bytes:
    return bytes((i * 7 + int(kind)) % 256 for i in range(size))


def frame_fields(frame: Frame) -> dict[str, object]:
    return {
        "kind": f"0x{int(frame.kind):02x}",
        "name": frame.kind.name.lower(),
        "stream": frame.stream,
        "seq": frame.seq,
        "bytes": len(frame.payload),
        "crc": f"{zlib.crc32(frame.payload):08x}",
    }


def build_frame(kind_text: str, options: list[str]) -> tuple[Frame, bool]:
    """A frame from `frame` command options; returns (frame, unchecked)."""
    kind = parse_kind(kind_text)
    values = {"bytes": 640, "stream": 1, "seq": 1}
    unchecked = False
    for option in options:
        if option == "unchecked":
            unchecked = True
            continue
        key, sep, value = option.partition("=")
        if not sep or key not in values:
            raise ValueError(f"unknown frame option {option!r}")
        values[key] = int(value, 0)
    ts_us = time.monotonic_ns() // 1000
    frame = Frame(
        kind, values["stream"], values["seq"], ts_us, pattern_payload(kind, values["bytes"])
    )
    return frame, unchecked


_FLOODS: set[asyncio.Task[None]] = set()


def start_flood(conn: Connection, kind_text: str, count_text: str, options: list[str]) -> None:
    """Send `count` frames as fast as the link takes them, in the background.

    Prints FLOOD-STARTED, then FLOOD-DONE when all went out or FLOOD-STOPPED when the link
    closed first (e.g. the write-stall watchdog dropped a frozen peer).
    """
    template, unchecked = build_frame(kind_text, options)
    if unchecked:
        raise ValueError("flood sends checked frames only")
    count = int(count_text, 0)
    if count < 1:
        raise ValueError("flood count must be >= 1")
    if not frame_allowed_from(conn.side, template.kind):
        raise WrongDirection(f"the {conn.side} may not send frame {template.kind.name}")
    conn.codec.encode(template)  # refuses 0x05 unless negotiated, before starting
    tags = {"device": conn.device_id, "kind": f"0x{int(template.kind):02x}", "count": count}

    async def run() -> None:
        sent = 0
        try:
            for seq in range(count):
                frame = Frame(template.kind, template.stream, seq % 2**32, 0, template.payload)
                await conn.send_frame(frame)
                sent += 1
        except LinkClosed as exc:
            emit("FLOOD-STOPPED", {"detail": str(exc)}, sent=sent, **tags)
        else:
            emit("FLOOD-DONE", sent=sent, **tags)

    task = asyncio.create_task(run(), name="flood")
    _FLOODS.add(task)
    task.add_done_callback(_FLOODS.discard)
    emit("FLOOD-STARTED", **tags)


def build_message(type_name: str, fields_json: str) -> Envelope:
    fields = json.loads(fields_json) if fields_json.strip() else {}
    if not isinstance(fields, dict):
        raise ValueError("message fields must be a JSON object")
    return parse_message({**fields, "type": type_name})


async def read_commands(handle: Callable[[str], Awaitable[bool]]) -> None:
    """Feed stdin lines to `handle` until it returns False or stdin ends.

    A daemon thread reads stdin (works for pipes, terminals and /dev/null alike) and hands
    lines to the event loop; at EOF the console keeps running until it is signalled.
    """
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[str | None] = asyncio.Queue()

    def pump() -> None:
        for raw in sys.stdin:
            loop.call_soon_threadsafe(lines.put_nowait, raw)
        loop.call_soon_threadsafe(lines.put_nowait, None)

    threading.Thread(target=pump, name="console-stdin", daemon=True).start()
    while (raw := await lines.get()) is not None:
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            if not await handle(text):
                return
        except (ValueError, KeyError, WrongDirection, FrameError, SendQueueFull, LinkClosed) as exc:
            emit("CONSOLE-ERROR", {"detail": f"{type(exc).__name__}: {exc}", "command": text})
    await asyncio.Event().wait()  # stdin ended: keep running until signalled


# ---------------------------------------------------------------- server console


def load_clip(clip: str, rate: int) -> tuple[bytes, int]:
    """(s16le mono PCM, its rate) for `tone:<hz>:<seconds>` or a 16-bit mono WAV path."""
    if clip.startswith("tone:"):
        _, hz, seconds = clip.split(":")
        return tone(float(hz), float(seconds), rate), rate
    with wave.open(clip, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError(f"{clip}: need 16-bit mono WAV")
        return wav.readframes(wav.getnframes()), wav.getframerate()


async def stream_speech(conn: Connection, stream_id: int, clip: str, options: list[str]) -> None:
    rate, text = 24000, None
    for i, option in enumerate(options):
        key, _, value = option.partition("=")
        if key == "rate":
            rate = int(value)
        elif key == "text":
            text = " ".join([value, *options[i + 1 :]])  # the rest of the line
            break
        else:
            raise ValueError(f"unknown stream option {option!r}")
    pcm, rate = load_clip(clip, rate)
    begin = parse_message(
        {"type": "speak.begin", "stream_id": stream_id, "rate": rate, "text": text}
    )
    await conn.send(begin)
    emit("SENT", dump_message(begin), device=conn.device_id, type=begin.type)
    step = rate * 2 // 50  # 20 ms
    for seq, offset in enumerate(range(0, len(pcm), step)):
        chunk = pcm[offset : offset + step]
        await conn.send_frame(Frame(FrameKind.OUT_PCM, stream_id, seq, 0, chunk))
    end = parse_message({"type": "speak.end", "stream_id": stream_id})
    await conn.send(end)
    emit("SENT", dump_message(end), device=conn.device_id, type=end.type)
    emit("STREAMED", device=conn.device_id, stream=stream_id, rate=rate,
         ms=len(pcm) * 1000 // (2 * rate), bytes=len(pcm))  # fmt: skip


class ServerConsole:
    def __init__(self, accept_opus: bool) -> None:
        self.accept_opus = accept_opus
        self.server: LinkServer | None = None
        self.jpeg: dict[tuple[str, int], bytearray] = {}

    async def on_connect(self, conn: Connection, hello: Hello) -> Welcome:
        caps = hello.body.capabilities
        emit(
            "CONNECTED",
            device=conn.device_id,
            session=conn.session_id,
            proto=hello.proto,
            opus=str(opus_negotiated(caps, self.accept_opus)).lower(),
            payload=dump_message(hello),
        )
        return Welcome(
            session_id=conn.session_id,
            audio=WelcomeAudio(opus=self.accept_opus, speak_text=caps.speak_text),
        )

    async def on_message(self, conn: Connection, message: Envelope) -> None:
        emit("RECV", dump_message(message), device=conn.device_id, type=message.type)

    async def on_frame(self, conn: Connection, frame: Frame) -> None:
        fields = frame_fields(frame)
        if frame.kind is FrameKind.MIC_PCM:
            fields["dbfs"] = f"{dbfs(frame.payload):.1f}"
        emit("FRAME", device=conn.device_id, **fields)
        if frame.kind is FrameKind.JPEG_CHUNK:
            key = (conn.device_id, frame.stream)
            if frame.seq == 0:
                self.jpeg[key] = bytearray()
            data = self.jpeg.setdefault(key, bytearray())
            data += frame.payload
            if data.endswith(b"\xff\xd9"):
                size = jpeg_size(bytes(data)) or (0, 0)
                emit("JPEG", device=conn.device_id, slot=frame.stream, bytes=len(data),
                     width=size[0], height=size[1])  # fmt: skip
                del self.jpeg[key]

    async def on_disconnect(self, conn: Connection, code: int | None, reason: str) -> None:
        stalled = str(conn.stalled).lower()
        emit("DISCONNECTED", {"reason": reason}, device=conn.device_id, code=code, stalled=stalled)

    async def on_refused(self, device_id: str | None, code: str, detail: str) -> None:
        emit("REFUSED", {"detail": detail}, device=device_id or "-", code=code)

    def _conn(self, device: str) -> Connection:
        assert self.server is not None
        conn = self.server.connections.get(device)
        if conn is None:
            raise KeyError(f"no connected device {device!r}")
        return conn

    async def handle(self, line: str) -> bool:
        command, _, rest = line.partition(" ")
        if command == "quit":
            return False
        if command == "send":
            device, _, rest = rest.strip().partition(" ")
            type_name, _, fields = rest.strip().partition(" ")
            message = build_message(type_name, fields)
            await self._conn(device).send(message)
            emit("SENT", dump_message(message), device=device, type=message.type)
        elif command == "frame":
            device, kind, *options = rest.split()
            frame, unchecked = build_frame(kind, options)
            conn = self._conn(device)
            if unchecked:
                await conn.send_raw(encode_frame(frame, opus=True))
            else:
                await conn.send_frame(frame)
            emit("SENT-FRAME", device=device, **frame_fields(frame))
        elif command == "flood":
            device, kind, count, *options = rest.split()
            start_flood(self._conn(device), kind, count, options)
        elif command == "raw":
            device, _, text = rest.strip().partition(" ")
            await self._conn(device).send_raw(text)
            emit("SENT-RAW", {"text": text}, device=device)
        elif command == "close":
            device, code, *reason = rest.split()
            await self._conn(device).close(int(code), " ".join(reason))
        elif command == "stream":
            device, stream_id, clip, *options = rest.split()
            await stream_speech(self._conn(device), int(stream_id), clip, options)
        else:
            raise ValueError(f"unknown command {command!r}")
        return True


def server_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_link.server")
    parser.add_argument("--console", action="store_true", help="interactive console (required)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", default=os.environ.get(DEV_TOKEN_ENV, DEFAULT_DEV_TOKEN))
    parser.add_argument("--accept-opus", action="store_true", help="accept 0x05 Opus frames")
    args = parser.parse_args(argv)
    if not args.console:
        parser.error("only --console mode exists here; the brain runs the server (S4)")
    console = ServerConsole(args.accept_opus)
    server = LinkServer(console, DevTokenVerifier(args.token), host=args.host, port=args.port)
    console.server = server

    async def main() -> None:
        await server.start()
        emit("LISTENING", url=server.url)
        commands = asyncio.create_task(read_commands(console.handle))
        serving = asyncio.create_task(server.serve_forever())
        await commands
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving
        await server.close()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
    return 0


# ---------------------------------------------------------------- client console


class ClientConsole:
    def __init__(self) -> None:
        self.client: LinkClient | None = None

    async def on_welcome(self, conn: Connection, welcome: Welcome) -> None:
        emit(
            "WELCOME",
            dump_message(welcome),
            session=welcome.session_id,
            opus=str(conn.opus).lower(),
        )

    async def on_message(self, conn: Connection, message: Envelope) -> None:
        emit("RECV", dump_message(message), type=message.type)

    async def on_frame(self, conn: Connection, frame: Frame) -> None:
        emit("FRAME", **frame_fields(frame))

    async def on_disconnect(self, code: int | None, reason: str) -> None:
        stalled = str(self.client.stalled if self.client is not None else False).lower()
        emit("CLOSED", {"reason": reason}, code=code, stalled=stalled)

    async def on_refused(self, code: str, detail: str) -> None:
        emit("REFUSED", {"detail": detail}, code=code)

    async def on_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        emit("RETRY", {"reason": reason}, attempt=attempt, delay=f"{delay_s:.3f}")

    def _conn(self) -> Connection:
        conn = self.client.connection if self.client is not None else None
        if conn is None:
            raise KeyError("not connected")
        return conn

    async def handle(self, line: str) -> bool:
        command, _, rest = line.partition(" ")
        if command == "quit":
            if self.client is not None:
                await self.client.stop()
            return False
        if command == "send":
            type_name, _, fields = rest.strip().partition(" ")
            message = build_message(type_name, fields)
            await self._conn().send(message)
            emit("SENT", dump_message(message), type=message.type)
        elif command == "frame":
            kind, *options = rest.split()
            frame, unchecked = build_frame(kind, options)
            conn = self._conn()
            if unchecked:
                await conn.send_raw(encode_frame(frame, opus=True))
            else:
                try:
                    await conn.send_frame(frame)
                except FrameError as exc:
                    emit("FRAME-REFUSED", {"detail": str(exc)}, **frame_fields(frame))
                    return True
            emit("SENT-FRAME", **frame_fields(frame))
        elif command == "flood":
            kind, count, *options = rest.split()
            start_flood(self._conn(), kind, count, options)
        elif command == "raw":
            await self._conn().send_raw(rest.strip())
            emit("SENT-RAW", {"text": rest.strip()})
        else:
            raise ValueError(f"unknown command {command!r}")
        return True


def client_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_link.client")
    parser.add_argument("--console", action="store_true", help="interactive console (required)")
    parser.add_argument("--url", default=f"ws://127.0.0.1:{DEFAULT_PORT}{EDGE_PATH}")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--token", default=os.environ.get(DEV_TOKEN_ENV, DEFAULT_DEV_TOKEN))
    parser.add_argument("--proto", default=PROTOCOL_VERSION, help="hello.proto to announce")
    parser.add_argument("--opus", action="store_true", help="advertise the opus capability")
    parser.add_argument("--speak-text", action="store_true", help="advertise speak_text")
    args = parser.parse_args(argv)
    if not args.console:
        parser.error("only --console mode exists here; the edge agent runs the client (S3)")

    def make_hello() -> Hello:
        caps = Capabilities(opus=args.opus, speak_text=args.speak_text)
        return Hello(
            device_id=args.device_id,
            proto=args.proto,
            sw_version=version("assistant-link"),
            body=BodyInfo(kind="console", capabilities=caps),
        )

    console = ClientConsole()
    client = LinkClient(args.url, args.token, make_hello, console)
    console.client = client

    async def main() -> int:
        commands = asyncio.create_task(read_commands(console.handle))
        code = await client.run()
        commands.cancel()
        if code != 1000:
            emit("GAVE-UP", code=code)
            return 2
        return 0

    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        return 0
