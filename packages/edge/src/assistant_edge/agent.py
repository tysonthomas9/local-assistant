"""The edge agent: one body, one EdgeLink connection to the brain.

    python -m assistant_edge --device-id lite --body reachy --url ws://127.0.0.1:8770/edge/v1

It dials the brain (`LinkClient`, reconnecting with backoff), says `hello` with the body's
capabilities and then:

- **Mic windows.** Microphone audio goes uplink (binary 0x01, 20 ms frames) only while a
  window is open: push-to-talk (`/ptt down` ... `/ptt up` on stdin), a wake (`/wake <word>
  <score>`, or the energy trigger standing in for a wake-word engine), or `mic.follow_up`.
  `mic.close`, `wake.cancel`, `privacy{muted}` or the window's time limit close it. Opening
  sends `vad{start}`, closing sends `vad{end}`.
- **Barge-in.** Opening a window while speech is playing flushes playback locally first and
  sends `vad{start, barge_in: true, stream_id, played_ms}`.
- **Speech.** `speak.begin` + 0x02 frames + `speak.end` play through the body; its playback
  clock is reported as `playback{started|progress|done|flushed, played_ms}`. The reply text
  of `speak.begin` is shown (`SAY`, by the body if it can) and a stream without audio is
  `done` at its `speak.end`. `flush` drops a
  stream (or all) and reports `flushed`; late frames of a flushed stream are dropped.
- **Requests.** `snapshot` is answered with 0x03 JPEG chunks on its slot and `result`;
  `express`, `look_at` and `play_sound` with `result`. What the body cannot do is answered
  `result{ok: false}`. `attention` goes to the body's motion, `privacy` (a request) is applied
  and echoed back as the authoritative state, `timer.sync` and `duck` are acknowledged on the
  console (timers ring from S5 on).
- **Typed input.** Any other stdin line is sent as `text.input`.

Every line it prints is `TAG key=value ... [json]` (the format of `assistant_link.console`).
Tags: BODY, WELCOME, RECV, SENT, MIC-OPEN, MIC-CLOSE, FLUSHED, PLAYBACK, RESULT, SAY, SHOW,
BODY-ERROR, BODY-OK, CLOSED, RETRY, REFUSED, CONSOLE-ERROR, STOPPED.
"""

import asyncio
import contextlib
import signal
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from pydantic import JsonValue

from assistant_contracts.body import AudioFrame, Body, ReportsHealth
from assistant_contracts.capabilities import Capabilities
from assistant_contracts.frames import Frame, FrameKind
from assistant_contracts.messages import (
    Attention,
    BodyInfo,
    Duck,
    EdgeEvent,
    Envelope,
    Error,
    Express,
    Flush,
    Hello,
    LookAt,
    MicClose,
    MicFollowUp,
    Playback,
    PlaySound,
    Privacy,
    Result,
    Snapshot,
    SpeakBegin,
    SpeakEnd,
    TextInput,
    TimerSync,
    Vad,
    Wake,
    WakeCancel,
    Welcome,
    dump_message,
)
from assistant_core.jpeg import jpeg_size
from assistant_core.levels import dbfs
from assistant_link.client import LinkClient
from assistant_link.connection import Connection, LinkClosed, SendQueueFull
from assistant_link.console import emit, read_commands

JPEG_CHUNK_BYTES = 16 * 1024
MIC_STREAM = 0
PTT_MAX_S = 30.0
ENERGY_FRAMES = 3
"""Consecutive loud frames that fire the energy trigger."""


@dataclass
class _Window:
    reason: str
    opened_at: float
    limit_s: float
    frames: int = 0
    level_sum: float = 0.0
    level_max: float = -120.0


@dataclass
class AgentOptions:
    device_id: str
    url: str
    token: str
    energy_trigger_dbfs: float | None = None
    """Open a mic window when the level stays above this (None: off)."""


class EdgeAgent:
    def __init__(self, body: Body, options: AgentOptions) -> None:
        self.body = body
        self.options = options
        self.caps = Capabilities()
        self.client = LinkClient(options.url, options.token, self._hello, self)
        self.conn: Connection | None = None
        self.window: _Window | None = None
        self.muted = False
        self.follow_up_max_s = 10.0
        self.mic_seq = 0
        self.speaking: dict[int, int] = {}
        """stream_id -> rate of speech streams begun and not yet done or flushed."""
        self.voiced: set[int] = set()
        """Streams of `speaking` that got audio (a text-only stream has none to play)."""
        self.flushed: set[int] = set()
        self.timers = 0
        self._loud = 0
        self._stop = asyncio.Event()

    # ------------------------------------------------------------ lifecycle

    def _hello(self) -> Hello:
        return Hello(
            device_id=self.options.device_id,
            sw_version=version("assistant-edge"),
            body=BodyInfo(kind=self.body.kind, capabilities=self.caps),
        )

    async def run(self) -> int:
        # The agent shows reply text (`SAY`) for every body, so every edge takes speak text.
        self.caps = (await self.body.start()).model_copy(update={"speak_text": True})
        emit("BODY", self.caps.model_dump(mode="json"), kind=self.body.kind)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._stop.set)
        tasks = [
            self._spawn(self._capture_loop, "capture"),
            self._spawn(self._playback_loop, "playback"),
            self._spawn(self._body_events_loop, "body-events"),
            self._spawn(self._window_timer, "mic-timer"),
            asyncio.create_task(read_commands(self._command), name="stdin"),
        ]
        if isinstance(self.body, ReportsHealth):
            tasks.append(self._spawn(self._health_loop, "health"))
        link = asyncio.create_task(self.client.run(), name="link")
        stop = asyncio.create_task(self._stop.wait(), name="stop")
        done, _ = await asyncio.wait({link, stop}, return_when=asyncio.FIRST_COMPLETED)
        code = link.result() if link in done else 1000
        await self.client.stop()
        for task in [*tasks, link, stop]:
            task.cancel()
        for task in [*tasks, link, stop]:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        try:
            await self.body.stop()
        finally:
            emit("STOPPED", code=code)
        return 0 if code == 1000 else 2

    def _spawn(self, fn: Callable[[], Coroutine[Any, Any, None]], name: str) -> asyncio.Task[None]:
        async def guarded() -> None:
            try:
                await fn()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                emit("CONSOLE-ERROR", {"detail": f"{name}: {type(exc).__name__}: {exc}"})
                self._stop.set()

        return asyncio.create_task(guarded(), name=name)

    # ------------------------------------------------------------ sending

    async def send(self, message: Envelope) -> None:
        conn = self.conn
        if conn is None:
            emit("DROPPED", dump_message(message), type=message.type)
            return
        with contextlib.suppress(LinkClosed):
            await conn.send(message)
            emit("SENT", dump_message(message), type=message.type)

    def send_frame(self, frame: Frame) -> bool:
        conn = self.conn
        if conn is None:
            return False
        try:
            conn.send_frame_nowait(frame)
        except (LinkClosed, SendQueueFull):
            return False
        return True

    # ------------------------------------------------------------ EdgeHandler

    async def on_welcome(self, conn: Connection, welcome: Welcome) -> None:
        self.conn = conn
        self.follow_up_max_s = welcome.follow_up_max_s
        self.timers = len(welcome.timers)
        emit("WELCOME", dump_message(welcome), session=welcome.session_id)
        if welcome.mute:
            await self._set_muted(True)

    async def on_disconnect(self, code: int | None, reason: str) -> None:
        self.conn = None
        emit("CLOSED", {"reason": reason}, code=code)
        await self._close_window("link_down")

    async def on_refused(self, code: str, detail: str) -> None:
        emit("REFUSED", {"detail": detail}, code=code)

    async def on_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        emit("RETRY", {"reason": reason}, attempt=attempt, delay=f"{delay_s:.3f}")

    async def on_frame(self, conn: Connection, frame: Frame) -> None:
        del conn
        if frame.kind is FrameKind.OUT_PCM:
            rate = self.speaking.get(frame.stream)
            if rate is None:
                return  # flushed, or no speak.begin: drop
            self.voiced.add(frame.stream)
            await self.body.audio.play(frame.stream, frame.payload, rate)
        elif frame.kind is FrameKind.SOUND_CLIP:
            emit("CLIP", stream=frame.stream, bytes=len(frame.payload))

    async def on_message(self, conn: Connection, message: Envelope) -> None:
        del conn
        emit("RECV", dump_message(message), type=message.type)
        match message:
            case SpeakBegin():
                self.flushed.discard(message.stream_id)
                self.speaking[message.stream_id] = message.rate
                if message.text and hasattr(self.body, "say"):
                    self.body.say(message.stream_id, message.text)  # pyright: ignore[reportAttributeAccessIssue]
                elif message.text:
                    emit("SAY", {"stream": message.stream_id, "text": message.text})
            case SpeakEnd():
                await self._speak_end(message.stream_id)
            case Flush():
                target = None if message.stream_id == "all" else message.stream_id
                await self._flush(target, local=False)
            case MicClose():
                await self._close_window(message.reason)
            case MicFollowUp():
                await self._open_window("follow_up", message.seconds)
            case WakeCancel():
                await self._close_window(f"wake.cancel:{message.reason}")
            case Attention():
                if self.body.motion is not None:
                    await self.body.motion.attention(message.state, message.assistant)
            case Express():
                await self._express(message)
            case LookAt():
                ok = self.body.motion is not None and await self.body.motion.look_at(message.target)
                await self._result(message, ok, None if ok else "look_at is not supported")
            case Snapshot():
                await self._snapshot(message)
            case PlaySound():
                await self._result(message, False, "sound clips are not supported yet")
            case Privacy():
                await self._set_muted(message.muted)
            case TimerSync():
                self.timers = len(message.jobs)
                emit("TIMERS", count=self.timers)
            case Duck():
                emit("DUCK", channel=message.channel, gain_db=message.gain_db)
            case Error():
                pass
            case _:
                emit("UNHANDLED", type=message.type)

    # ------------------------------------------------------------ requests

    async def _result(
        self, request: Envelope, ok: bool, error: str | None = None, data: Any = None
    ) -> None:
        emit("RESULT", {"error": error}, re=request.id, type=request.type, ok=str(ok).lower())
        await self.send(Result(re=request.id, ok=ok, error=error, data=data))

    async def _express(self, message: Express) -> None:
        motion = self.body.motion
        if motion is None:
            await self._result(message, False, "this body has no motion")
            return
        try:
            ok = await motion.express(message.name, message.intensity)
        except Exception as exc:
            await self._result(message, False, f"{type(exc).__name__}: {exc}")
            return
        await self._result(message, ok, None if ok else f"no expression {message.name!r}")

    async def _snapshot(self, message: Snapshot) -> None:
        camera = self.body.camera
        if camera is None:
            await self._result(message, False, "this body has no camera")
            return
        try:
            jpeg = await camera.snapshot(message.max_side)
        except Exception as exc:
            await self._result(message, False, f"{type(exc).__name__}: {exc}")
            return
        chunks = [jpeg[i : i + JPEG_CHUNK_BYTES] for i in range(0, len(jpeg), JPEG_CHUNK_BYTES)]
        ts = time.monotonic_ns() // 1000
        for seq, chunk in enumerate(chunks):
            frame = Frame(FrameKind.JPEG_CHUNK, message.slot, seq, ts, chunk)
            if self.conn is not None:
                with contextlib.suppress(LinkClosed):
                    await self.conn.send_frame(frame)
        size = jpeg_size(jpeg)
        data = {"bytes": len(jpeg), "chunks": len(chunks), "slot": message.slot}
        if size is not None:
            data |= {"width": size[0], "height": size[1]}
        await self._result(message, True, data=data)

    # ------------------------------------------------------------ playback

    async def _speak_end(self, stream_id: int) -> None:
        rate = self.speaking.get(stream_id)
        if rate is None:
            return
        if stream_id in self.voiced:
            await self.body.audio.play(stream_id, b"", rate)
            return
        # Text only (no 0x02 frames): nothing to play, so the stream is done at once.
        self.speaking.pop(stream_id, None)
        emit("PLAYBACK", stream=stream_id, state="done", played_ms=0)
        await self.send(Playback(stream_id=stream_id, played_ms=0, state="done"))

    async def _flush(self, stream_id: int | None, *, local: bool) -> tuple[int | None, int]:
        """Flush one stream or all; returns (the stream that was playing, its played ms)."""
        playing = next(iter(self.speaking), None)
        started = time.monotonic()
        played = await self.body.audio.flush(stream_id)
        targets = list(self.speaking) if stream_id is None else [stream_id]
        for sid in targets:
            self.speaking.pop(sid, None)
            self.voiced.discard(sid)
            self.flushed.add(sid)
        took_ms = (time.monotonic() - started) * 1000
        emit("FLUSHED", stream=stream_id if stream_id is not None else "all",
             played_ms=played, local=str(local).lower(), took_ms=f"{took_ms:.1f}")  # fmt: skip
        return playing, played

    async def _playback_loop(self) -> None:
        async for event in self.body.audio.playback_events():
            if event.state in ("done", "flushed"):
                self.speaking.pop(event.stream_id, None)
                self.voiced.discard(event.stream_id)
            emit("PLAYBACK", stream=event.stream_id, state=event.state, played_ms=event.played_ms)
            await self.send(
                Playback(stream_id=event.stream_id, played_ms=event.played_ms, state=event.state)
            )

    # ------------------------------------------------------------ microphone

    async def _open_window(self, reason: str, limit_s: float) -> None:
        if self.muted:
            emit("MIC-REFUSED", reason=reason, muted="true")
            return
        if self.window is not None:
            self.window.limit_s = max(
                self.window.limit_s, time.monotonic() - self.window.opened_at + limit_s
            )
            return
        barge_in = bool(self.speaking)
        vad = Vad(state="start")
        if barge_in:
            stream, played = await self._flush(None, local=True)
            vad = Vad(state="start", barge_in=True, stream_id=stream, played_ms=played)
        self.window = _Window(reason, time.monotonic(), limit_s)
        emit("MIC-OPEN", reason=reason, barge_in=str(barge_in).lower())
        await self.send(vad)

    async def _close_window(self, reason: str) -> None:
        window, self.window = self.window, None
        if window is None:
            return
        mean = window.level_sum / window.frames if window.frames else -120.0
        emit("MIC-CLOSE", reason=reason, opened_by=window.reason, frames=window.frames,
             mean_dbfs=f"{mean:.1f}", max_dbfs=f"{window.level_max:.1f}")  # fmt: skip
        await self.send(Vad(state="end"))

    async def _window_timer(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            window = self.window
            if window is not None and time.monotonic() - window.opened_at > window.limit_s:
                await self._close_window("timeout")

    async def _capture_loop(self) -> None:
        async for frame in self.body.audio.capture():
            await self._on_mic(frame)

    async def _on_mic(self, frame: AudioFrame) -> None:
        level = dbfs(frame.pcm)
        threshold = self.options.energy_trigger_dbfs
        if self.window is None and threshold is not None and not self.muted:
            self._loud = self._loud + 1 if level > threshold else 0
            if self._loud >= ENERGY_FRAMES:
                self._loud = 0
                score = min(1.0, max(0.0, (level - threshold) / 30 + 0.5))
                await self.send(Wake(word="energy", score=round(score, 3)))
                await self._open_window("energy", self.follow_up_max_s)
        window = self.window
        if window is None or self.muted:
            return  # no uplink outside a mic window
        self.mic_seq = (self.mic_seq + 1) % 2**32
        if self.send_frame(
            Frame(FrameKind.MIC_PCM, MIC_STREAM, self.mic_seq, frame.capture_ts_us, frame.pcm)
        ):
            window.frames += 1
            window.level_sum += level
            window.level_max = max(window.level_max, level)

    async def _set_muted(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            await self._close_window("muted")
        emit("PRIVACY", muted=str(muted).lower())
        await self.send(Privacy(muted=muted))

    # ------------------------------------------------------------ body

    async def _body_events_loop(self) -> None:
        async for event in self.body.events():
            if event.kind == "doa":
                continue
            data: dict[str, JsonValue] = {
                k: v for k, v in event.data.items() if isinstance(v, str | int | float | bool)
            }
            await self.send(EdgeEvent(kind=event.kind, data=data))

    async def _health_loop(self) -> None:
        assert isinstance(self.body, ReportsHealth)
        async for health in self.body.health():
            if health.ok:
                emit("BODY-OK", {"detail": health.detail})
            else:
                emit("BODY-ERROR", {"detail": health.detail})
                await self.send(Error(code="body_unavailable", message=health.detail))

    # ------------------------------------------------------------ stdin

    async def _command(self, line: str) -> bool:
        if not line.startswith("/"):
            await self.send(TextInput(text=line))
            return True
        command, *args = line[1:].split()
        match command, args:
            case "quit", []:
                self._stop.set()
                return False
            case "ptt", ["down"]:
                await self._open_window("ptt", PTT_MAX_S)
            case "ptt", ["up"]:
                await self._close_window("ptt")
            case "wake", [word, score]:
                await self.send(Wake(word=word, score=float(score)))
                await self._open_window("wake", self.follow_up_max_s)
            case "wake", [word]:
                await self.send(Wake(word=word, score=1.0))
                await self._open_window("wake", self.follow_up_max_s)
            case "mute", []:
                await self._set_muted(True)
            case "unmute", []:
                await self._set_muted(False)
            case _:
                raise ValueError(
                    f"unknown command {line!r}: /ptt down|up, /wake <word> [score], "
                    "/mute, /unmute, /quit, or plain text"
                )
        return True
