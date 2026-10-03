"""The edge agent: one body, one EdgeLink connection to the brain.

    python -m assistant_edge --device-id lite --body reachy --url ws://127.0.0.1:8770/edge/v1

It dials the brain (`LinkClient`, reconnecting with backoff), says `hello` with the body's
capabilities and then:

- **Mic windows.** Microphone audio goes uplink (binary 0x01, 20 ms frames) only while a
  window is open: push-to-talk (`/ptt down` ... `/ptt up` on stdin), a wake (the wake-word
  engine, `/wake <word> <score>`, or the energy trigger), speech (open mic), or `mic.follow_up`.
  `mic.close`, `wake.cancel`, `privacy{muted}`, the end of speech or the window's time limit
  (`[edge.wake] max_window_s`, never the brain's follow-up time) close it. Opening sends
  `vad{start}`, closing sends `vad{end}`. No window, whoever asks for it and for how long, stays
  open longer than `max_window_s`, itself at most `MIC_CAP_S` (2 minutes): the edge enforces
  it, a longer request (`mic.follow_up`, push-to-talk, ...) is clamped.
- **Listening modes** (`--listen`, `[edge.listen] mode`; see `assistant_edge.listen`):
  `wake_word` (the default) runs the wake-word engine on every frame: a detection sends
  `wake{word, score}` and opens a window with the last `[edge.wake] pre_roll_s` of audio first.
  `open_mic` opens a window when the speech detector (Silero VAD) hears speech, with its
  pre-roll. In both the speech detector ends the window (`vad_end`, after `[edge.vad] end_ms` of
  no speech; a wake window in which no speech followed the wake word ends `no_speech`), and
  `mic.follow_up` lets speech open a window without the wake word until it runs out (the
  window opens only when speech starts, so silence or the speaker's tail sends nothing).
  While speech plays and for `ECHO_TAIL_MS` after, speech opens a window (a barge-in) only if
  it is `[edge.vad] barge_in_margin_db` above the echo-cancelled level heard meanwhile (or loud,
  `LOUD_MARGIN_DB` above it: the board distorts a voice over its own), and not in the first
  `BARGE_IN_GRACE_MS` of the speech: the robot never answers its own voice (VAD-ECHO says why
  speech was not taken, ECHO-END what was heard while the speech played). `push_to_talk`
  keeps only push-to-talk (and the test aids), and a follow-up opens its window at once.
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
  and echoed back as the authoritative state, except that only a human unmutes: a remote
  unmute is refused (UNMUTE-REFUSED, the edge answers it is still muted) and only `/unmute` on
  the edge itself lifts a mute, which outlives reconnects (re-sent after each `welcome`).
  `timer.sync` and `duck` are acknowledged on the console (timers ring from S5 on).
- **Golden audio.** `/feed <wav>` plays a 16 kHz mono 16-bit WAV into the agent at the mic
  input point, right after capture: its 20 ms frames, paced in real time, replace the
  microphone's until it ends, so everything after capture (energy trigger, mic windows, the
  end-of-speech detector, uplink) runs as for a voice (spec "testing rule": voice input).
- **Energy trigger and end of speech** (`--energy-trigger-dbfs`, standing in for a wake-word
  engine and VAD): a few loud frames open a mic window, with the last `PRE_ROLL_MS` of audio
  sent first so the start of the utterance is not lost; it does not fire while speech plays
  nor for `ECHO_TAIL_MS` after (by capture time: the speaker's echo would barge in on
  itself). With `--vad-end-ms`, a window in which speech was heard closes after that much
  quiet (`MIC-CLOSE reason=vad_end`). Tests pass `--energy-trigger-feed-only`: the trigger
  then fires only on fed audio, so room sound cannot open a window before the golden WAV.
- **Room level.** `/level <s>` measures the real microphone for that long (`LEVEL frames
  mean_dbfs max_dbfs`).
- **Recording** (`--record-dir`): each played speech stream is also written to
  `<dir>/stream-<id>.wav` (what the speaker was given) when it is done or flushed (`RECORDED`).
- **Typed input.** Any other stdin line is sent as `text.input`.

Every line it prints is `TAG key=value ... [json]` (the format of `assistant_link.console`).
Tags: BODY, AEC-MISMATCH, LISTEN, WELCOME, RECV, SENT, WAKE, WAKE-NEAR, FOLLOW-UP, VAD-ECHO,
ECHO-END, MIC-OPEN, MIC-CLOSE, MIC-REFUSED, PRIVACY, UNMUTE-REFUSED, FLUSHED, PLAYBACK, RESULT,
SAY, SHOW, FEED, FED, LEVEL, RECORDED, BODY-ERROR, BODY-OK, CLOSED, RETRY, REFUSED,
CONSOLE-ERROR, STOPPED.
"""

import asyncio
import contextlib
import signal
import time
import wave
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import JsonValue

from assistant_contracts.body import AudioFrame, Body, LinkAware, ReportsHealth
from assistant_contracts.capabilities import Capabilities
from assistant_contracts.common import Aec
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
from assistant_edge.listen import SpeechDetector, WakeEngine, make_wake_engine
from assistant_link.client import LinkClient
from assistant_link.connection import Connection, LinkClosed, SendQueueFull
from assistant_link.console import emit, read_commands

JPEG_CHUNK_BYTES = 16 * 1024
MIC_STREAM = 0
PTT_MAX_S = 30.0
ENERGY_FRAMES = 3
"""Consecutive loud frames that fire the energy trigger."""
ECHO_TAIL_MS = 1500
"""The energy trigger stays off this long after a speech stream is done or flushed, by the
frames' capture time: the speaker still sounds after the body reports it done (the robot's
audio pipeline buffers up to about 1.2 s) and frames captured during speech may be read late,
so its own reply would otherwise fire the trigger and cut the turn it ends."""
MIC_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = MIC_RATE * 2 * FRAME_MS // 1000
PRE_ROLL_MS = 300
"""Audio before the energy trigger fired that is sent when its window opens."""
MIC_CAP_S = 120.0
"""Hard cap of any mic window (user decision 2026-10-03): `max_window_s` is clamped to it."""
WAKE_SPEECH_MS = 240
"""In a wake window, speech counts as the request once this much of it (in all) follows the
wake (the tail of the wake word itself is shorter)."""
ECHO_ALPHA = 0.1
"""Smoothing of the echo-cancelled level heard while speech plays (per 20 ms frame)."""
BARGE_IN_GRACE_MS = 1500
"""No barge-in for this long after speech starts playing: the XVF3800's echo canceller may
still be converging and let the robot's own first words through (seen on a fresh daemon)."""
LOUD_MARGIN_DB = 20.0
LOUD_SPEECH_MS = 60
"""While speech plays the XVF3800 suppresses (and distorts) the user's voice after its first
moment, so the speech detector may not take all of it for speech: sound this far above the
echo for the start run, with at least this much speech in it, is a barge-in too (the robot's
own voice leaking through as the head moves is loud too, but not speech)."""

ListenMode = Literal["wake_word", "open_mic", "push_to_talk"]
LISTEN_MODES: tuple[ListenMode, ...] = ("wake_word", "open_mic", "push_to_talk")


@dataclass
class _Window:
    reason: str
    opened_at: float
    limit_s: float
    frames: int = 0
    level_sum: float = 0.0
    level_max: float = -120.0
    heard: bool = False
    """Speech was heard in this window (energy: a frame above the threshold; speech detector:
    speech, for a wake window speech after the wake word)."""
    quiet_ms: int = 0
    vad: bool = False
    """The speech detector ends this window (listening modes, not push-to-talk)."""
    speech_ms: int = 0
    """All speech frames since the window opened (speech detector)."""
    vad_max: float = 0.0


@dataclass
class AgentOptions:
    device_id: str
    url: str
    token: str
    energy_trigger_dbfs: float | None = None
    """Open a mic window when the level stays above this (None: off)."""
    aec: Aec | None = None
    """The echo cancelling the config expects (`[edge.audio] aec`); a body with another one is
    reported (AEC-MISMATCH), not refused (None: no expectation)."""
    vad_end_ms: int | None = None
    """Close a window after this much quiet once speech was heard in it (None: off)."""
    record_dir: Path | None = None
    """Write each played speech stream to `<dir>/stream-<id>.wav` (None: off)."""
    energy_trigger_feed_only: bool = False
    """Tests: the energy trigger fires only on a golden WAV being fed (`/feed`), so room sound
    on the real microphone cannot open a window before (or instead of) the fed utterance. The
    real microphone still flows into the windows the brain opens and into push-to-talk."""
    listen: ListenMode = "push_to_talk"
    """`wake_word`, `open_mic` or `push_to_talk` (the CLI defaults to `[edge.listen] mode`)."""
    max_window_s: float = MIC_CAP_S
    """Time limit of a window opened on the edge (wake, speech, energy) and the cap of every
    window and follow-up (clamped to `MIC_CAP_S`)."""
    wake_engine: str = "openwakeword"
    wake_words: tuple[str, ...] = ("hey_jarvis",)
    wake_threshold: float = 0.4
    """Until the brain's `welcome` gives each word's threshold."""
    wake_pre_roll_ms: int = 1500
    wake_no_speech_s: float = 5.0
    phrase_model_dir: Path | None = None
    vad_threshold: float = 0.5
    vad_start_ms: int = 160
    """Speech this long in a row opens a window (open mic, follow-up); claps are shorter."""
    vad_pre_roll_ms: int = 600
    speech_end_ms: int = 700
    """The speech detector ends a window after this much no speech (`vad_end`)."""
    barge_in_margin_db: float = 8.0


class EdgeAgent:
    def __init__(self, body: Body, options: AgentOptions) -> None:
        self.body = body
        self.options = options
        self.caps = Capabilities()
        self.client = LinkClient(options.url, options.token, self._hello, self)
        self.conn: Connection | None = None
        self.window: _Window | None = None
        self.muted = False
        self._hard_mute = False
        self._cap = min(options.max_window_s, MIC_CAP_S)
        """Every window closes after at most this long (`MIC_CAP_S`)."""
        self.follow_up_max_s = 10.0
        self.mic_seq = 0
        self.speaking: dict[int, int] = {}
        """stream_id -> rate of speech streams begun and not yet done or flushed."""
        self.voiced: set[int] = set()
        """Streams of `speaking` that got audio (a text-only stream has none to play)."""
        self.flushed: set[int] = set()
        self.timers = 0
        self._loud = 0
        self._echo_until_us = 0
        """Capture time before which the energy trigger stays off (`ECHO_TAIL_MS`)."""
        self._pre_roll: deque[AudioFrame] = deque(maxlen=PRE_ROLL_MS // FRAME_MS)
        self._feeding: asyncio.Task[None] | None = None
        self._level_probe: list[float] | None = None
        self._level_speech: list[float] = []
        """Speech probabilities of the frames `/level` measured (listening modes)."""
        """Levels (dBFS) of the real microphone's frames while `/level` measures."""
        self._level_task: asyncio.Task[None] | None = None
        self._recordings: dict[int, tuple[int, bytearray]] = {}
        self._stop = asyncio.Event()
        listening = options.listen != "push_to_talk"
        self.vad: SpeechDetector | None = (
            SpeechDetector(options.vad_threshold) if listening else None
        )
        self.wake: WakeEngine | None = (
            make_wake_engine(
                options.wake_engine,
                list(options.wake_words),
                options.wake_threshold,
                options.phrase_model_dir,
            )
            if options.listen == "wake_word"
            else None
        )
        self._spoken: dict[str, str] = {}
        """Wake model -> its spoken word (from `welcome`)."""
        history = max(options.wake_pre_roll_ms, options.vad_pre_roll_ms) // FRAME_MS
        self._history: deque[AudioFrame] = deque(maxlen=max(1, history))
        """The last frames heard, for a listening window's pre-roll."""
        self._follow_until = 0.0
        """Until then (monotonic) speech opens a window without the wake word (follow-up)."""
        self._echo_level: float | None = None
        """Smoothed echo-cancelled level while speech plays (None outside the echo period)."""
        self._echo_refused = False
        self._echo: dict[str, Any] | None = None
        self._echo_floor = -120.0
        self._echo_since_us = 0
        self._loud_ms = 0
        self._loud_speech_ms = 0

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
        if self.options.aec is not None and self.caps.audio_in.aec != self.options.aec:
            emit("AEC-MISMATCH", configured=self.options.aec, body=self.caps.audio_in.aec)
        emit("LISTEN", mode=self.options.listen,
             wake=",".join(self.options.wake_words) if self.wake is not None else "-",
             engine=self.wake.name if self.wake is not None else "-",
             vad="silero" if self.vad is not None else "-")  # fmt: skip
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
        self._spoken = {w.model: w.word for w in welcome.wake_words}
        if self.wake is not None:
            self.wake.set_thresholds({w.model: w.threshold for w in welcome.wake_words})
        emit("WELCOME", dump_message(welcome), session=welcome.session_id)
        if isinstance(self.body, LinkAware):
            self.body.link_changed(True)
        if welcome.mute:
            await self._set_muted(True)
        elif self.muted:  # a mute outlives reconnects: the new session learns it
            await self.send(Privacy(muted=True, hard=self._hard_mute))

    async def on_disconnect(self, code: int | None, reason: str) -> None:
        self.conn = None
        if isinstance(self.body, LinkAware):
            self.body.link_changed(False)
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
            if self.options.record_dir is not None:
                self._recordings.setdefault(frame.stream, (rate, bytearray()))[1].extend(
                    frame.payload
                )
            await self.body.audio.play(frame.stream, frame.payload, rate)
        elif frame.kind is FrameKind.SOUND_CLIP:
            emit("CLIP", stream=frame.stream, bytes=len(frame.payload))

    async def on_message(self, conn: Connection, message: Envelope) -> None:
        del conn
        emit("RECV", dump_message(message), type=message.type)
        match message:
            case SpeakBegin():
                self.flushed.discard(message.stream_id)
                self._recordings.pop(message.stream_id, None)
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
                seconds = min(message.seconds, self._cap)
                if self.vad is None:
                    await self._open_window("follow_up", seconds)
                elif not self.muted:
                    # Listening modes: speech opens the window (no wake word needed) until then.
                    self._follow_until = time.monotonic() + seconds
                    emit("FOLLOW-UP", seconds=f"{seconds:g}")
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
                if message.muted or not self.muted:
                    await self._set_muted(message.muted)
                else:
                    # Only a human unmutes (`/unmute` on the edge): the brain's request is
                    # refused and answered with the state that holds.
                    emit("UNMUTE-REFUSED", source="brain")
                    await self.send(Privacy(muted=True, hard=self._hard_mute))
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
            self._save_recording(sid)
        took_ms = (time.monotonic() - started) * 1000
        emit("FLUSHED", stream=stream_id if stream_id is not None else "all",
             played_ms=played, local=str(local).lower(), took_ms=f"{took_ms:.1f}")  # fmt: skip
        return playing, played

    async def _playback_loop(self) -> None:
        async for event in self.body.audio.playback_events():
            if event.state in ("done", "flushed"):
                self.speaking.pop(event.stream_id, None)
                self._echo_until_us = time.monotonic_ns() // 1000 + ECHO_TAIL_MS * 1000
                self.voiced.discard(event.stream_id)
                self._save_recording(event.stream_id)
            emit("PLAYBACK", stream=event.stream_id, state=event.state, played_ms=event.played_ms)
            await self.send(
                Playback(stream_id=event.stream_id, played_ms=event.played_ms, state=event.state)
            )

    def _save_recording(self, stream_id: int) -> None:
        recording = self._recordings.pop(stream_id, None)
        if recording is None or self.options.record_dir is None:
            return
        rate, pcm = recording
        self.options.record_dir.mkdir(parents=True, exist_ok=True)
        path = self.options.record_dir / f"stream-{stream_id}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(bytes(pcm))
        emit("RECORDED", stream=stream_id, path=path, ms=len(pcm) * 1000 // (2 * rate))

    # ------------------------------------------------------------ microphone

    async def _open_window(self, reason: str, limit_s: float, *, heard: bool = False) -> None:
        if self.muted:
            emit("MIC-REFUSED", reason=reason, muted="true")
            return
        limit_s = min(limit_s, self._cap)
        if self.window is not None:
            self.window.limit_s = min(
                self._cap,
                max(self.window.limit_s, time.monotonic() - self.window.opened_at + limit_s),
            )
            return
        barge_in = bool(self.speaking)
        vad = Vad(state="start")
        if barge_in:
            stream, played = await self._flush(None, local=True)
            vad = Vad(state="start", barge_in=True, stream_id=stream, played_ms=played)
        self.window = _Window(reason, time.monotonic(), limit_s, heard=heard)
        self.window.vad = self.vad is not None and reason in ("wake", "vad", "follow_up")
        if self.window.vad:
            self._follow_until = 0.0  # a follow-up opens one window
        emit("MIC-OPEN", reason=reason, barge_in=str(barge_in).lower(), limit_s=f"{limit_s:g}")
        await self.send(vad)

    async def _close_window(self, reason: str) -> None:
        window, self.window = self.window, None
        if window is None:
            return
        mean = window.level_sum / window.frames if window.frames else -120.0
        speech = (
            {"speech_ms": window.speech_ms, "vad_max": f"{window.vad_max:.2f}"}
            if window.vad
            else {}
        )
        emit("MIC-CLOSE", reason=reason, opened_by=window.reason, frames=window.frames,
             mean_dbfs=f"{mean:.1f}", max_dbfs=f"{window.level_max:.1f}", **speech)  # fmt: skip
        await self.send(Vad(state="end"))

    async def _window_timer(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            window = self.window
            if window is None:
                continue
            open_s = time.monotonic() - window.opened_at
            if open_s > window.limit_s:
                await self._close_window("timeout")
            elif window.vad and not window.heard and open_s > self.options.wake_no_speech_s:
                await self._close_window("no_speech")

    async def _capture_loop(self) -> None:
        async for frame in self.body.audio.capture():
            if self._feeding is not None:
                continue  # a golden WAV stands in for the microphone meanwhile
            if self._level_probe is not None:
                self._level_probe.append(dbfs(frame.pcm))
            await self._on_mic(frame)

    async def _measure_level(self, seconds: float) -> None:
        """`/level <s>`: the real microphone's level over `seconds` (LEVEL)."""
        self._level_probe, self._level_speech = [], []
        try:
            await asyncio.sleep(seconds)
        finally:
            levels, self._level_probe = self._level_probe, None
        mean = sum(levels) / len(levels) if levels else -120.0
        peak = max(levels) if levels else -120.0
        speech: dict[str, object] = {}
        if self.vad is not None:  # how much of it the speech detector took for speech
            probs, threshold = self._level_speech, self.vad.threshold
            speech = {"speech_ms": FRAME_MS * sum(p >= threshold for p in probs),
                      "vad_max": f"{max(probs, default=0.0):.2f}"}  # fmt: skip
        emit("LEVEL", frames=len(levels), seconds=seconds, mean_dbfs=f"{mean:.1f}",
             max_dbfs=f"{peak:.1f}", **speech)  # fmt: skip

    async def _feed(self, path: str) -> None:
        """Play a 16 kHz mono 16-bit WAV in at the mic input point, in real time."""
        try:
            with wave.open(path) as wav:
                shape = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
                pcm = wav.readframes(wav.getnframes())
        except (OSError, wave.Error, EOFError) as exc:
            emit("CONSOLE-ERROR", {"detail": f"feed {path}: {type(exc).__name__}: {exc}"})
            return
        if shape != (MIC_RATE, 1, 2):
            emit("CONSOLE-ERROR", {"detail": f"feed {path}: need 16 kHz mono 16-bit, got {shape}"})
            return
        frames = (len(pcm) + FRAME_BYTES - 1) // FRAME_BYTES
        emit("FEED", path=path, frames=frames, ms=frames * FRAME_MS)
        started = time.monotonic()
        for index in range(frames):
            chunk = pcm[index * FRAME_BYTES : (index + 1) * FRAME_BYTES].ljust(FRAME_BYTES, b"\0")
            frame = AudioFrame(pcm=chunk, capture_ts_us=time.monotonic_ns() // 1000)
            await self._on_mic(frame, fed=True)
            due = started + (index + 1) * FRAME_MS / 1000
            await asyncio.sleep(max(0.0, due - time.monotonic()))
        emit("FED", path=path, frames=frames, took_ms=f"{(time.monotonic() - started) * 1000:.0f}")

    def _start_feed(self, path: str) -> None:
        if self._feeding is not None:
            raise ValueError("a WAV is already being fed")

        async def run() -> None:
            try:
                await self._feed(path)
            finally:
                self._feeding = None

        self._feeding = asyncio.create_task(run(), name="feed")

    async def _on_mic(self, frame: AudioFrame, *, fed: bool = False) -> None:
        level = dbfs(frame.pcm)
        if self.vad is not None and not self.muted and await self._listen(frame, level):
            return  # a window opened with this frame as the last of its pre-roll
        threshold = self.options.energy_trigger_dbfs
        armed = fed or not self.options.energy_trigger_feed_only
        if self.window is None and threshold is not None and armed and not self.muted:
            # Not while speech plays, nor its tail: the speaker's own echo would trigger a
            # barge-in.
            loud = (
                level > threshold
                and not self.speaking
                and frame.capture_ts_us >= self._echo_until_us
            )
            self._loud = self._loud + 1 if loud else 0
            self._pre_roll.append(frame)
            if self._loud >= ENERGY_FRAMES:
                self._loud = 0
                score = min(1.0, max(0.0, (level - threshold) / 30 + 0.5))
                await self.send(Wake(word="energy", score=round(score, 3)))
                await self._open_window("energy", self.options.max_window_s)
                pre_roll = list(self._pre_roll)
                self._pre_roll.clear()
                for earlier in pre_roll:  # this frame included, as the last one
                    await self._uplink(earlier, dbfs(earlier.pcm))
                return
        await self._uplink(frame, level)

    async def _uplink(self, frame: AudioFrame, level: float, *, live: bool = True) -> None:
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
        if window.vad and self.vad is not None:
            if live:  # the speech detector has already heard the pre-roll
                await self._vad_window(window)
            return
        threshold, end_ms = self.options.energy_trigger_dbfs, self.options.vad_end_ms
        if threshold is None or end_ms is None:
            return
        if level > threshold:
            window.heard, window.quiet_ms = True, 0
        elif window.heard:
            window.quiet_ms += FRAME_MS
            if window.quiet_ms >= end_ms:
                await self._close_window("vad_end")

    # ------------------------------------------------------------ listening

    def _in_echo(self, frame: AudioFrame) -> bool:
        """Speech is playing, or its tail may still sound (by the frame's capture time)."""
        return bool(self.speaking) or frame.capture_ts_us < self._echo_until_us

    async def _listen(self, frame: AudioFrame, level: float) -> bool:
        """Listening modes: run the speech detector and the wake-word engine on a frame and
        open a window on a wake or (open mic, follow-up) on speech. True if a window opened
        (its pre-roll, this frame included, is already uplink)."""
        assert self.vad is not None
        prob = self.vad.feed(frame.pcm)
        self._history.append(frame)
        if self._level_probe is not None:
            self._level_speech.append(prob)
        echo = self._in_echo(frame)
        self._echo_stats(echo, prob, level, frame.capture_ts_us)
        if not echo:
            self._echo_level = None
        elif self._echo_level is None:
            self._echo_level = level
        if self._echo_level is not None:
            self._echo_floor = self._echo_level
        baseline = self._echo_level
        near = baseline is not None and level >= baseline + self.options.barge_in_margin_db
        if baseline is not None and prob < self.vad.threshold and not near:
            # Not speech: what the echo canceller lets through of the robot's own voice.
            self._echo_level = (1 - ECHO_ALPHA) * baseline + ECHO_ALPHA * level
        if baseline is not None and level >= baseline + LOUD_MARGIN_DB:
            self._loud_ms += FRAME_MS
            self._loud_speech_ms += FRAME_MS if prob >= self.vad.threshold else 0
        else:
            self._loud_ms, self._loud_speech_ms = 0, 0
        loud = self._loud_ms >= self.options.vad_start_ms and self._loud_speech_ms >= LOUD_SPEECH_MS
        if self.wake is not None:
            hit = self.wake.feed(frame.pcm)
            for word, score, threshold in self.wake.near_misses():
                emit("WAKE-NEAR", word=word, score=f"{score:.2f}", threshold=threshold)
            if hit is not None and self.window is None:
                word = self._spoken.get(hit.model, hit.model.replace("_", " "))
                emit("WAKE", word=word, model=hit.model, score=f"{hit.score:.3f}",
                     engine=self.wake.name)  # fmt: skip
                await self.send(Wake(word=word, score=round(min(1.0, hit.score), 3)))
                await self._open_window("wake", self.options.max_window_s)
                await self._send_pre_roll(self.options.wake_pre_roll_ms)
                return True
        if self.window is not None or not (self.vad.run_ms >= self.options.vad_start_ms or loud):
            self._echo_refused = False
            return False
        follow_up = time.monotonic() < self._follow_until
        if self.options.listen != "open_mic" and not follow_up:
            return False
        if echo:
            frames = self.options.vad_start_ms // FRAME_MS
            recent = list(self._history)[-frames:]
            speech = sum(dbfs(f.pcm) for f in recent) / len(recent)
            echo_dbfs = baseline if baseline is not None else -120.0
            floor = echo_dbfs + self.options.barge_in_margin_db
            refused = None
            if frame.capture_ts_us - self._echo_since_us < BARGE_IN_GRACE_MS * 1000:
                refused = "grace"
            elif not loud and speech < floor:
                refused = "echo"
            if refused is not None:
                if not self._echo_refused:
                    self._echo_refused = True
                    emit("VAD-ECHO", reason=refused, speech_dbfs=f"{speech:.1f}",
                         echo_dbfs=f"{echo_dbfs:.1f}", need_dbfs=f"{floor:.1f}",
                         prob=f"{prob:.2f}", loud=str(loud).lower())  # fmt: skip
                return False
        reason = "vad" if self.options.listen == "open_mic" and not follow_up else "follow_up"
        await self._open_window(reason, self.options.max_window_s, heard=True)
        await self._send_pre_roll(self.options.vad_pre_roll_ms)
        return True

    def _echo_stats(self, echo: bool, prob: float, level: float, ts_us: int) -> None:
        """While speech plays (and its tail): what the microphone heard, printed when it ends
        (ECHO-END: the echo-cancelled level, and how much the speech detector took for
        speech; a barge-in shows here as speech above the echo)."""
        if echo:
            if self._echo is None:
                self._echo_since_us = ts_us
            stats = self._echo or {"frames": 0, "speech_ms": 0, "vad_max": 0.0, "max_dbfs": -120.0}
            stats["frames"] += 1
            stats["speech_ms"] += FRAME_MS if prob >= self.options.vad_threshold else 0
            stats["vad_max"] = max(stats["vad_max"], prob)
            stats["max_dbfs"] = max(stats["max_dbfs"], level)
            self._echo = stats
        elif self._echo is not None:
            stats, self._echo = self._echo, None
            emit("ECHO-END", frames=stats["frames"], speech_ms=stats["speech_ms"],
                 vad_max=f"{stats['vad_max']:.2f}", max_dbfs=f"{stats['max_dbfs']:.1f}",
                 echo_dbfs=f"{self._echo_floor:.1f}")  # fmt: skip

    async def _send_pre_roll(self, ms: int) -> None:
        frames = list(self._history)[-max(1, ms // FRAME_MS) :]
        for index, earlier in enumerate(frames):
            await self._uplink(earlier, dbfs(earlier.pcm), live=index == len(frames) - 1)

    async def _vad_window(self, window: _Window) -> None:
        """End a listening window once speech was heard and then stopped (`vad_end`)."""
        assert self.vad is not None
        window.vad_max = max(window.vad_max, self.vad.prob)
        if self.vad.prob >= self.vad.threshold:
            window.speech_ms += FRAME_MS
        if not window.heard and window.speech_ms >= WAKE_SPEECH_MS:
            window.heard = True
        end_ms = self.options.speech_end_ms
        if window.heard and self.vad.quiet_ms >= end_ms:
            await self._close_window("vad_end")

    async def _set_muted(self, muted: bool, *, hard: bool = False) -> None:
        """`hard`: muted by a human at the edge (`/mute`)."""
        self.muted = muted
        self._hard_mute = muted and (hard or self._hard_mute)
        if muted:
            await self._close_window("muted")
        emit("PRIVACY", muted=str(muted).lower(), hard=str(self._hard_mute).lower())
        await self.send(Privacy(muted=muted, hard=self._hard_mute))

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
                await self._open_window("wake", self.options.max_window_s)
            case "wake", [word]:
                await self.send(Wake(word=word, score=1.0))
                await self._open_window("wake", self.options.max_window_s)
            case "feed", [path]:
                self._start_feed(path)
            case "level", [seconds]:
                probe = float(seconds)
                self._level_task = self._spawn(lambda: self._measure_level(probe), "level")
            case "mute", []:
                await self._set_muted(True, hard=True)
            case "unmute", []:
                await self._set_muted(False)
            case _:
                raise ValueError(
                    f"unknown command {line!r}: /ptt down|up, /wake <word> [score], "
                    "/feed <wav>, /level <s>, /mute, /unmute, /quit, or plain text"
                )
        return True
