"""DialogManager: one per EdgeSession. Turn state, follow-up window, proactive speech queue.

Turn states and the attention the edge is told (through `body.intent` -> BodyController):

    idle --(wake / vad start / text.input)--> listening --(input complete)--> thinking
    thinking --(first reply out)--> speaking --(reply played)--> idle
                                                           \\--> listening (voice turn: the
                                                                follow-up window) --> idle

- **Input.** A wake or `vad{start}` opens listening and the brain collects the uplink audio
  (0x01) until `vad{end}`; audio with speech in it becomes a voice turn (silence ends the
  window). `text.input` is a text turn. New input during a turn interrupts it.
- **Reply.** A voice turn's transcript (`InputTranscript`) becomes the turn's input text. The
  engine's reply text goes out as `speak.begin{text}` + `speak.end` when there is no audio
  (no speech server); reply audio as `speak.begin{first sentence}` + 0x02 frames +
  `speak.end`, and the turn then waits for the edge's playback clock to report the stream
  `done` (or `flushed`).
- **Barge-in.** `vad{start, barge_in}` (the edge already flushed its speaker) or new input
  cancels the running turn (LLM and TTS included) and tells the engine how much was heard
  (`played_ms`); the engine's cut goes in the turn log (`truncated`, `TURN-TRUNCATED`).
- **Follow-up.** After a voice turn that was answered the brain sends
  `mic.follow_up{follow_up_s}` and stays listening; a window that ends without speech (or
  `follow_up_s` + a grace, a clock that speech starting stops) returns to idle. A voice turn
  in which nothing was heard (empty transcript) gets no reply and no window.
- **Proactive speech** (`speech.request`): spoken at once when idle, queued while a turn or
  window is active and spoken when the session is idle again, dropped while muted or once
  its `ttl_s` has passed.
- **Errors.** A model server that is down ends the turn (any reply audio already out is ended
  first) with a spoken (speak text) apology and attention back to idle.
"""

import asyncio
import contextlib
import math
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal

from assistant_brain.bus import EventBus
from assistant_brain.console import emit
from assistant_brain.engine import (
    MIC_RATE,
    EngineSession,
    EngineUnavailable,
    InputTranscript,
    ReplyAudio,
    ReplyText,
    TurnMetrics,
    UserTurn,
)
from assistant_brain.turnlog import TurnLog, TurnRecord, TurnState
from assistant_contracts.events import (
    BodyIntent,
    SpeechFinished,
    SpeechRequest,
    SpeechStarted,
    TurnFinished,
    TurnInterrupted,
    TurnStarted,
    TurnUserText,
)
from assistant_contracts.frames import Frame, FrameKind
from assistant_contracts.messages import (
    Envelope,
    Flush,
    MicFollowUp,
    Playback,
    SpeakBegin,
    SpeakEnd,
    TextInput,
    Vad,
)

if TYPE_CHECKING:
    from assistant_brain.sessions import EdgeSession

FRAME_MS = 20
PLAYBACK_GRACE_S = 5.0
"""How long after the reply's audio should have played the turn still waits for `done`."""
FOLLOW_UP_GRACE_S = 1.0
SPEECH_DBFS = -50.0
"""Collected audio whose loudest 20 ms frame stays below this is silence, not a turn.
TODO(S8): Silero VAD + Smart Turn endpointing instead of this level gate."""


def loudest_dbfs(pcm: bytes) -> float:
    """The level of the loudest 20 ms frame of s16le mono 16 kHz audio (dBFS)."""
    step = MIC_RATE * 2 * FRAME_MS // 1000
    loudest = -120.0
    for offset in range(0, len(pcm) - 1, step):
        chunk = memoryview(pcm[offset : offset + step]).cast("h")
        if not len(chunk):
            continue
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
        if rms > 0:
            loudest = max(loudest, 20 * math.log10(rms / 32768.0))
    return loudest


def audio_ms(pcm: bytes, rate: int) -> int:
    return len(pcm) * 1000 // (2 * rate)


class DialogManager:
    def __init__(
        self,
        session: "EdgeSession",
        engine: EngineSession,
        engine_name: str,
        bus: EventBus,
        turns: TurnLog,
        follow_up_s: float,
        send: Callable[[Envelope], Awaitable[bool]],
    ) -> None:
        self.session = session
        self.engine = engine
        self.engine_name = engine_name
        self.bus = bus
        self.turns = turns
        self.follow_up_s = follow_up_s
        self.send = send
        self.state: TurnState = "idle"
        self.muted = False
        self.record: TurnRecord | None = None
        self._task: asyncio.Task[None] | None = None
        self._mic: bytearray | None = None
        self._follow_up: asyncio.Task[None] | None = None
        self._playback: dict[int, asyncio.Future[str]] = {}
        self._stream = 0
        self._speaking_stream: int | None = None
        self._metrics: TurnMetrics | None = None
        self.queue: deque[tuple[SpeechRequest, float]] = deque()

    # ------------------------------------------------------------ state

    @property
    def busy(self) -> bool:
        """A turn runs, or the user may be talking (listening / follow-up window)."""
        return (self._task is not None and not self._task.done()) or self.state != "idle"

    async def set_state(self, state: TurnState) -> None:
        self.state = state
        if self.record is not None and self.record.outcome is None:
            self.record.state(state)
        await self.bus.publish(
            BodyIntent(
                device_id=self.session.device_id,
                kind="attention",
                state=state,
                assistant=self.session.assistant.id,
            )
        )

    async def start(self) -> None:
        """The edge was (re)welcomed: tell it the state (idle; any old turn was abandoned)."""
        await self.set_state("idle")

    async def close(self) -> None:
        """The edge went away: abandon the turn and the window, drop queued speech."""
        self._cancel_follow_up()
        await self._cancel_turn("abandoned")
        self.queue.clear()
        await self.engine.close()

    def _new_stream(self) -> int:
        self._stream = self._stream % 255 + 1
        return self._stream

    # ------------------------------------------------------------ edge input

    async def on_wake(self) -> None:
        self._cancel_follow_up()
        if self._turn_running():
            await self.interrupt(None)
        self._mic = bytearray()
        await self.set_state("listening")

    async def on_vad(self, vad: Vad) -> None:
        if vad.state == "start":
            # Speech started: the window is the edge's now (it ends it), not the follow-up
            # timer's, which would drop speech that runs past it.
            self._cancel_follow_up()
            if vad.barge_in or self._turn_running():
                await self.interrupt(vad.played_ms)
            if self._mic is None:
                self._mic = bytearray()
            if self.state != "listening":
                await self.set_state("listening")
            return
        audio, self._mic = self._mic, None
        if audio is None:
            return
        self._cancel_follow_up()
        if audio and loudest_dbfs(bytes(audio)) >= SPEECH_DBFS:
            await self._start_turn("voice", audio=bytes(audio))
        elif not self._turn_running():
            await self._become_idle()

    async def on_mic_frame(self, frame: Frame) -> None:
        if frame.kind is FrameKind.MIC_PCM and self._mic is not None:
            self._mic += frame.payload
            await self.engine.push_audio(frame.payload)

    async def on_text(self, message: TextInput) -> None:
        self._cancel_follow_up()
        self._mic = None
        if self._turn_running():
            await self.interrupt(None)
        await self._start_turn("text", text=message.text)

    async def on_playback(self, message: Playback) -> None:
        if message.state in ("done", "flushed"):
            waiter = self._playback.pop(message.stream_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(message.state)

    async def on_privacy(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            self._cancel_follow_up()
            self._mic = None
            self.queue.clear()

    # ------------------------------------------------------------ proactive speech

    async def request_speech(self, request: SpeechRequest) -> None:
        if self.muted:
            emit("SPEECH-DROPPED", device=self.session.device_id, reason="muted")
            return
        if self.busy:
            self.queue.append((request, time.monotonic()))
            emit(
                "SPEECH-QUEUED",
                {"text": request.text, "prompt": request.prompt},
                device=self.session.device_id,
                state=self.state,
                queued=len(self.queue),
            )
            return
        await self._start_turn("proactive", request=request)

    async def _drain_queue(self) -> None:
        while self.queue and not self.busy:
            request, queued_at = self.queue.popleft()
            if request.ttl_s is not None and time.monotonic() - queued_at > request.ttl_s:
                emit("SPEECH-DROPPED", device=self.session.device_id, reason="expired")
                continue
            await self._start_turn("proactive", request=request)

    # ------------------------------------------------------------ turns

    def _turn_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _start_turn(
        self,
        kind: Literal["text", "voice", "proactive"],
        *,
        text: str | None = None,
        audio: bytes | None = None,
        request: SpeechRequest | None = None,
    ) -> None:
        turn_id = f"t-{uuid.uuid4().hex[:10]}"
        record = TurnRecord(
            turn_id=turn_id,
            session_id=self.session.session_id,
            device_id=self.session.device_id,
            assistant=self.session.assistant.id,
            engine=self.engine_name,
            kind=kind,
            input_text=text if request is None else (request.text or request.prompt),
            input_audio_ms=audio_ms(audio, MIC_RATE) if audio else 0,
        )
        self.record = self.turns.start(record)
        self._task = asyncio.create_task(self._run(record, text, audio, request), name=turn_id)

    async def _run(
        self,
        record: TurnRecord,
        text: str | None,
        audio: bytes | None,
        request: SpeechRequest | None,
    ) -> None:
        sid, tid = self.session.session_id, record.turn_id
        await self.bus.publish(
            TurnStarted(session_id=sid, turn_id=tid, device_id=self.session.device_id)
        )
        outcome: Literal["finished", "error"] = "finished"
        error: str | None = None
        metrics = TurnMetrics()
        self._metrics = metrics
        try:
            if request is not None and request.mode == "verbatim":
                assert request.text is not None
                await self._speak_text(record, request.text)
            else:
                if record.kind != "proactive":
                    if self.state != "listening":
                        await self.set_state("listening")
                    else:
                        record.state("listening")  # the mic window opened before the turn
                    if text:
                        await self.bus.publish(TurnUserText(session_id=sid, turn_id=tid, text=text))
                await self.set_state("thinking")
                turn = UserTurn(
                    turn_id=tid,
                    text=text if request is None else request.prompt,
                    audio=audio,
                    request_class="proactive" if request is not None else "voice",
                    assistant=self.session.assistant,
                )
                try:
                    await self._reply(record, turn, metrics)
                except EngineUnavailable as exc:
                    outcome, error = "error", str(exc)
                    emit("BRAIN-ERROR", {"detail": error}, device=self.session.device_id, turn=tid)
                    await self._speak_text(record, exc.spoken)
        finally:
            record.llm = {
                k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in {
                    "request_id": metrics.llm_request_id,
                    "queued_ms": metrics.llm_queued_ms,
                    "ttft_ms": metrics.llm_ttft_ms,
                    "total_ms": metrics.llm_total_ms,
                }.items()
                if v is not None
            }
            record.speech = {
                k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in {
                    "stt_ms": metrics.stt_ms,
                    "tts_first_audio_ms": metrics.tts_first_audio_ms,
                    "first_audio_ms": metrics.first_audio_ms,
                    "voice": metrics.voice,
                    "tts_requests": metrics.tts_requests or None,
                }.items()
                if v is not None
            }
        await self.bus.publish(TurnFinished(session_id=sid, turn_id=tid))
        self.turns.end(record, outcome, error)
        answered = bool(record.reply_text) or record.reply_audio_ms > 0
        if record.kind == "voice" and answered and self.follow_up_s > 0 and not self.muted:
            await self._open_follow_up()
        else:
            await self._become_idle()

    async def _reply(self, record: TurnRecord, turn: UserTurn, metrics: TurnMetrics) -> None:
        stream: int | None = None
        rate = 0
        sent_ms = 0
        texts: list[str] = []
        events = self.engine.respond(turn, metrics)
        try:
            async for event in events:
                if isinstance(event, InputTranscript):
                    await self._transcript(record, event.text)
                    continue
                if isinstance(event, ReplyText):
                    texts.append(event.text)
                    record.reply_text = "".join(texts)
                    continue
                assert isinstance(event, ReplyAudio)
                if stream is None:
                    stream, rate = self._new_stream(), event.rate
                    await self._begin(record, stream, rate, "".join(texts).strip() or None)
                sent_ms += await self._send_audio(stream, event.pcm, rate)
                record.reply_audio_ms = sent_ms
        except EngineUnavailable:
            if stream is not None:  # end the reply that was already playing, then apologise
                await self._end(stream)
                await self._wait_played(stream, sent_ms)
            raise
        finally:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()  # a cancelled turn stops the engine's reply (LLM, TTS) now
        reply = "".join(texts).strip()
        record.reply_text = reply
        if stream is None:
            if reply:
                await self._speak_text(record, reply)
            return
        record.reply_audio_ms = sent_ms
        await self._end(stream)
        await self._wait_played(stream, sent_ms)

    async def _transcript(self, record: TurnRecord, text: str) -> None:
        record.input_text = text
        emit("TRANSCRIPT", {"text": text}, device=self.session.device_id, turn=record.turn_id)
        if text:
            await self.bus.publish(
                TurnUserText(session_id=self.session.session_id, turn_id=record.turn_id, text=text)
            )

    async def _begin(self, record: TurnRecord, stream: int, rate: int, text: str | None) -> None:
        if self.state != "speaking":
            await self.set_state("speaking")
        self._speaking_stream = stream
        self._playback[stream] = asyncio.get_running_loop().create_future()
        shown = text if self.session.capabilities.speak_text else None
        await self.send(SpeakBegin(stream_id=stream, rate=rate, text=shown, turn_id=record.turn_id))
        emit(
            "SPEAK",
            {"text": text},
            device=self.session.device_id,
            stream=stream,
            turn=record.turn_id,
            rate=rate,
        )
        await self.bus.publish(
            SpeechStarted(
                device_id=self.session.device_id,
                stream_id=stream,
                assistant_id=self.session.assistant.id,
            )
        )

    async def _end(self, stream: int) -> None:
        await self.send(SpeakEnd(stream_id=stream))
        await self.bus.publish(
            SpeechFinished(
                device_id=self.session.device_id,
                stream_id=stream,
                assistant_id=self.session.assistant.id,
            )
        )

    async def _send_audio(self, stream: int, pcm: bytes, rate: int) -> int:
        step = rate * 2 * FRAME_MS // 1000
        conn = self.session.conn
        for seq, offset in enumerate(range(0, len(pcm), step)):
            chunk = pcm[offset : offset + step]
            frame = Frame(FrameKind.OUT_PCM, stream, seq, time.monotonic_ns() // 1000, chunk)
            await conn.send_frame(frame)
        return audio_ms(pcm, rate)

    async def _speak_text(self, record: TurnRecord, text: str) -> None:
        """A reply without audio: `speak.begin{text}` + `speak.end` (no playback to wait for)."""
        stream = self._new_stream()
        if not record.reply_text:
            record.reply_text = text
        await self._begin(record, stream, 24000, text)
        self._playback.pop(stream, None)
        await self._end(stream)

    async def _wait_played(self, stream: int, ms: int) -> None:
        waiter = self._playback.get(stream)
        if waiter is None:
            return
        try:
            await asyncio.wait_for(waiter, ms / 1000 + PLAYBACK_GRACE_S)
        except TimeoutError:
            emit(
                "BRAIN-ERROR",
                {"detail": f"no playback done for stream {stream}"},
                device=self.session.device_id,
            )
        finally:
            self._playback.pop(stream, None)
            self._speaking_stream = None

    async def interrupt(self, played_ms: int | None) -> None:
        """Barge-in or new input: cancel the running turn and tell the engine what was heard."""
        record = self.record
        metrics = self._metrics
        stream = self._speaking_stream
        if not await self._cancel_turn("interrupted", end=False):
            return
        if stream is not None:
            await self.send(Flush(stream_id=stream))
        await self.engine.interrupt(played_ms)
        if record is not None:
            if metrics is not None and metrics.truncation is not None:
                record.truncated = metrics.truncation
                emit(
                    "TURN-TRUNCATED",
                    metrics.truncation,
                    device=self.session.device_id,
                    turn=record.turn_id,
                    played_ms=played_ms,
                )
            self.turns.end(record, "interrupted")
            await self.bus.publish(
                TurnInterrupted(
                    session_id=self.session.session_id,
                    turn_id=record.turn_id,
                    played_ms=played_ms,
                )
            )

    async def _cancel_turn(
        self, outcome: Literal["interrupted", "abandoned"], *, end: bool = True
    ) -> bool:
        task, self._task = self._task, None
        if task is None or task.done():
            return False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if end and self.record is not None:
            self.turns.end(self.record, outcome)
        for waiter in self._playback.values():
            waiter.cancel()
        self._playback.clear()
        self._speaking_stream = None
        return True

    # ------------------------------------------------------------ follow-up window

    async def _open_follow_up(self) -> None:
        await self.send(MicFollowUp(seconds=self.follow_up_s))
        await self.set_state("listening")
        emit("FOLLOW-UP", device=self.session.device_id, seconds=self.follow_up_s)
        self._follow_up = asyncio.create_task(self._follow_up_timer(), name="follow-up")

    async def _follow_up_timer(self) -> None:
        await asyncio.sleep(self.follow_up_s + FOLLOW_UP_GRACE_S)
        self._follow_up = None
        if not self._turn_running() and self.state == "listening":
            self._mic = None
            await self._become_idle()

    def _cancel_follow_up(self) -> None:
        task, self._follow_up = self._follow_up, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _become_idle(self) -> None:
        await self.set_state("idle")
        await self._drain_queue()
