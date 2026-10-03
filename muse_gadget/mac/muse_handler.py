"""MuseHandler: a ConversationHandler for Pollen's conversation app that talks to Muse.

One turn:

    robot mic -> VAD (one utterance) -> local STT on the Mac -> POST /turn on the Muse bridge
    (127.0.0.1) -> reply text -> TTS (Qwen3-TTS streamed, Kokoro-82M sentence by sentence, or
    macOS `say`; see muse_tts.py) -> PCM frames on output_queue -> the app plays them on the
    robot's speaker and the daemon's wobbler moves the head. Playback starts as soon as the first
    chunk (Qwen3) or sentence (Kokoro) is ready.

Floor control (on by default; MUSE_BARGE_IN=0, from run_poc.sh --no-barge-in, turns it off): the
mic is always open, while the user talks, while the turn is transcribed and Muse thinks, while the
robot moves and while it speaks, trusting the XVF3800's echo cancellation (Pollen's app applies its
tuned startup config). Whenever the VAD finds the start of speech, the user has the floor:
* a move Muse started (dance, emotion, head look) stops at once, through the movement manager's
  clear_move_queue (what Pollen's stop_dance/stop_emotion tools call; the head holds where it is,
  no new motion), and Muse's later moves from that turn are refused. Face tracking isn't a move.
* if the reply is playing, it's a barge-in: the robot keeps talking for MUSE_BARGE_IN_STOP_MS
  (default 350 ms), then fades out over 80 ms, drops the rest of the reply (the bridge stream is
  closed, so Muse's turn ends there) and the new utterance is the next turn.
* if the reply hasn't started, nothing is cancelled: Muse can't take back a message it has been
  sent, so its answer is kept. If the user is still talking when it's ready, it waits until they
  finish (no talking over them), then plays; their new words are the next turn.
The robot's own voice is held off by the VAD's start rule (about 96 ms of speech in a row). Without
barge-in the mic is closed from the end of an utterance until the reply has played (half-duplex).

Speech-to-text: the turn's text comes from one pass over the whole utterance, after the end
silence. MUSE_STT_AT_PAUSES=1 (run_poc.sh --stt-at-pauses) adds early passes: when the speaker
pauses, the utterance so far is transcribed at once, on the STT thread; if the pause turns out to be
the end, that text is the turn's text (ready at end of speech instead of a pass later), and a check
pass over the full utterance runs once the reply starts and logs whether the texts match (yes/no).
Either way the utterance's boundaries are the VAD's alone: a pause pass never ends or splits one.

Transcripts go to the app only through `_emit_transcript` (its UI/JSON-RPC push, never logged). They are not put on the
output queue as AdditionalOutputs unless transcript logging is turned on (`log_transcripts=True`,
or MUSE_LOG_TRANSCRIPTS=1 from `run_app.py --log-transcripts`), because the app logs those at INFO
as `role=... content=<text>`. MuseHandler's own log lines hold only lengths and timings.

Robot tools: when the run gives a secret (MUSE_ROBOT_TOOLS_SECRET_FILE, see robot_tools.py), the
handler also serves the robot-tools endpoint on 127.0.0.1 while it's up, so Muse's `reachy.*`
gadget commands can run a few of Pollen's own tools (emotions, dances, look, face tracking)
through the app's BackgroundToolManager, as the Hugging Face backend does for its model.

The STT engine, bridge client and TTS are injectable so the unit tests run on the PC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np

from reachy_mini_conversation_app.conversation_handler import AudioFrame, ConversationHandler
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.background_tool_manager import (
    BackgroundToolManager,
    ToolCallRoutine,
    ToolNotification,
)
from reachy_mini_conversation_app.tools.tool_constants import ToolState

from muse_bridge import BridgeClient, BridgeError, spoken_error
from muse_tts import Tts
from muse_vad import SAMPLE_RATE, WINDOW, UtteranceSegmenter, end_silence_from_env, make_vad, to_mono_16k
import robot_tools

logger = logging.getLogger(__name__)

Transcriber = Callable[[np.ndarray], str]
Synthesizer = Callable[[str, int], np.ndarray]  # (text, sample_rate) -> int16 mono PCM

FRAME_S = 0.2  # seconds of reply audio per output_queue item
TAIL_S = 0.6  # keep the mic closed this long after the reply should have finished playing
BARGE_IN_ENV = "MUSE_BARGE_IN"  # 0: half-duplex (no barge-in)
BARGE_IN_STOP_ENV = "MUSE_BARGE_IN_STOP_MS"
BARGE_IN_STOP_MS = 350  # keep speaking this long after a barge-in is detected, then fade out
FADE_S = 0.08
EARLY_AFTER_WINDOWS = 3  # a pause this long (3 x 32 ms) gets an early speech-to-text pass
STT_AT_PAUSES_ENV = "MUSE_STT_AT_PAUSES"  # 1: early speech-to-text passes at pauses (opt-in)
MOVE_GRACE_S = 0.5  # a move just queued may not show in the movement manager's state yet


def barge_in_from_env() -> bool:
    return os.environ.get(BARGE_IN_ENV, "1").strip() != "0"


def barge_in_stop_ms_from_env() -> int:
    try:
        value = int(os.environ.get(BARGE_IN_STOP_ENV) or BARGE_IN_STOP_MS)
    except ValueError:
        return BARGE_IN_STOP_MS
    return value if 0 <= value <= 2000 else BARGE_IN_STOP_MS


class _Turn:
    """One user utterance on its way to Muse."""

    __slots__ = ("utterance", "ended", "early")

    def __init__(self, utterance: np.ndarray, ended: float, early: Optional[asyncio.Future] = None):
        self.utterance, self.ended, self.early = utterance, ended, early


DEFAULT_VOICE = "system"
# Pollen's move tools return as soon as the move is queued ("queued", "looking left"). The robot
# moves and speaks the reply at the same time, so Muse is told the move is in progress.
MOVE_TOOLS = frozenset({"dance", "play_emotion", "move_head"})
IN_PROGRESS = "in progress"


class CallableTts:
    """A plain (text, rate) -> PCM function as a one-chunk Tts (the tests' fake engines)."""

    name = "injected"

    def __init__(self, synthesizer: Synthesizer) -> None:
        self.synthesizer = synthesizer
        self.voice: Optional[str] = None

    def chunks(self, text: str, rate: int):
        yield self.synthesizer(text, rate)

    def voices(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


class MuseHandler(ConversationHandler):
    """Speech in, Muse's reply spoken (Qwen3-TTS, Kokoro or macOS `say`) out."""

    def __init__(
        self,
        deps: Any,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
        *,
        bridge: Optional[BridgeClient] = None,
        transcriber: Optional[Transcriber] = None,
        synthesizer: Optional[Synthesizer] = None,
        tts: Optional[Tts] = None,
        segmenter: Optional[UtteranceSegmenter] = None,
        output_sample_rate: Optional[int] = None,
        log_transcripts: Optional[bool] = None,
        robot_tools_secret: Optional[str] = None,
        robot_tools_port: Optional[int] = None,
        barge_in: Optional[bool] = None,
        barge_in_stop_ms: Optional[int] = None,
        stt_at_pauses: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.deps = deps
        self.instance_path = instance_path
        # Started only with the robot-tools endpoint: Muse's reachy.* commands run Pollen's tools.
        self.tool_manager = BackgroundToolManager()
        self._robot_tools_secret = robot_tools_secret if robot_tools_secret is not None else robot_tools.read_secret()
        self._robot_tools_port = robot_tools_port if robot_tools_port is not None else robot_tools.port_from_env()
        self._robot_tools_server: Optional[asyncio.AbstractServer] = None
        self._tool_waiters: dict[str, asyncio.Future] = {}
        self._turn_tools: Counter[str] = Counter()  # robot tool calls in the current turn (names only)
        self._turn_tool_s = 0.0  # their total time, start to result
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.connection: Optional[str] = None  # LocalStream reads this to show "connected"
        self.bridge = bridge or BridgeClient()
        self.transcriber = transcriber
        self.stt_engine = "injected" if transcriber else None
        self._owns_stt = False  # loaded by start_up (so a worker process is closed by shutdown)
        self._startup_voice = None if startup_voice in (None, "", DEFAULT_VOICE) else startup_voice
        self.tts: Optional[Tts] = tts or (CallableTts(synthesizer) if synthesizer else None)
        self._owns_tts = False  # loaded by start_up (so closed by shutdown)
        # MLX streams are per thread: parakeet must run on the thread that loaded it.
        self._stt_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="muse-stt")
        self.segmenter = segmenter or UtteranceSegmenter(make_vad(), silence_s=end_silence_from_env())
        self._output_rate = output_sample_rate
        if log_transcripts is None:
            log_transcripts = os.environ.get("MUSE_LOG_TRANSCRIPTS") == "1"
        self.log_transcripts = log_transcripts  # opt-in: the app would log the text
        self._utterances: asyncio.Queue[_Turn] = asyncio.Queue()
        self._muse_moved_at: Optional[float] = None  # a move Muse started may be running
        self.barge_in = barge_in_from_env() if barge_in is None else barge_in
        self.barge_in_stop_s = (barge_in_stop_ms_from_env() if barge_in_stop_ms is None else barge_in_stop_ms) / 1000
        self._armed = False  # barge-in: the mic is open because this turn's reply is being spoken
        self._turn_active = False
        self._cut = False  # barge-in: the rest of this reply is dropped
        self._barge_at: Optional[float] = None
        self._segments: list[tuple[float, np.ndarray, int]] = []  # (play start, pcm, rate) of this reply
        self._echo_peak = 0.0  # highest VAD score while the robot spoke (self-interrupt margin)
        self._echo_open = False
        self._early: Optional[tuple[asyncio.Future, int]] = None  # (STT pass at a pause, speech windows)
        if stt_at_pauses is None:
            stt_at_pauses = os.environ.get(STT_AT_PAUSES_ENV) == "1"
        self.stt_at_pauses = stt_at_pauses  # early passes at pauses; off: one pass over the utterance
        self._busy = False
        self._busy_until = 0.0
        self._speak_lock = asyncio.Lock()
        self._play_end = 0.0  # when the audio queued so far should have finished playing
        self._stopped: Optional[asyncio.Event] = None
        self._worker: Optional[asyncio.Task[None]] = None
        # MUSE_MIC_LOG=<seconds>: log the mic level and peak VAD score that often (no audio, no text).
        self._mic_log_s = float(os.environ.get("MUSE_MIC_LOG") or 0)
        self._mic_log_at = 0.0
        self._mic_peak = 0.0
        self._mic_frames = 0

    # ------------------------------------------------------------------ lifecycle
    def _is_connected(self) -> bool:
        return self.connection is not None

    def _idle_behavior_ready(self) -> bool:
        return False  # no idle tool calls: they need the tool manager and a model

    def output_sample_rate(self) -> int:
        if self._output_rate is None:
            rate = -1
            try:
                rate = int(self.deps.reachy_mini.media.get_output_audio_samplerate())
            except Exception:
                pass
            # The app pushes our frames to the speaker as they are (no resampling), so render
            # at the speaker's rate: 16 kHz on Reachy Mini.
            self._output_rate = rate if rate > 0 else 16000
        return self._output_rate

    async def start_up(self) -> None:
        """Load STT and TTS, then stay 'connected' until shutdown (the app treats a return as a drop)."""
        self._stopped = asyncio.Event()
        loads = []
        if self.transcriber is None:
            from muse_stt import make_transcriber

            loads.append(asyncio.get_running_loop().run_in_executor(self._stt_thread, make_transcriber))
        if self.tts is None:
            from muse_tts import make_tts

            loads.append(asyncio.to_thread(make_tts))
        started = time.perf_counter()
        results = await asyncio.gather(*loads, return_exceptions=True)
        for result in results:  # own whatever loaded, even if the other one failed
            if isinstance(result, BaseException):
                continue
            if isinstance(result, tuple):
                self.stt_engine, self.transcriber = result
                self._owns_stt = True
            else:
                self.tts, self._owns_tts = result, True
        failed = next((r for r in results if isinstance(r, BaseException)), None)
        if failed is not None:  # don't leave a worker process behind for the app's retry to pile up
            await self._close_loaded()
            raise failed
        assert self.tts is not None
        if self._startup_voice and self._startup_voice in self.tts.voices():
            self.tts.voice = self._startup_voice
        logger.info("MuseHandler: speech-to-text %s, text-to-speech %s (voice %s) loaded in %.1f s",
                    self.stt_engine, self.tts.name, self.tts.voice or DEFAULT_VOICE, time.perf_counter() - started)
        try:
            health = await asyncio.to_thread(self.bridge.health)
            logger.info("MuseHandler: bridge %s health %s", self.bridge.base_url, health)
        except Exception as e:
            logger.warning("MuseHandler: bridge %s not answering yet (%s)", self.bridge.base_url, e)
        self.segmenter.reset()
        self.connection = "muse-bridge"
        self._worker = asyncio.create_task(self._turn_worker(), name="muse-turns")
        self._mark_activity("muse_connected")
        try:
            await self._start_robot_tools()
            await self._stopped.wait()
        finally:
            self.connection = None
            await self._stop_robot_tools()
            if self._worker is not None:
                self._worker.cancel()
                try:
                    await self._worker
                except (asyncio.CancelledError, Exception):
                    pass
                self._worker = None

    async def shutdown(self) -> None:
        self.connection = None
        if self._stopped is not None:
            self._stopped.set()
        if self._worker is not None:
            self._worker.cancel()
        await self._stop_robot_tools()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._close_loaded()

    async def _close_loaded(self) -> None:
        """Close the STT and TTS that start_up loaded (their worker processes), if any."""
        if self._owns_stt:  # the Qwen3-ASR worker; not while a clip is being transcribed
            stt, self._owns_stt = self.transcriber, False
            close_stt = getattr(stt, "close", None)
            if close_stt is not None:
                await asyncio.get_running_loop().run_in_executor(self._stt_thread, close_stt)
        if self._owns_tts and self.tts is not None:
            tts, self.tts, self._owns_tts = self.tts, None, False
            async with self._speak_lock:  # not while a sentence is being rendered
                await asyncio.to_thread(tts.close)

    # ------------------------------------------------------------------ mic in
    def _listening(self) -> bool:
        if self.connection is None:
            return False
        if self.barge_in:
            return True  # always: the user's voice takes the floor in any phase
        return self._armed or (not self._busy and time.monotonic() >= self._busy_until)

    def _robot_speaking(self) -> bool:
        return self._turn_active or time.monotonic() < self._play_end

    async def receive(self, frame: AudioFrame) -> None:
        if not self._listening():
            if self.segmenter.in_speech:
                self.segmenter.reset()
            self._early = None
            return
        rate, audio = frame
        if audio.size == 0:
            return
        mono = to_mono_16k(rate, audio)
        if self._mic_log_s > 0:
            self._log_mic(mono)
        was_in_speech = self.segmenter.in_speech
        utterances = self.segmenter.feed(mono)
        onset = not was_in_speech and bool(self.segmenter.in_speech or utterances)
        if self._armed:
            speaking = self._robot_speaking()
            if speaking and self._barge_at is None:
                self._echo_open = True
                self._echo_peak = max(self._echo_peak, self.segmenter.feed_peak)
            elif not speaking and self._echo_open:
                self._log_echo()
        if onset and self.barge_in:
            self._user_took_floor()
        for utterance in utterances:
            early = self._early if self._early and self._early[1] == self.segmenter.last_speech_windows else None
            self._early = None
            self._busy = True  # half-duplex: closed until this turn's reply has been spoken
            self._armed = False
            if self._echo_open:
                self._log_echo()
            self.segmenter.reset()
            # The speech ended one silence window ago (the segmenter waited that long).
            ended = time.perf_counter() - self.segmenter.silence_windows * WINDOW / SAMPLE_RATE
            self._utterances.put_nowait(_Turn(utterance, ended, early[0] if early else None))
            self._mark_activity("user_utterance")
            break
        else:
            if self.segmenter.in_speech:
                self._transcribe_early()
            else:
                self._early = None  # no utterance in progress (or one too short to be a turn)

    # ------------------------------------------------------------------ floor control
    def _user_took_floor(self) -> None:
        """The user started talking: stop Muse's moves, and the reply if it's playing.

        A reply that hasn't started is kept (Muse can't take a message back): _speak holds it
        until the user has finished."""
        self._stop_moves()
        if self._armed and self._robot_speaking() and self._barge_at is None and not self._cut:
            self._barge_in()

    def _muse_move_running(self) -> bool:
        """Whether a move Muse started (dance, emotion, look) is running or queued."""
        if self._muse_moved_at is None:
            return False
        if time.monotonic() - self._muse_moved_at < MOVE_GRACE_S:
            return True  # just queued: the movement manager may not have picked it up yet
        mm = getattr(self.deps, "movement_manager", None)
        state, queue = getattr(mm, "state", None), getattr(mm, "move_queue", None)
        if state is None and queue is None:
            return True  # can't tell: assume it runs (stopping an idle robot is harmless)
        try:
            moves = [getattr(state, "current_move", None), *list(queue or ())]
        except RuntimeError:  # changed while we looked
            return True
        running = any(m is not None and type(m).__name__ != "BreathingMove" for m in moves)
        if not running:
            self._muse_moved_at = None
        return running

    def _stop_moves(self) -> None:
        if not self._muse_move_running():
            return
        mm = getattr(self.deps, "movement_manager", None)
        moved_at, self._muse_moved_at = self._muse_moved_at, None
        try:
            mm.clear_move_queue()  # Pollen's stop_dance / stop_emotion: stop the move, the head holds
        except Exception as e:
            logger.warning("MuseHandler: move stop failed (%s)", type(e).__name__)
            return
        logger.info("MuseHandler: move stopped by user speech, %.1f s after Muse started it",
                    time.monotonic() - (moved_at or time.monotonic()))

    def _floor_is_users(self) -> bool:
        """Muse's moves from now on are refused: the user is talking, or this reply was cut."""
        return self.barge_in and (self.segmenter.in_speech or self._cut)

    def _transcribe_early(self) -> None:
        """At a pause, transcribe the utterance so far, so the text is ready if the pause is the end."""
        seg = self.segmenter
        if not self.stt_at_pauses or self.transcriber is None or not seg.in_speech:
            return
        if seg.silence_run < EARLY_AFTER_WINDOWS:
            return
        if seg.speech_windows < seg.min_speech_windows:
            return
        if self._early is not None and (self._early[1] == seg.speech_windows or not self._early[0].done()):
            return  # this pause has its pass already, or a pass for an earlier pause is still running
        future = asyncio.get_running_loop().run_in_executor(self._stt_thread, self.transcriber, seg.snapshot())
        future.add_done_callback(lambda f: f.cancelled() or f.exception())  # an unused pass may fail quietly
        self._early = (future, seg.speech_windows)

    def _barge_in(self) -> None:
        """The user started talking over the reply: stop it after barge_in_stop_s."""
        self._barge_at = time.monotonic()
        heard_ms = 0.0
        if self._segments:
            heard_ms = (self._barge_at - self._segments[0][0]) * 1000
        logger.info("MuseHandler: barge-in after %.0f ms of reply audio (peak VAD while speaking %.2f)",
                    heard_ms, self._echo_peak)
        self._echo_open = False
        asyncio.get_running_loop().call_later(self.barge_in_stop_s, self._cut_reply)

    def _cut_reply(self) -> None:
        """Fade out what's playing, drop the rest of the reply and the bridge stream."""
        now = time.monotonic()
        self._cut = True
        drop = getattr(self.bridge, "drop_stream", None)
        if drop is not None:
            drop()
        fade = self._fade_tail(now)
        clear = getattr(self, "_clear_queue", None)
        if callable(clear):
            clear()  # the app flushes the player and our output queue
        else:
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
        if fade is not None:
            self.output_queue.put_nowait(fade)
        self._play_end = now + (FADE_S if fade is not None else 0.0)
        self._busy_until = self._play_end
        self._segments.clear()
        logger.info("MuseHandler: barge-in: reply stopped %.0f ms after the barge-in (%s)",
                    (now - (self._barge_at or now)) * 1000, "faded" if fade is not None else "already over")

    def _fade_tail(self, now: float) -> Optional[tuple[int, np.ndarray]]:
        """The next FADE_S of what should be playing now, faded to silence (None if nothing plays)."""
        for start, pcm, rate in self._segments:
            pos = int((now - start) * rate)
            if 0 <= pos < len(pcm):
                tail = pcm[pos : pos + max(1, int(FADE_S * rate))].astype(np.float32)
                tail *= np.linspace(1.0, 0.0, len(tail), dtype=np.float32)
                return rate, tail.astype(np.int16).reshape(1, -1)
        return None

    def _log_echo(self) -> None:
        """After a reply played with the mic open: how close the robot's own voice came to a barge-in."""
        self._echo_open = False
        logger.info("MuseHandler: reply played with the mic open, no barge-in; peak VAD %.2f (threshold %.2f)",
                    self._echo_peak, self.segmenter.threshold)

    def _log_mic(self, mono: np.ndarray) -> None:
        self._mic_frames += 1
        if mono.size:
            self._mic_peak = max(self._mic_peak, float(np.max(np.abs(mono))))
        now = time.monotonic()
        if now - self._mic_log_at >= self._mic_log_s:
            logger.info("MuseHandler mic: %d frames, peak %.3f, peak VAD %.2f",
                        self._mic_frames, self._mic_peak, self.segmenter.peak_prob)
            self._mic_log_at, self._mic_peak, self._mic_frames = now, 0.0, 0
            self.segmenter.peak_prob = 0.0

    # ------------------------------------------------------------------ one turn
    async def _turn_worker(self) -> None:
        while True:
            turn = await self._utterances.get()
            try:
                await self._run_turn(turn)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("MuseHandler: turn failed")
            finally:
                if self._utterances.empty():  # a barge-in's utterance keeps the mic closed for its turn
                    self._busy = False

    async def _run_turn(self, turn: _Turn | np.ndarray, speech_ended: Optional[float] = None,
                        early: Optional[asyncio.Future] = None) -> None:
        if not isinstance(turn, _Turn):
            turn = _Turn(turn, time.perf_counter() if speech_ended is None else speech_ended, early)
        self._turn_active, self._cut, self._barge_at = True, False, None
        self._echo_peak, self._echo_open = 0.0, False
        self._segments = []
        try:
            await self._turn(turn)
        finally:
            self._turn_active = False

    async def _turn(self, turn: _Turn) -> None:
        assert self.transcriber is not None
        utterance, early = turn.utterance, turn.early
        started = time.perf_counter()
        speech_ended = turn.ended
        loop = asyncio.get_running_loop()
        text = None
        if early is not None:
            try:
                text = (await early).strip()
            except Exception:
                text = None
        early_used = text is not None
        if text is None:
            text = (await loop.run_in_executor(self._stt_thread, self.transcriber, utterance)).strip()
        stt_ms = (time.perf_counter() - started) * 1000
        if not text:
            logger.info("MuseHandler: empty transcript (%.1f s of audio), ignored", len(utterance) / 16000)
            return
        logger.info("MuseHandler: heard %d chars in %.0f ms", len(text), stt_ms)
        logger.info("MuseHandler: end of speech to text ready %.0f ms (%s)",
                    (time.perf_counter() - speech_ended) * 1000,
                    "transcribed at the pause" if early_used else "transcribed after the end silence")
        await self._transcript("user", text)
        self._turn_tools.clear()
        self._turn_tool_s = 0.0
        # Each sentence is spoken as soon as the bridge streams it, while Muse is still writing.
        sentences: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def on_sentence(sentence: str) -> None:   # on the bridge thread
            loop.call_soon_threadsafe(sentences.put_nowait, sentence)

        def fetch() -> str:
            try:
                if hasattr(self.bridge, "turn_stream"):
                    return self.bridge.turn_stream(text, on_sentence)
                reply = self.bridge.turn(text)
                on_sentence(reply)
                return reply
            finally:
                loop.call_soon_threadsafe(sentences.put_nowait, None)

        fetching = asyncio.ensure_future(asyncio.to_thread(fetch))
        first_audio: Optional[float] = None
        dropped = 0
        while (sentence := await sentences.get()) is not None:
            if self._cut:
                dropped += 1
                continue
            heard_at = await self._speak(sentence)
            if heard_at is not None and first_audio is None and early_used:
                asyncio.ensure_future(self._check_early(utterance, text))  # off the critical path now
            first_audio = first_audio or heard_at
        try:
            reply = await fetching
        except BridgeError as e:
            reply = spoken_error(e)
            if self._cut:
                reply = ""
            else:
                logger.warning("MuseHandler: bridge error %s", e.code)
                heard_at = await self._speak(reply)
                first_audio = first_audio or heard_at
        if self._cut:
            logger.info("MuseHandler: barge-in: %d later sentence(s) dropped", dropped)
        self._log_turn_tools()
        logger.info("MuseHandler: reply %d chars after %.0f ms", len(reply), (time.perf_counter() - started) * 1000)
        if first_audio is not None:
            logger.info("MuseHandler: end of speech to first audio %.0f ms (speech-to-text %.0f ms)",
                        (first_audio - speech_ended) * 1000, stt_ms)

    async def _check_early(self, utterance: np.ndarray, early_text: str) -> None:
        """Transcribe the full utterance too (after the reply started) and log if the texts match."""
        await asyncio.sleep(0)
        assert self.transcriber is not None
        started = time.perf_counter()
        try:
            full = (await asyncio.get_running_loop().run_in_executor(
                self._stt_thread, self.transcriber, utterance)).strip()
        except Exception:
            return
        logger.info("MuseHandler: text from the pause matches the full pass: %s (full pass %.0f ms)",
                    "yes" if full == early_text else "no", (time.perf_counter() - started) * 1000)

    async def _speak(self, text: str) -> Optional[float]:
        """Speak ``text``; returns when its first audio was queued (perf_counter), or None.

        After a barge-in the rest is rendered but not played (Qwen3's worker must finish a reply)."""
        text = text.strip()
        if not text or self._dropping():
            return None
        first_at: Optional[float] = None
        assert self.tts is not None
        async with self._speak_lock:
            rate = self.output_sample_rate()
            started = time.perf_counter()
            chunks = self.tts.chunks(text, rate)
            first = True
            while True:
                # One chunk (sentence) at a time in a thread; frames of the earlier ones are already playing.
                pcm = await asyncio.to_thread(next, chunks, None)
                if pcm is None:
                    break
                pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
                if self._dropping():
                    continue  # barge-in: render out, don't play
                if first and self.barge_in and self.segmenter.in_speech:
                    await self._hold_while_user_talks()
                    if self._dropping():
                        continue
                if first:
                    first_at = time.perf_counter()
                    if self.barge_in:
                        self._armed = True  # the mic opens: speech from now on is a barge-in
                    logger.info("MuseHandler: first audio after %.0f ms (%s)",
                                (time.perf_counter() - started) * 1000, self.tts.name)
                    await self._transcript("assistant", text)
                    first = False
                step = max(1, int(rate * FRAME_S))
                for i in range(0, len(pcm), step):
                    await self.output_queue.put((rate, pcm[i : i + step].reshape(1, -1)))
                now = time.monotonic()
                start = max(self._play_end, now)
                self._play_end = start + len(pcm) / rate
                self._busy_until = self._play_end + TAIL_S
                self._segments = [seg for seg in self._segments if seg[0] + len(seg[1]) / seg[2] > now]
                self._segments.append((start, pcm, rate))
                self._mark_activity("assistant_audio")
        return first_at

    def _dropping(self) -> bool:
        return self._cut

    async def _hold_while_user_talks(self) -> None:
        """The reply is ready but the user is talking: start it once they've finished."""
        held = time.monotonic()
        while self.segmenter.in_speech and self.connection is not None:
            await asyncio.sleep(0.02)
        logger.info("MuseHandler: reply held %.0f ms until you finished", (time.monotonic() - held) * 1000)

    async def _transcript(self, role: str, text: str) -> None:
        self._emit_transcript(role, text, True)
        if self.log_transcripts:
            # One line, so run_poc.sh's line-by-line redaction covers all of it.
            flat = " ".join(text.split())
            await self.output_queue.put(AdditionalOutputs({"role": role, "content": flat}))

    async def say(self, text: str) -> None:
        """Speak `text` verbatim with the current TTS (no Muse turn)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if not self._is_connected():
            raise RuntimeError("say: no active session")
        await self._speak(text)

    # ------------------------------------------------------------------ robot tools (Muse's reachy.* commands)
    async def _start_robot_tools(self) -> None:
        if not self._robot_tools_secret or self._robot_tools_server is not None:
            if not self._robot_tools_secret:
                logger.info("MuseHandler: robot tools off (no run secret)")
            return
        self.tool_manager.start_up(tool_callbacks=[self._on_tool_done])
        server = robot_tools.RobotToolsServer(self._robot_tools_secret, self._dispatch_robot_tool, self._is_connected)
        try:
            self._robot_tools_server = await server.serve(robot_tools.DEFAULT_HOST, self._robot_tools_port)
        except OSError as e:
            logger.error("MuseHandler: robot tools endpoint failed to start (%s); Muse can't move the robot", e)
            await self.tool_manager.shutdown()

    async def _stop_robot_tools(self) -> None:
        server, self._robot_tools_server = self._robot_tools_server, None
        if server is None:
            return
        server.close()
        try:
            await server.wait_closed()
        except Exception:
            pass
        await self.tool_manager.shutdown()
        for fut in self._tool_waiters.values():
            if not fut.done():
                fut.cancel()
        self._tool_waiters.clear()
        logger.info("MuseHandler: robot tools endpoint closed")

    def _log_turn_tools(self) -> None:
        """One line per turn: how many robot tool calls Muse made and their total time (names only)."""
        calls = sum(self._turn_tools.values())
        names = ", ".join(f"{name} x{n}" for name, n in self._turn_tools.most_common())
        logger.info("MuseHandler: %d robot tool call(s) this turn%s, tool time %.0f ms",
                    calls, f" ({names})" if names else "", self._turn_tool_s * 1000)

    async def _dispatch_robot_tool(self, tool: str, args: dict) -> dict:
        """Run one of Pollen's tools through the tool manager, as the HF backend does, and wait for it."""
        started = time.perf_counter()
        try:
            if tool in MOVE_TOOLS and self._floor_is_users():
                logger.info("MuseHandler: robot tool %s refused: the user has the floor", tool)
                return {"error": "the user is speaking; move not started"}
            result = await self._run_robot_tool(tool, args)
            if tool in MOVE_TOOLS and not result.get("error"):
                self._muse_moved_at = time.monotonic()
                if tool == "move_head":
                    result = {"direction": args.get("direction"), **result}
                result["status"] = IN_PROGRESS
            return result
        finally:
            self._turn_tools[tool] += 1
            self._turn_tool_s += time.perf_counter() - started

    async def _run_robot_tool(self, tool: str, args: dict) -> dict:
        call_id = f"muse-{uuid.uuid4().hex[:12]}"
        done: asyncio.Future = asyncio.get_running_loop().create_future()
        self._tool_waiters[call_id] = done
        try:
            await self.tool_manager.start_tool(
                call_id=call_id,
                tool_call_routine=ToolCallRoutine(tool_name=tool, args_json_str=json.dumps(args), deps=self.deps),
                is_idle_tool_call=False,
            )
            notification: ToolNotification = await done
        finally:
            self._tool_waiters.pop(call_id, None)
        if notification.status == ToolState.COMPLETED:
            return dict(notification.result or {})
        return {"error": notification.error or notification.status.value}

    async def _on_tool_done(self, notification: ToolNotification) -> None:
        done = self._tool_waiters.get(notification.id)
        if done is not None and not done.done():
            done.set_result(notification)

    # ------------------------------------------------------------------ app settings hooks
    async def apply_personality(self, profile: Optional[str]) -> str:
        return "Muse decides how it answers; app personalities don't apply to the Muse backend."

    async def get_available_voices(self) -> list[str]:
        if self.tts is not None and self.tts.voices():
            return self.tts.voices()
        voices = [DEFAULT_VOICE]
        if self.tts is not None and self.tts.voice:
            voices.append(self.tts.voice)
        return voices

    def get_current_voice(self) -> str:
        return (self.tts.voice if self.tts is not None else None) or DEFAULT_VOICE

    async def change_voice(self, voice: str) -> str:
        if self.tts is None:
            return "Speech isn't loaded yet."
        known = self.tts.voices()
        if known and voice not in known:
            return f"Unknown voice {voice}; voices: {', '.join(known)}."
        self.tts.voice = None if voice in ("", DEFAULT_VOICE) else voice
        return f"Voice set to {self.get_current_voice()}."
