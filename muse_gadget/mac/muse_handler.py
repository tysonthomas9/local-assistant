"""MuseHandler: a ConversationHandler for Pollen's conversation app that talks to Muse.

One turn, half-duplex (the POC has no barge-in):

    robot mic -> VAD (one utterance) -> local STT on the Mac -> POST /turn on the Muse bridge
    (127.0.0.1) -> reply text -> macOS `say` -> PCM frames on output_queue -> the app plays them
    on the robot's speaker and the daemon's wobbler moves the head.

While a turn is being transcribed, sent, or spoken, mic input is dropped (XVF3800 echo
cancellation helps, but the POC doesn't rely on it for turn-taking). Transcripts go to the app
only through `_emit_transcript` (its UI/JSON-RPC push, never logged). They are not put on the
output queue as AdditionalOutputs unless transcript logging is turned on (`log_transcripts=True`,
or MUSE_LOG_TRANSCRIPTS=1 from `run_app.py --log-transcripts`), because the app logs those at INFO
as `role=... content=<text>`. MuseHandler's own log lines hold only lengths and timings.

The STT engine, bridge client and TTS are injectable so the unit tests run on the PC.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Callable, Optional

import numpy as np

from reachy_mini_conversation_app.conversation_handler import AudioFrame, ConversationHandler
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager

from muse_bridge import BridgeClient, BridgeError, spoken_error
from muse_vad import UtteranceSegmenter, make_vad, to_mono_16k

logger = logging.getLogger(__name__)

Transcriber = Callable[[np.ndarray], str]
Synthesizer = Callable[[str, int], np.ndarray]  # (text, sample_rate) -> int16 mono PCM

FRAME_S = 0.2  # seconds of reply audio per output_queue item
TAIL_S = 0.6  # keep the mic closed this long after the reply should have finished playing
DEFAULT_VOICE = "system"


def _default_synthesizer(voice_getter: Callable[[], Optional[str]]) -> Synthesizer:
    def synth(text: str, rate: int) -> np.ndarray:
        from muse_tts import synthesize

        return synthesize(text, rate, voice=voice_getter())

    return synth


class MuseHandler(ConversationHandler):
    """Speech in, Muse's reply spoken with macOS `say` out."""

    def __init__(
        self,
        deps: Any,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
        *,
        bridge: Optional[BridgeClient] = None,
        transcriber: Optional[Transcriber] = None,
        synthesizer: Optional[Synthesizer] = None,
        segmenter: Optional[UtteranceSegmenter] = None,
        output_sample_rate: Optional[int] = None,
        log_transcripts: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.deps = deps
        self.instance_path = instance_path
        self.tool_manager = BackgroundToolManager()  # never started: Muse can't call the app's tools
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.connection: Optional[str] = None  # LocalStream reads this to show "connected"
        self.bridge = bridge or BridgeClient()
        self.transcriber = transcriber
        self.stt_engine = "injected" if transcriber else None
        self._voice: Optional[str] = None if startup_voice in (None, "", DEFAULT_VOICE) else startup_voice
        self.synthesizer = synthesizer or _default_synthesizer(lambda: self._voice)
        self.segmenter = segmenter or UtteranceSegmenter(make_vad())
        self._output_rate = output_sample_rate
        if log_transcripts is None:
            log_transcripts = os.environ.get("MUSE_LOG_TRANSCRIPTS") == "1"
        self.log_transcripts = log_transcripts  # opt-in: the app would log the text
        self._utterances: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
        self._busy = False
        self._busy_until = 0.0
        self._speak_lock = asyncio.Lock()
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
        """Load STT, then stay 'connected' until shutdown (the app treats a return as a drop)."""
        self._stopped = asyncio.Event()
        if self.transcriber is None:
            from muse_stt import make_transcriber

            self.stt_engine, self.transcriber = await asyncio.to_thread(make_transcriber)
            logger.info("MuseHandler: speech-to-text %s loaded", self.stt_engine)
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
            await self._stopped.wait()
        finally:
            self.connection = None
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
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------ mic in
    def _listening(self) -> bool:
        return self.connection is not None and not self._busy and time.monotonic() >= self._busy_until

    async def receive(self, frame: AudioFrame) -> None:
        if not self._listening():
            if self.segmenter.in_speech:
                self.segmenter.reset()
            return
        rate, audio = frame
        if audio.size == 0:
            return
        mono = to_mono_16k(rate, audio)
        if self._mic_log_s > 0:
            self._log_mic(mono)
        for utterance in self.segmenter.feed(mono):
            self._busy = True  # half-duplex: closed until this turn has been spoken
            self.segmenter.reset()
            try:
                self._utterances.put_nowait(utterance)
            except asyncio.QueueFull:
                pass
            self._mark_activity("user_utterance")
            break

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
            utterance = await self._utterances.get()
            try:
                await self._run_turn(utterance)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("MuseHandler: turn failed")
            finally:
                self._busy = False

    async def _run_turn(self, utterance: np.ndarray) -> None:
        assert self.transcriber is not None
        started = time.perf_counter()
        text = (await asyncio.to_thread(self.transcriber, utterance)).strip()
        stt_ms = (time.perf_counter() - started) * 1000
        if not text:
            logger.info("MuseHandler: empty transcript (%.1f s of audio), ignored", len(utterance) / 16000)
            return
        logger.info("MuseHandler: heard %d chars in %.0f ms", len(text), stt_ms)
        await self._transcript("user", text)
        try:
            reply = await asyncio.to_thread(self.bridge.turn, text)
        except BridgeError as e:
            logger.warning("MuseHandler: bridge error %s", e.code)
            reply = spoken_error(e)
        logger.info("MuseHandler: reply %d chars after %.0f ms", len(reply), (time.perf_counter() - started) * 1000)
        await self._speak(reply)

    async def _speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        async with self._speak_lock:
            rate = self.output_sample_rate()
            pcm = await asyncio.to_thread(self.synthesizer, text, rate)
            pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
            await self._transcript("assistant", text)
            step = max(1, int(rate * FRAME_S))
            for i in range(0, len(pcm), step):
                await self.output_queue.put((rate, pcm[i : i + step].reshape(1, -1)))
            self._busy_until = time.monotonic() + len(pcm) / rate + TAIL_S
            self._mark_activity("assistant_audio")

    async def _transcript(self, role: str, text: str) -> None:
        self._emit_transcript(role, text, True)
        if self.log_transcripts:
            # One line, so run_poc.sh's line-by-line redaction covers all of it.
            flat = " ".join(text.split())
            await self.output_queue.put(AdditionalOutputs({"role": role, "content": flat}))

    async def say(self, text: str) -> None:
        """Speak `text` verbatim with `say` (no Muse turn)."""
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if not self._is_connected():
            raise RuntimeError("say: no active session")
        await self._speak(text)

    # ------------------------------------------------------------------ app settings hooks
    async def apply_personality(self, profile: Optional[str]) -> str:
        return "Muse decides how it answers; app personalities don't apply to the Muse backend."

    async def get_available_voices(self) -> list[str]:
        voices = [DEFAULT_VOICE]
        if self._voice:
            voices.append(self._voice)
        return voices

    def get_current_voice(self) -> str:
        return self._voice or DEFAULT_VOICE

    async def change_voice(self, voice: str) -> str:
        self._voice = None if voice in ("", DEFAULT_VOICE) else voice
        return f"Voice set to {self.get_current_voice()}."
