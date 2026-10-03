"""MuseHandler: a ConversationHandler for Pollen's conversation app that talks to Muse.

One turn, half-duplex (the POC has no barge-in):

    robot mic -> VAD (one utterance) -> local STT on the Mac -> POST /turn on the Muse bridge
    (127.0.0.1) -> reply text -> TTS (Kokoro-82M sentence by sentence, or macOS `say`; see
    muse_tts.py) -> PCM frames on output_queue -> the app plays them on the robot's speaker and
    the daemon's wobbler moves the head. Playback starts as soon as the first sentence is ready.

While a turn is being transcribed, sent, or spoken, mic input is dropped (XVF3800 echo
cancellation helps, but the POC doesn't rely on it for turn-taking). Transcripts go to the app
only through `_emit_transcript` (its UI/JSON-RPC push, never logged). They are not put on the
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
from muse_vad import UtteranceSegmenter, make_vad, to_mono_16k
import robot_tools

logger = logging.getLogger(__name__)

Transcriber = Callable[[np.ndarray], str]
Synthesizer = Callable[[str, int], np.ndarray]  # (text, sample_rate) -> int16 mono PCM

FRAME_S = 0.2  # seconds of reply audio per output_queue item
TAIL_S = 0.6  # keep the mic closed this long after the reply should have finished playing
DEFAULT_VOICE = "system"


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
    """Speech in, Muse's reply spoken (Kokoro or macOS `say`) out."""

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
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.connection: Optional[str] = None  # LocalStream reads this to show "connected"
        self.bridge = bridge or BridgeClient()
        self.transcriber = transcriber
        self.stt_engine = "injected" if transcriber else None
        self._startup_voice = None if startup_voice in (None, "", DEFAULT_VOICE) else startup_voice
        self.tts: Optional[Tts] = tts or (CallableTts(synthesizer) if synthesizer else None)
        self._owns_tts = False  # loaded by start_up (so closed by shutdown)
        # MLX streams are per thread: parakeet must run on the thread that loaded it.
        self._stt_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="muse-stt")
        self.segmenter = segmenter or UtteranceSegmenter(make_vad())
        self._output_rate = output_sample_rate
        if log_transcripts is None:
            log_transcripts = os.environ.get("MUSE_LOG_TRANSCRIPTS") == "1"
        self.log_transcripts = log_transcripts  # opt-in: the app would log the text
        self._utterances: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
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
        for result in await asyncio.gather(*loads):
            if isinstance(result, tuple):
                self.stt_engine, self.transcriber = result
            else:
                self.tts, self._owns_tts = result, True
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
        if self._owns_tts and self.tts is not None:
            tts, self.tts, self._owns_tts = self.tts, None, False
            async with self._speak_lock:  # not while a sentence is being rendered
                await asyncio.to_thread(tts.close)

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
        loop = asyncio.get_running_loop()
        text = (await loop.run_in_executor(self._stt_thread, self.transcriber, utterance)).strip()
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
        assert self.tts is not None
        async with self._speak_lock:
            rate = self.output_sample_rate()
            started = time.perf_counter()
            chunks = self.tts.chunks(text, rate)
            first = True
            while True:
                # One sentence at a time in a thread; frames of the earlier ones are already playing.
                pcm = await asyncio.to_thread(next, chunks, None)
                if pcm is None:
                    break
                pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
                if first:
                    logger.info("MuseHandler: first audio after %.0f ms (%s)",
                                (time.perf_counter() - started) * 1000, self.tts.name)
                    await self._transcript("assistant", text)
                    first = False
                step = max(1, int(rate * FRAME_S))
                for i in range(0, len(pcm), step):
                    await self.output_queue.put((rate, pcm[i : i + step].reshape(1, -1)))
                now = time.monotonic()
                self._play_end = max(self._play_end, now) + len(pcm) / rate
                self._busy_until = self._play_end + TAIL_S
                self._mark_activity("assistant_audio")

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

    async def _dispatch_robot_tool(self, tool: str, args: dict) -> dict:
        """Run one of Pollen's tools through the tool manager, as the HF backend does, and wait for it."""
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
