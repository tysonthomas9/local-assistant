"""Wake-word gate: only send microphone audio to the speech server after "Hey Jarvis" (or similar).

Enabled with REACHY_WAKE_WORD (start_conversation.sh --wake). run_app.py then calls install(), which
replaces LocalStream.record_loop with a version that runs every mic frame through this gate:

  * A local openWakeWord detector (0.4.0, ONNX, models bundled in the package, offline; ~1.5 ms of CPU
    per 80 ms frame) listens all the time, including while muted (not when hard-muted).
  * Closed gate: frames are only kept in a 1.5 s ring buffer and never leave the process.
  * On detection: un-mute if muted, send the ring buffer (so the words right after the wake word
    survive), and open a listening window.
  * The window stays open while the conversation is active (the user speaking, a response being
    generated, a reminder via conversation.say) and for FOLLOWUP_S after the robot has finished
    *speaking* a reply (the playback clock, not the end of generation: audio keeps playing a few
    seconds after the transcript is done, and the audio board suppresses the mic meanwhile),
    so follow-ups and barge-in need no new wake word. Book reading (out-of-band passages) does NOT keep
    it open, so the robot can't hear its own reading; "Hey Jarvis, stop" still interrupts it.

Near misses (scores between NEAR_MISS and the threshold) are logged, to tell "not heard" from
"heard but under the threshold" when tuning.

Settings: REACHY_WAKE_WORD (hey_jarvis | hey_mycroft | hey_marvin | alexa, or a path to a custom
.onnx model), REACHY_WAKE_THRESHOLD (0.5), REACHY_WAKE_WINDOW_S (8), REACHY_WAKE_FOLLOWUP_S (10).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import reachy_bridge

logger = logging.getLogger("reachy_wake")

RATE = 16000
FRAME = 1280                       # openWakeWord's 80 ms step
PREROLL_S = 1.5
COOLDOWN_S = 2.0
NEAR_MISS = 0.2


def _model_path(word: str) -> str:
    if os.path.sep in word or word.endswith(".onnx"):
        return word
    import openwakeword
    models = Path(openwakeword.__file__).parent / "resources" / "models"
    hits = sorted(models.glob(f"{word}_v*.onnx"))
    if not hits:
        raise ValueError(f"No bundled wake-word model {word!r}; available: "
                         f"{sorted(p.name.rsplit('_v', 1)[0] for p in models.glob('*_v*.onnx'))}")
    return str(hits[-1])


class Detector:
    def __init__(self, word: str, threshold: float) -> None:
        from openwakeword.model import Model
        self.word, self.threshold = word, threshold
        self.model = Model(wakeword_model_paths=[_model_path(word)])
        self._buf = np.zeros(0, dtype=np.int16)
        self._last = 0.0
        self._last_near = 0.0

    def feed(self, mono16k_int16: np.ndarray) -> float | None:
        """Add samples; returns the score if the wake word was just detected, else None."""
        self._buf = np.concatenate([self._buf, mono16k_int16])
        hit = None
        while len(self._buf) >= FRAME:
            chunk, self._buf = self._buf[:FRAME], self._buf[FRAME:]
            score = max(float(v) for v in self.model.predict(chunk).values())
            now = time.monotonic()
            if score >= self.threshold and now - self._last > COOLDOWN_S:
                self._last, hit = now, score
            elif NEAR_MISS <= score < self.threshold and now - self._last_near > 1.0:
                self._last_near = now
                logger.info("Wake word near miss: score %.2f (threshold %.2f)", score, self.threshold)
        return hit


def to_mono_int16(frame: np.ndarray, rate: int) -> np.ndarray:
    a = np.asarray(frame, dtype=np.float32)
    if a.ndim == 2:
        a = a[:, 0] if a.shape[1] <= a.shape[0] else a[0]
    if rate != RATE:
        from scipy.signal import resample_poly
        a = resample_poly(a, RATE, rate)
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


class Gate:
    def __init__(self, detector: Detector | Any, window_s: float = 8.0, followup_s: float = 10.0) -> None:
        self.detector = detector
        self.window_s, self.followup_s = window_s, followup_s
        self.open_until = 0.0
        self._ring: deque[np.ndarray] = deque()
        self._ring_samples = 0
        self._lock = threading.Lock()
        self.detections = 0

    # -- window --------------------------------------------------------------------------------
    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def extend(self, seconds: float) -> None:
        with self._lock:
            now = time.monotonic()
            if now >= self.open_until:     # closed -> open: pre-window audio must not be sent later
                self._ring.clear()
                self._ring_samples = 0
            self.open_until = max(self.open_until, now + seconds)

    def on_activity(self, reason: str) -> None:
        reading = _reader_reading()
        if reason in ("user_speech_started", "user_speech_stopped", "user_transcription_delta",
                      "user_transcription_completed", "tool_call_received", "tool_result_ready", "say", "wake_word"):
            self.extend(self.window_s)
        elif reason in ("response_created", "assistant_audio_delta") and self.is_open() and not reading:
            self.extend(self.window_s)       # keep an open conversation open while the robot answers
        elif reason == "assistant_transcript_done" and self.is_open() and not reading:
            # count the follow-up from when the reply has finished *playing*, not generating
            self.extend(self.followup_s + reachy_bridge.audio_seconds_left())

    # -- frames --------------------------------------------------------------------------------
    def process(self, frame: np.ndarray, rate: int, muted: bool, hard_muted: bool) -> list[np.ndarray]:
        """Frames to forward to the speech server now (possibly the pre-roll + this one)."""
        if not hard_muted and self.detector.feed(to_mono_int16(frame, rate)) is not None:
            self.detections += 1
            logger.info("Wake word %r detected", getattr(self.detector, "word", "?"))
            with self._lock:                    # take the pre-roll *before* opening the window (which clears it)
                preroll = list(self._ring)
                self._ring.clear()
                self._ring_samples = 0
            if muted:
                import reachy_listening
                reachy_listening.resume(reason="wake word")
                muted = False
            reachy_bridge.publish("wake_word")   # opens the window via on_activity
            self.extend(self.window_s)
            return preroll + [frame]
        if muted:
            return []
        if self.is_open():
            return [frame]
        self._ring.append(frame)
        self._ring_samples += len(frame)
        while self._ring and self._ring_samples > PREROLL_S * rate:
            self._ring_samples -= len(self._ring.popleft())
        return []


def _reader_reading() -> bool:
    try:
        import reachy_reader
        return bool(reachy_reader.READER.reading)
    except Exception:
        return False


GATE: Gate | None = None


def install() -> None:
    """Create the detector/gate and replace LocalStream.record_loop (call before the app starts)."""
    global GATE
    word = os.environ.get("REACHY_WAKE_WORD", "").strip()
    if not word:
        return
    GATE = Gate(Detector(word, float(os.environ.get("REACHY_WAKE_THRESHOLD", 0.5))),
                window_s=float(os.environ.get("REACHY_WAKE_WINDOW_S", 8)),
                followup_s=float(os.environ.get("REACHY_WAKE_FOLLOWUP_S", 10)))
    reachy_bridge.subscribe(GATE.on_activity)

    from reachy_mini_conversation_app import console
    import reachy_listening

    async def record_loop(self: Any) -> None:
        """Upstream record_loop plus the wake-word gate (console.py LocalStream.record_loop)."""
        rate = self._robot.media.get_input_audio_samplerate()
        while not self._stop_event.is_set():
            frame = self._robot.media.get_audio_sample()
            if frame is not None:
                for f in GATE.process(frame, rate, self._mic_muted, reachy_listening.is_hard_muted()):
                    await self.handler.receive((rate, f))
                if not self._mic_muted and GATE.is_open():
                    self._emit_level("user", frame)
            await asyncio.sleep(0)

    console.LocalStream.record_loop = record_loop
    logger.info("Wake-word gate on: say %r to talk", word)
