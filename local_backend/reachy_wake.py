"""Wake-word gate: only send microphone audio to the speech server after "Hey Marvin" (or similar).

Enabled with REACHY_WAKE_WORD (start_conversation.sh --wake). run_app.py then calls install(), which
replaces LocalStream.record_loop with a version that runs every mic frame through this gate:

  * A local detector listens all the time, including while muted (not when hard-muted). Two engines:
      - any phrase (e.g. "hey reachy"): sherpa-onnx's open-vocabulary keyword spotter (a 3.3M-
        parameter streaming zipformer trained on GigaSpeech; the phrase is given as text, no training).
        Model in local_backend/models/kws/ (cache_models.sh). ~1.4 % of one CPU core.
      - a bundled openWakeWord model (hey_marvin, the launcher default; hey_jarvis, hey_mycroft, alexa) or a custom .onnx
        path: openWakeWord 0.4.0 (ONNX, offline; ~1.5 ms of CPU per 80 ms frame).
  * Closed gate: frames are only kept in a 1.5 s ring buffer and never leave the process.
  * On detection: un-mute if muted, send the ring buffer (so the words right after the wake word
    survive), and open a listening window.
  * The window stays open while the conversation is active (the user speaking, a response being
    generated, a reminder via conversation.say) and for FOLLOWUP_S after the robot has finished
    *speaking* a reply (the playback clock, not the end of generation: audio keeps playing a few
    seconds after the transcript is done, and the audio board suppresses the mic meanwhile),
    so follow-ups and barge-in need no new wake word. Book reading (out-of-band passages) does NOT keep
    it open, so the robot can't hear its own reading; "Hey Jarvis, stop" still interrupts it.

openWakeWord near misses (scores between NEAR_MISS and the threshold) are logged, to tell "not heard"
from "heard but under the threshold" when tuning (the keyword spotter reports no scores).

Settings: REACHY_WAKE_WORD (a phrase such as "hey reachy", or hey_jarvis | hey_mycroft | hey_marvin |
alexa | a path to a .onnx model), REACHY_WAKE_THRESHOLD (keyword spotter: trigger probability, 0.15;
openWakeWord: score, 0.4 - a clear "Hey Jarvis" scored 0.48 live), REACHY_WAKE_WINDOW_S (8),
REACHY_WAKE_FOLLOWUP_S (10).

Keyword-spotter settings were chosen on synthetic clips (Piper, 7 "Hey Reachy ..." sentences x 3 speeds,
14 near-miss sentences, 4.6 min of other speech incl. the robot reading and saying "reaching"):
full-precision model, 8 active paths, boost 2.0, threshold 0.15 -> 18/21 detected clean, 17/21 with
room noise at 10 dB SNR, 0 false alarms in the 4.6 min; "Hey Richie" also triggers. The int8 model
with 4 paths missed a third of the "Hey Reachy, <request>" clips.
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
ARBITRATE_FRAMES = 2               # with several wake words: after one fires, compare scores for 2 more frames
HOLD_MAX_S = 10.0                  # audio held during an assistant switch (reachy_assistants)


KWS_MODEL = Path(__file__).parent / "models" / "kws" / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KWS_SCORE = 2.0                    # per-token boost of the phrase in the beam search
KWS_PATHS = 8


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


def uses_openwakeword(word: str) -> bool:
    """A .onnx path or a bundled openWakeWord name (hey_jarvis, ...); anything else is a phrase."""
    return os.path.sep in word or word.endswith(".onnx") or word in ("hey_jarvis", "hey_mycroft", "hey_marvin", "alexa")


def make_detector(word: str, threshold: float | None = None) -> "Detector | PhraseDetector":
    if uses_openwakeword(word):
        return Detector(word, 0.4 if threshold is None else threshold)
    return PhraseDetector(word, 0.15 if threshold is None else threshold)


class Detector:
    """openWakeWord, one or several words (`words` = {name: threshold}); `last_word` says which fired.

    With several words, similar ones can both score (hey_marvin reached 0.4 on a real "Hey Jarvis, stop"),
    so after the first one crosses its threshold the scores are compared over ARBITRATE_FRAMES more frames
    (160 ms) and the highest word that crossed its own threshold wins.
    """

    def __init__(self, word: str = "", threshold: float = 0.5, words: dict[str, float] | None = None) -> None:
        from openwakeword.model import Model
        self.thresholds = dict(words) if words else {word: threshold}
        self.word = ",".join(self.thresholds)
        self.threshold = min(self.thresholds.values())
        self.model = Model(wakeword_model_paths=[_model_path(w) for w in self.thresholds])
        # prediction keys are model file stems ("hey_jarvis_v0.1"); map them back to our names
        self._names = {k: next((w for w in self.thresholds if Path(_model_path(w)).stem == k or k.startswith(w)), k)
                       for k in self.model.models}
        self.last_word = next(iter(self.thresholds))
        self._pending: dict[str, float] | None = None
        self._pending_frames = 0
        self._buf = np.zeros(0, dtype=np.int16)
        self._last = 0.0
        self._last_near = 0.0

    def feed(self, mono16k_int16: np.ndarray) -> float | None:
        """Add samples; returns the score if a wake word was just detected (see last_word), else None."""
        self._buf = np.concatenate([self._buf, mono16k_int16])
        hit = None
        while len(self._buf) >= FRAME:
            chunk, self._buf = self._buf[:FRAME], self._buf[FRAME:]
            scores = {self._names.get(k, k): float(v) for k, v in self.model.predict(chunk).items()}
            now = time.monotonic()
            if self._pending is not None:
                for w, sc in scores.items():
                    self._pending[w] = max(self._pending.get(w, 0.0), sc)
                self._pending_frames += 1
                if self._pending_frames >= ARBITRATE_FRAMES:
                    hit = self._decide(now)
                continue
            fired = [w for w, sc in scores.items() if sc >= self.thresholds.get(w, self.threshold)]
            if fired and now - self._last > COOLDOWN_S:
                self._pending, self._pending_frames = dict(scores), 0
                if len(self.thresholds) == 1:
                    hit = self._decide(now)
            elif not fired and now - self._last_near > 1.0:
                w, sc = max(scores.items(), key=lambda kv: kv[1])
                if sc >= NEAR_MISS:
                    self._last_near = now
                    logger.info("Wake word near miss: %s score %.2f (threshold %.2f)", w, sc,
                                self.thresholds.get(w, self.threshold))
        return hit

    def _decide(self, now: float) -> float:
        scores, self._pending = self._pending or {}, None
        over = {w: sc for w, sc in scores.items() if sc >= self.thresholds.get(w, self.threshold)}
        self.last_word = max(over or scores, key=(over or scores).get)
        if len(scores) > 1:
            logger.info("Wake word %s (scores %s)", self.last_word, {w: round(sc, 2) for w, sc in scores.items()})
        self._last = now
        return scores[self.last_word]


class PhraseDetector:
    """Open-vocabulary keyword spotter (sherpa-onnx): wakes on any phrase given as text, e.g. "hey reachy"."""

    def __init__(self, phrase: str, threshold: float = 0.15, model_dir: Path = KWS_MODEL) -> None:
        import tempfile
        import sentencepiece as spm
        import sherpa_onnx
        self.word = " ".join(phrase.replace("_", " ").split()).lower()
        self.threshold = threshold
        pieces = spm.SentencePieceProcessor(model_file=str(model_dir / "bpe.model")).encode(self.word.upper(), out_type=str)
        vocab = {line.split()[0] for line in (model_dir / "tokens.txt").read_text().splitlines() if line.strip()}
        if not pieces or any(p not in vocab for p in pieces):
            raise ValueError(f"Wake phrase {phrase!r} can't be spelled with the keyword model's tokens: {pieces}")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="reachy_kws_", delete=False) as f:
            f.write(f"{' '.join(pieces)} :{KWS_SCORE} #{threshold} @{self.word.replace(' ', '_')}\n")
        try:
            name = "epoch-12-avg-2-chunk-16-left-64.onnx"
            self.kws = sherpa_onnx.KeywordSpotter(
                tokens=str(model_dir / "tokens.txt"), encoder=str(model_dir / f"encoder-{name}"),
                decoder=str(model_dir / f"decoder-{name}"), joiner=str(model_dir / f"joiner-{name}"),
                keywords_file=f.name, num_threads=1, max_active_paths=KWS_PATHS,
                keywords_score=KWS_SCORE, keywords_threshold=threshold)
        finally:
            os.unlink(f.name)
        self.stream = self.kws.create_stream()
        self.last_word = self.word
        self._last = 0.0

    def feed(self, mono16k_int16: np.ndarray) -> float | None:
        """Add samples; returns 1.0 if the phrase was just detected (the spotter gives no score), else None."""
        self.stream.accept_waveform(RATE, np.asarray(mono16k_int16, dtype=np.float32) / 32768.0)
        hit = None
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            if self.kws.get_result(self.stream):
                self.kws.reset_stream(self.stream)
                now = time.monotonic()
                if now - self._last > COOLDOWN_S:
                    self._last, hit = now, 1.0
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
        self.last_word = getattr(detector, "last_word", getattr(detector, "word", ""))
        self.holding = False               # an assistant switch is in progress: keep frames, send them later
        self._held: list[np.ndarray] = []
        self._held_samples = 0

    # -- holding (reachy_assistants) -------------------------------------------------------------
    def hold(self) -> None:
        """Keep every forwarded frame until release() (the session is being rebuilt)."""
        with self._lock:
            self.holding = True

    def release(self) -> None:
        """Send the held frames with the next mic frame."""
        with self._lock:
            self.holding = False

    def _stash(self, frames: list[np.ndarray], rate: int) -> None:
        with self._lock:
            self._held.extend(frames)
            self._held_samples += sum(len(f) for f in frames)
            while self._held and self._held_samples > HOLD_MAX_S * rate:
                self._held_samples -= len(self._held.pop(0))

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
            self.last_word = getattr(self.detector, "last_word", getattr(self.detector, "word", ""))
            logger.info("Wake word %r detected", self.last_word)
            with self._lock:                    # take the pre-roll *before* opening the window (which clears it)
                preroll = list(self._ring)
                self._ring.clear()
                self._ring_samples = 0
            if muted:
                import reachy_listening
                reachy_listening.resume(reason="wake word")
                muted = False
            reachy_bridge.publish("wake_word")   # opens the window via on_activity; may start a switch (hold)
            self.extend(self.window_s)
            if self.holding:
                self._stash(preroll + [frame], rate)
                return []
            return preroll + [frame]
        if muted:
            return []
        if self.holding:
            self._stash([frame], rate)
            return []
        if self._held:
            with self._lock:
                out, self._held, self._held_samples = self._held + [frame], [], 0
            return out
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
    try:
        import reachy_assistants
        multi = reachy_assistants.wake_words() if reachy_assistants.enabled() else None
    except Exception:
        multi = None
    if not word and not multi:
        return
    threshold = os.environ.get("REACHY_WAKE_THRESHOLD")
    detector = Detector(words=multi) if multi else make_detector(word, float(threshold) if threshold else None)
    word = detector.word
    GATE = Gate(detector,
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
