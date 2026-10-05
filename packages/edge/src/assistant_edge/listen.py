"""Hands-free listening on the edge: a speech detector (Silero VAD) and wake-word engines.

The edge agent (`assistant_edge.agent`) has three listening modes (`[edge.listen] mode`):

- `wake_word` (the default): a wake-word engine hears every microphone frame; a detection
  sends `wake{word, score}` and opens a mic window with the audio of the last `pre_roll_s`
  first (so the words right after the wake word are not lost). After a reply the brain's
  `mic.follow_up` lets the user speak again without the wake word for a while.
- `open_mic`: no wake word; the speech detector opens a window when speech starts.
- `push_to_talk`: only `/ptt down` ... `/ptt up` (and the test aids) open a window.

In the first two the speech detector also ends a window (`vad_end`) once speech was heard and
then stopped. Engines:

- `SpeechDetector`: Silero VAD v6 (pysilero-vad: the model runs on the CPU through ggml, no
  other dependency) on 32 ms chunks, with a start run and an end silence, so claps, music
  and noise do not count as speech.
- `OpenWakeWordEngine`: openWakeWord 0.4.0 (ONNX on the CPU; its bundled models hey_jarvis,
  hey_marvin, hey_mycroft, alexa, or a path to a custom `.onnx`), one or several words. Once a
  score crosses its threshold the peak over 2 more steps (160 ms) is taken: the score reported,
  and with several words the highest over its own threshold wins (similar words can both
  score).
- `PhraseSpotter`: any phrase given as text (sherpa-onnx's open-vocabulary keyword spotter;
  the optional `phrase` extra, its model directory in `[edge.wake] phrase_model_dir`).

Ported from the legacy stack's `local_backend/reachy_wake.py` (Detector, PhraseDetector).
"""

import os
import tempfile
import warnings
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# onnxruntime (openWakeWord) reports usage to Microsoft from Linux unless this is set.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

RATE = 16000
VAD_CHUNK = 512
"""Samples per Silero chunk (32 ms at 16 kHz)."""
WAKE_CHUNK = 1280
"""Samples per openWakeWord step (80 ms)."""
COOLDOWN_S = 2.0
NEAR_MISS = 0.2
ARBITRATE_FRAMES = 2
OPENWAKEWORD_MODELS = ("hey_jarvis", "hey_marvin", "hey_mycroft", "alexa")


# ---------------------------------------------------------------- speech detector


class SpeechDetector:
    """Silero VAD over 20 ms frames, with hysteresis.

    `feed(pcm)` takes one frame (s16le mono 16 kHz) and returns the speech probability of the
    latest complete 32 ms chunk. `run_ms` counts the frames in a row at or above `threshold`
    (speech starts when it reaches the start run); `quiet_ms` the frames in a row below
    `end_threshold` (speech ended after the end silence).
    """

    def __init__(self, threshold: float = 0.5, end_threshold: float = 0.35) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self.threshold = threshold
        self.end_threshold = end_threshold
        self._vad = SileroVoiceActivityDetector()
        self._pending = bytearray()
        self.prob = 0.0
        self.run_ms = 0
        self.quiet_ms = 0

    def feed(self, pcm: bytes, frame_ms: int = 20) -> float:
        self._pending += pcm
        while len(self._pending) >= VAD_CHUNK * 2:
            chunk = bytes(self._pending[: VAD_CHUNK * 2])
            del self._pending[: VAD_CHUNK * 2]
            self.prob = float(self._vad(chunk))
        if self.prob >= self.threshold:
            self.run_ms += frame_ms
        else:
            self.run_ms = 0
        if self.prob < self.end_threshold:
            self.quiet_ms += frame_ms
        else:
            self.quiet_ms = 0
        return self.prob


# ---------------------------------------------------------------- wake words


@dataclass(frozen=True)
class WakeHit:
    model: str
    """The detector's name for the word (an openWakeWord model name, or the phrase)."""
    score: float


class WakeEngine(Protocol):
    name: str

    def feed(self, pcm: bytes) -> WakeHit | None:
        """Add one frame; a detection, if one completed with it."""
        ...

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        """Update per-word thresholds (from the brain's `welcome`)."""
        ...

    def near_misses(self) -> list[tuple[str, float, float]]:
        """(word, score, threshold) heard since the last call but under the threshold."""
        ...


def openwakeword_model_path(word: str) -> str:
    """A bundled openWakeWord model by name (hey_jarvis, ...) or a `.onnx` path."""
    if os.path.sep in word or word.endswith(".onnx"):
        return word
    import openwakeword

    models = Path(openwakeword.__file__).parent / "resources" / "models"
    hits = sorted(models.glob(f"{word}_v*.onnx"))
    if not hits:
        available = sorted(p.name.rsplit("_v", 1)[0] for p in models.glob("*_v*.onnx"))
        raise ValueError(f"no bundled wake-word model {word!r}; available: {available}")
    return str(hits[-1])


def uses_openwakeword(word: str) -> bool:
    """A `.onnx` path or a bundled openWakeWord name; anything else is a phrase."""
    return os.path.sep in word or word.endswith(".onnx") or word in OPENWAKEWORD_MODELS


class OpenWakeWordEngine:
    """openWakeWord, one or several words (`words` = {model name: threshold})."""

    name = "openwakeword"

    def __init__(self, words: dict[str, float]) -> None:
        import onnxruntime
        from openwakeword.model import Model

        # openWakeWord asks for CUDA first (a warning without it) and onnxruntime logs its GPU
        # probe: neither is a problem on the edge's CPU.
        onnxruntime.set_default_logger_severity(3)
        warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

        if not words:
            raise ValueError("no wake words")
        self.thresholds = dict(words)
        paths = {w: openwakeword_model_path(w) for w in words}
        self.model: Any = Model(wakeword_model_paths=list(paths.values()))
        # Prediction keys are model file stems ("hey_jarvis_v0.1"): map them back.
        stems = {Path(p).stem: w for w, p in paths.items()}
        keys: list[str] = [str(k) for k in self.model.models]
        self._names: dict[str, str] = {k: stems.get(k, k) for k in keys}
        self._buf = np.zeros(0, dtype=np.int16)
        self._pending: dict[str, float] | None = None
        self._pending_frames = 0
        self._heard_s = 0.0
        """Audio fed so far, in seconds: the clock of the cooldowns (real time or not)."""
        self._last = -COOLDOWN_S
        self._last_near = -1.0
        self._near: list[tuple[str, float, float]] = []

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        for word, value in thresholds.items():
            if word in self.thresholds:
                self.thresholds[word] = value

    def near_misses(self) -> list[tuple[str, float, float]]:
        near, self._near = self._near, []
        return near

    def feed(self, pcm: bytes) -> WakeHit | None:
        self._buf = np.concatenate([self._buf, np.frombuffer(pcm, dtype="<i2")])
        hit: WakeHit | None = None
        while len(self._buf) >= WAKE_CHUNK:
            chunk, self._buf = self._buf[:WAKE_CHUNK], self._buf[WAKE_CHUNK:]
            raw: dict[str, Any] = self.model.predict(chunk)
            self._heard_s += WAKE_CHUNK / RATE
            scores: dict[str, float] = {
                self._names.get(str(k), str(k)): float(v) for k, v in raw.items()
            }
            now = self._heard_s
            if self._pending is not None:
                for word, score in scores.items():
                    self._pending[word] = max(self._pending.get(word, 0.0), score)
                self._pending_frames += 1
                if self._pending_frames >= ARBITRATE_FRAMES:
                    hit = self._decide(now)
                continue
            fired = [w for w, s in scores.items() if s >= self.thresholds.get(w, 1.0)]
            if fired and now - self._last > COOLDOWN_S:
                # Wait 2 more steps: the score peaks a little after it first crosses (and with
                # several words, a similar word may score too).
                self._pending, self._pending_frames = dict(scores), 0
            elif not fired and now - self._last_near > 1.0:
                word, score = max(scores.items(), key=lambda kv: kv[1])
                if score >= NEAR_MISS:
                    self._last_near = now
                    self._near.append((word, score, self.thresholds.get(word, 1.0)))
        return hit

    def _decide(self, now: float) -> WakeHit:
        scores, self._pending = self._pending or {}, None
        over = {w: s for w, s in scores.items() if s >= self.thresholds.get(w, 1.0)}
        pool = over or scores
        word = max(pool, key=lambda w: pool[w])
        self._last = now
        return WakeHit(model=word, score=scores[word])


class PhraseSpotter:
    """sherpa-onnx's open-vocabulary keyword spotter: wakes on a phrase given as text."""

    name = "phrase"
    KWS_SCORE = 2.0
    KWS_PATHS = 8
    MODEL = "epoch-12-avg-2-chunk-16-left-64.onnx"

    def __init__(self, phrase: str, threshold: float, model_dir: Path) -> None:
        try:
            import sentencepiece as spm  # pyright: ignore[reportMissingImports]
            import sherpa_onnx  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ValueError(
                "the phrase spotter needs the edge's `phrase` extra (sherpa-onnx, sentencepiece)"
            ) from exc
        if not (model_dir / "tokens.txt").exists():
            raise ValueError(
                f"no keyword-spotting model in {model_dir} ([edge.wake] phrase_model_dir)"
            )
        self.word = " ".join(phrase.replace("_", " ").split()).lower()
        self.threshold = threshold
        processor: Any = spm.SentencePieceProcessor(model_file=str(model_dir / "bpe.model"))
        pieces: list[str] = processor.encode(self.word.upper(), out_type=str)
        vocab = {
            line.split()[0]
            for line in (model_dir / "tokens.txt").read_text().splitlines()
            if line.strip()
        }
        if not pieces or any(p not in vocab for p in pieces):
            raise ValueError(
                f"wake phrase {phrase!r} cannot be spelled with the model's tokens: {pieces}"
            )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="kws_", delete=False) as f:
            label = self.word.replace(" ", "_")
            f.write(f"{' '.join(pieces)} :{self.KWS_SCORE} #{threshold} @{label}\n")
        try:
            self.kws: Any = sherpa_onnx.KeywordSpotter(
                tokens=str(model_dir / "tokens.txt"),
                encoder=str(model_dir / f"encoder-{self.MODEL}"),
                decoder=str(model_dir / f"decoder-{self.MODEL}"),
                joiner=str(model_dir / f"joiner-{self.MODEL}"),
                keywords_file=f.name,
                num_threads=1,
                max_active_paths=self.KWS_PATHS,
                keywords_score=self.KWS_SCORE,
                keywords_threshold=threshold,
            )
        finally:
            os.unlink(f.name)
        self.stream: Any = self.kws.create_stream()
        self._heard_s = 0.0
        self._last = -COOLDOWN_S

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        del thresholds  # fixed when the spotter is built

    def near_misses(self) -> list[tuple[str, float, float]]:
        return []  # the spotter reports no scores

    def feed(self, pcm: bytes) -> WakeHit | None:
        samples = np.asarray(array("h", pcm), dtype=np.float32) / 32768.0
        self.stream.accept_waveform(RATE, samples)
        self._heard_s += len(samples) / RATE
        hit = None
        while self.kws.is_ready(self.stream):
            self.kws.decode_stream(self.stream)
            if self.kws.get_result(self.stream):
                self.kws.reset_stream(self.stream)
                now = self._heard_s
                if now - self._last > COOLDOWN_S:
                    self._last, hit = now, WakeHit(model=self.word, score=1.0)
        return hit


def make_wake_engine(
    engine: str, words: list[str], threshold: float, phrase_model_dir: Path | None = None
) -> WakeEngine:
    """`engine` = `openwakeword` (bundled model names or .onnx paths) or `phrase` (one phrase)."""
    if engine == "openwakeword":
        return OpenWakeWordEngine(dict.fromkeys(words, threshold))
    if engine == "phrase":
        if len(words) != 1:
            raise ValueError(f"the phrase spotter takes one phrase, got {words}")
        if phrase_model_dir is None:
            raise ValueError("the phrase spotter needs [edge.wake] phrase_model_dir")
        return PhraseSpotter(words[0], threshold, phrase_model_dir)
    raise ValueError(f"unknown wake engine {engine!r}: openwakeword or phrase")
