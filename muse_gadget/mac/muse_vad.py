"""Voice activity detection: cut the robot's mic stream into one utterance at a time.

`UtteranceSegmenter` takes 16 kHz mono float32 audio in any chunk size, scores it in 32 ms
windows with a VAD, and returns the whole utterance (with a little audio before the speech
started) once the speaker has been quiet for `silence_s`.

Two VADs:
- `SileroVad`: Silero VAD v5 (the `silero_vad.onnx` file shipped in the `silero-vad` wheel),
  run with onnxruntime and numpy only. The wheel is installed with `--no-deps`, so torch is not
  needed. This is the one used on the robot.
- `EnergyVad`: a plain RMS threshold, used when Silero isn't available and in unit tests.
"""

from __future__ import annotations

import collections
import os
from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
WINDOW = 512  # samples per VAD window at 16 kHz (32 ms), what Silero v5 expects


class Vad(Protocol):
    def __call__(self, window: np.ndarray) -> float:
        """Return the speech probability (0..1) of one WINDOW-sample float32 window."""

    def reset(self) -> None: ...


class EnergyVad:
    """Speech when the window's RMS is above `threshold` (float32 audio in -1..1)."""

    def __init__(self, threshold: float = 0.02) -> None:
        self.threshold = threshold

    def __call__(self, window: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        return 1.0 if rms >= self.threshold else 0.0

    def reset(self) -> None:
        pass


def silero_model_path() -> Path | None:
    """The Silero ONNX model inside the installed `silero-vad` wheel, if any."""
    import importlib.util

    try:  # find_spec doesn't import the package (its __init__ needs torch)
        spec = importlib.util.find_spec("silero_vad")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / "data" / "silero_vad.onnx"
    return path if path.exists() else None


class SileroVad:
    """Silero VAD v5 through onnxruntime (a numpy port of silero_vad.utils_vad.OnnxWrapper)."""

    CONTEXT = 64  # samples of the previous window prepended to each window at 16 kHz

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self._context, window.reshape(1, WINDOW).astype(np.float32)], axis=1)
        out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -self.CONTEXT :]
        return float(out.reshape(-1)[0])


def make_vad() -> Vad:
    """Silero when its model is installed, else the energy VAD."""
    path = silero_model_path()
    if path is not None:
        try:
            return SileroVad(path)
        except Exception:
            pass
    return EnergyVad()


# Quiet time that ends an utterance (run_poc.sh --end-silence sets MUSE_END_SILENCE). Shorter is a
# faster turn; too short cuts the user off at a pause.
END_SILENCE_S = 0.5
END_SILENCE_ENV = "MUSE_END_SILENCE"


def end_silence_from_env() -> float:
    try:
        value = float(os.environ.get(END_SILENCE_ENV) or END_SILENCE_S)
    except ValueError:
        return END_SILENCE_S
    return value if 0.1 <= value <= 5.0 else END_SILENCE_S


class UtteranceSegmenter:
    """Collect one utterance: starts on `start_windows` speech windows, ends after `silence_s`."""

    def __init__(
        self,
        vad: Vad,
        *,
        threshold: float = 0.5,
        start_windows: int = 3,
        silence_s: float = END_SILENCE_S,
        preroll_s: float = 0.3,
        min_speech_s: float = 0.3,
        max_utterance_s: float = 15.0,
    ) -> None:
        self.vad = vad
        self.threshold = threshold
        self.start_windows = start_windows
        self.silence_windows = max(1, round(silence_s * SAMPLE_RATE / WINDOW))
        self.min_speech_windows = max(1, round(min_speech_s * SAMPLE_RATE / WINDOW))
        self.max_windows = max(1, round(max_utterance_s * SAMPLE_RATE / WINDOW))
        self.peak_prob = 0.0  # highest VAD score since the caller last zeroed it (diagnostics)
        self.last_speech_windows = 0  # speech windows of the last utterance that ended
        self.feed_peak = 0.0
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(start_windows, round(preroll_s * SAMPLE_RATE / WINDOW))
        )
        self.reset()

    @property
    def in_speech(self) -> bool:
        return self._speaking

    @property
    def silence_run(self) -> int:
        """Quiet windows since the last speech window of the utterance in progress (0 while talking)."""
        return self._silence if self._speaking else 0

    @property
    def speech_windows(self) -> int:
        """Speech windows in the utterance so far (only grows while it lasts)."""
        return self._speech_windows if self._speaking else 0

    def snapshot(self) -> np.ndarray:
        """The utterance in progress so far (pre-roll included), as one array."""
        return np.concatenate(self._windows) if self._speaking and self._windows else np.zeros(0, np.float32)

    def reset(self) -> None:
        """Drop any partial utterance and VAD state (used while the robot speaks)."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._run = 0
        self._speaking = False
        self._windows: list[np.ndarray] = []
        self._speech_windows = 0
        self._silence = 0
        self.vad.reset()

    def feed(self, audio: np.ndarray) -> list[np.ndarray]:
        """Add 16 kHz mono float32 audio; return the utterances it completed (usually none)."""
        done: list[np.ndarray] = []
        self.feed_peak = 0.0  # highest VAD score in this call (MuseHandler's echo check)
        self._pending = np.concatenate([self._pending, audio.astype(np.float32, copy=False)])
        n = len(self._pending) // WINDOW
        for i in range(n):
            window = self._pending[i * WINDOW : (i + 1) * WINDOW]
            utterance = self._step(window)
            if utterance is not None:
                done.append(utterance)
        self._pending = self._pending[n * WINDOW :].copy()
        return done

    def _step(self, window: np.ndarray) -> np.ndarray | None:
        prob = self.vad(window)
        self.peak_prob = max(self.peak_prob, prob)
        self.feed_peak = max(self.feed_peak, prob)
        speech = prob >= self.threshold
        if not self._speaking:
            self._preroll.append(window)
            self._run = self._run + 1 if speech else 0
            if self._run >= self.start_windows:
                self._speaking = True
                self._windows = list(self._preroll)
                self._preroll.clear()
                self._speech_windows = self._run
                self._silence = 0
            return None
        self._windows.append(window)
        if speech:
            self._speech_windows += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self.silence_windows or len(self._windows) >= self.max_windows:
            enough = self._speech_windows >= self.min_speech_windows
            self.last_speech_windows = self._speech_windows
            utterance = np.concatenate(self._windows)
            self._speaking = False
            self._windows = []
            self._run = 0
            self._speech_windows = 0
            self._silence = 0
            self.vad.reset()
            return utterance if enough else None
        return None


def to_mono_16k(sample_rate: int, audio: np.ndarray) -> np.ndarray:
    """Robot mic frame (int16 or float32, mono or channels-last stereo) -> 16 kHz mono float32."""
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = audio[:, 0]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32, copy=False)
    if sample_rate != SAMPLE_RATE and len(audio):
        n = int(round(len(audio) * SAMPLE_RATE / sample_rate))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio
