"""Smart Turn v3.2: is the user's turn over, or only paused? (`[engine] smart_turn`)

Pipecat's end-of-turn model (pipecat-ai/smart-turn-v3, BSD-2), as the speech-to-speech
pipeline of Pollen's app runs it (third_party/speech-to-speech, VAD/smart_turn.py): once the
speech detector hears the end silence, the model looks at the last 8 s of the turn's audio
(its sound and wording) and says whether the turn sounds complete. A turn that stops
mid-sentence ("What I'd like to know is... ") is held open a little longer.

The model is a Whisper-tiny encoder over Whisper log-mel features; the features are computed
here with numpy (the same as transformers' `WhisperFeatureExtractor`, 8 s chunk), so the edge
needs only onnxruntime. The ONNX file comes from the Hugging Face cache (`HF_HOME`), put
there by scripts/edge_host_bootstrap.sh on the robot's machine.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_REPO = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
RATE = 16000
SECONDS = 8
N_FFT = 400
HOP = 160
N_MELS = 80


@dataclass(frozen=True)
class TurnEnd:
    complete: bool
    probability: float
    inference_ms: float


def model_path() -> Path:
    """The cached ONNX file (`HF_HOME`, else ~/.cache/huggingface); FileNotFoundError if none."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    repo = home / "hub" / f"models--{MODEL_REPO.replace('/', '--')}"
    found = sorted(repo.glob(f"snapshots/*/{MODEL_FILE}"))
    if not found:
        raise FileNotFoundError(
            f"Smart Turn model {MODEL_REPO}/{MODEL_FILE} is not in the Hugging Face cache "
            f"({repo}): run scripts/edge_host_bootstrap.sh, or set [engine] smart_turn = false"
        )
    return found[-1]


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale (linear below 1 kHz, logarithmic above)."""
    hz = np.asarray(hz, dtype=np.float64)
    mel = 3.0 * hz / 200.0
    log = hz >= 1000.0
    return np.where(log, 15.0 + np.log(np.maximum(hz, 1e-10) / 1000.0) * (27.0 / np.log(6.4)), mel)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    hz = 200.0 * mel / 3.0
    return np.where(mel >= 15.0, 1000.0 * np.exp(np.log(6.4) / 27.0 * (mel - 15.0)), hz)


def mel_filters() -> np.ndarray:
    """(N_FFT // 2 + 1, N_MELS) triangular Slaney-normalised filters, 0-8 kHz (Whisper's)."""
    fft_freqs = np.linspace(0, RATE // 2, N_FFT // 2 + 1)
    mels = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(RATE / 2)), N_MELS + 2)
    freqs = _mel_to_hz(mels)
    diff = np.diff(freqs)
    slopes = freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    return filters * (2.0 / (freqs[2 : N_MELS + 2] - freqs[:N_MELS]))[None, :]


def features(audio: np.ndarray) -> np.ndarray:
    """Whisper log-mel features of up to `SECONDS` of 16 kHz float audio: (1, 80, 800)."""
    audio = np.asarray(audio, dtype=np.float32)[-SECONDS * RATE :]
    audio = np.pad(audio, (SECONDS * RATE - audio.size, 0))  # silence before, as the model expects
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    padded = np.pad(audio.astype(np.float64), N_FFT // 2, mode="reflect")
    frames = 1 + (padded.size - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
    power = np.abs(np.fft.rfft(padded[index] * window, n=N_FFT)) ** 2
    mel = mel_filters().T @ power.T
    log = np.log10(np.maximum(mel, 1e-10))[:, :-1]
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)[None, :, :]


class SmartTurn:
    """The model on the CPU (one thread); `predict` takes s16le mono 16 kHz PCM."""

    def __init__(self, path: Path | None = None, threshold: float = 0.5) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self.path = path or model_path()
        self.threshold = threshold
        self.session = ort.InferenceSession(
            str(self.path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        self.predict(b"\0\0" * RATE)  # warm up

    def predict(self, pcm: bytes) -> TurnEnd:
        started = time.perf_counter()
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        out = self.session.run(None, {self.input: features(audio)})
        probability = float(np.asarray(out[0]).reshape(-1)[0])
        return TurnEnd(
            complete=probability > self.threshold,
            probability=probability,
            inference_ms=(time.perf_counter() - started) * 1000,
        )
