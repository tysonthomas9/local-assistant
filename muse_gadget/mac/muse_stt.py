"""Local speech-to-text on the Mac (Apple silicon, MLX): parakeet-mlx, falling back to mlx-whisper.

Both take 16 kHz mono float32 audio straight from memory (no ffmpeg needed). Models are
downloaded once by install.sh into HF_HOME (under ~/assistant-edge/muse-app) and loaded offline
afterwards. MUSE_STT=parakeet|whisper|qwen3-asr forces one engine; MUSE_STT_MODEL overrides its model.

Qwen3-ASR (MUSE_STT=qwen3-asr) needs mlx-audio, which the app venv's pins don't allow, so it runs
as qwen3_asr_worker.py in the Kokoro venv (like the Qwen3-TTS worker). MUSE_STT_MODEL picks the
model: 0.6b (default, mlx-community/Qwen3-ASR-0.6B-bf16), 1.7b (mlx-community/Qwen3-ASR-1.7B-8bit)
or a full repo id. If it can't load, parakeet is used (and a warning logged). Only one STT model
is loaded at a time.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

import numpy as np

from muse_tts import KokoroEngine, worker_command

logger = logging.getLogger(__name__)

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
QWEN3_ASR_MODELS = {"0.6b": "mlx-community/Qwen3-ASR-0.6B-bf16", "1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit"}
QWEN3_ASR_MODEL = QWEN3_ASR_MODELS["0.6b"]

Transcriber = Callable[[np.ndarray], str]


def parakeet_transcriber(model_id: str = PARAKEET_MODEL) -> Transcriber:
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel

    model = from_pretrained(model_id)
    config = model.preprocessor_config

    def transcribe(audio: np.ndarray) -> str:
        if len(audio) < config.hop_length:
            return ""
        mel = get_logmel(mx.array(audio.astype(np.float32)), config)  # float32, as load_audio gives
        return model.generate(mel)[0].text.strip()

    transcribe(np.zeros(16000, dtype=np.float32))  # warm up (the first call takes ~2 s)
    return transcribe


def whisper_transcriber(model_id: str = WHISPER_MODEL) -> Transcriber:
    import mlx_whisper

    def transcribe(audio: np.ndarray) -> str:
        result = mlx_whisper.transcribe(
            audio.astype(np.float32), path_or_hf_repo=model_id, language="en", verbose=None
        )
        return str(result.get("text", "")).strip()

    transcribe(np.zeros(16000, dtype=np.float32))  # load the model now, not on the first turn
    return transcribe


def qwen3_asr_model(model: str | None) -> str:
    """0.6b / 1.7b (any case) -> the repo id; anything else is taken as a repo id; None -> 0.6B."""
    if not model:
        return QWEN3_ASR_MODEL
    return QWEN3_ASR_MODELS.get(model.strip().lower(), model.strip())


class Qwen3AsrEngine(KokoroEngine):
    """Client for qwen3_asr_worker.py (same start-up, framing and timeouts as the TTS workers)."""

    def __init__(self, cmd: list[str], start_timeout_s: float = 180.0, timeout_s: float = 30.0) -> None:
        super().__init__(cmd, start_timeout_s=start_timeout_s, synth_timeout_s=timeout_s)

    def __call__(self, audio: np.ndarray) -> str:
        with self._lock:
            if self.proc.poll() is not None:
                raise EOFError("qwen3-asr worker exited")
            try:
                data = np.asarray(audio, dtype="<f4").reshape(-1).tobytes()
                self.proc.stdin.write(f"audio {len(data) // 4}\n".encode() + data)  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
                deadline = time.monotonic() + self.synth_timeout_s
                header = self._read_line(deadline)
                if header.startswith("err"):
                    raise RuntimeError(f"qwen3-asr worker: {header}")
                if not header.startswith("text "):
                    raise ValueError("qwen3-asr worker: bad header")
                return self._read_exact(int(header.split()[1]), deadline).decode(errors="replace").strip()
            except (OSError, EOFError, TimeoutError, ValueError):
                self.close()  # out of step with the worker: don't reuse it
                raise


def qwen3_asr_command(model: str) -> list[str]:
    return worker_command("qwen3_asr_worker.py", "--model", model)


def qwen3_asr_transcriber(model_id: str = QWEN3_ASR_MODEL) -> Transcriber:
    return Qwen3AsrEngine(qwen3_asr_command(model_id))


def make_transcriber() -> tuple[str, Transcriber]:
    """Return (engine name, transcriber): parakeet-mlx first, mlx-whisper if that fails.
    MUSE_STT=qwen3-asr tries Qwen3-ASR first and falls back to parakeet (then whisper)."""
    forced = os.environ.get("MUSE_STT", "").strip().lower()
    model = os.environ.get("MUSE_STT_MODEL") or None
    if forced == "qwen3-asr":
        repo = qwen3_asr_model(model)
        try:
            return f"qwen3-asr ({repo.rsplit('/', 1)[-1]})", qwen3_asr_transcriber(repo)
        except Exception as e:
            logger.warning("Qwen3-ASR unavailable (%s); falling back to parakeet-mlx", e)
        forced, model = "", None
    if forced in ("", "parakeet"):
        try:
            return "parakeet-mlx", parakeet_transcriber(model or PARAKEET_MODEL)
        except Exception as e:
            if forced:
                raise
            logger.warning("parakeet-mlx unavailable (%s); falling back to mlx-whisper", e)
    return "mlx-whisper", whisper_transcriber(model or WHISPER_MODEL)
