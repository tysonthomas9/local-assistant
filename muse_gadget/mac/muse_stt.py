"""Local speech-to-text on the Mac (Apple silicon, MLX): parakeet-mlx, falling back to mlx-whisper.

Both take 16 kHz mono float32 audio straight from memory (no ffmpeg needed). Models are
downloaded once by install.sh into HF_HOME (under ~/assistant-edge/muse-app) and loaded offline
afterwards. MUSE_STT=parakeet|whisper forces one engine; MUSE_STT_MODEL overrides its model.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"

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


def make_transcriber() -> tuple[str, Transcriber]:
    """Return (engine name, transcriber): parakeet-mlx first, mlx-whisper if that fails."""
    forced = os.environ.get("MUSE_STT", "").strip().lower()
    model = os.environ.get("MUSE_STT_MODEL") or None
    if forced in ("", "parakeet"):
        try:
            return "parakeet-mlx", parakeet_transcriber(model or PARAKEET_MODEL)
        except Exception as e:
            if forced:
                raise
            logger.warning("parakeet-mlx unavailable (%s); falling back to mlx-whisper", e)
    return "mlx-whisper", whisper_transcriber(model or WHISPER_MODEL)
