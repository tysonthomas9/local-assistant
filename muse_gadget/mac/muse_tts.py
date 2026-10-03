"""Text-to-speech with macOS's built-in `say`: text -> 16-bit mono PCM at the robot's rate.

`say -o reply.wav --file-format=WAVE --data-format=LEI16@<rate>` writes the audio without
playing it, so the app can push it to the robot's speaker (and wobble the head) itself.
MUSE_SAY_VOICE picks a voice (`say -v '?'` lists them); empty means the system voice.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

SAY = "/usr/bin/say"


def say_command(text_file: str, out: str, sample_rate: int, voice: str | None) -> list[str]:
    cmd = [SAY, "-o", out, "--file-format=WAVE", f"--data-format=LEI16@{sample_rate}"]
    if voice:
        cmd += ["-v", voice]
    return cmd + ["-f", text_file]


def read_wav_int16(path: str | Path) -> tuple[int, np.ndarray]:
    """(sample rate, mono int16 samples) of a 16-bit PCM WAV."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0].copy()
    return rate, pcm


def synthesize(text: str, sample_rate: int = 16000, voice: str | None = None, timeout_s: float = 60) -> np.ndarray:
    """Render `text` with `say`; return mono int16 PCM at `sample_rate`."""
    voice = voice if voice is not None else (os.environ.get("MUSE_SAY_VOICE") or None)
    with tempfile.TemporaryDirectory(prefix="muse-say-") as tmp:
        text_file = os.path.join(tmp, "text.txt")
        out = os.path.join(tmp, "reply.wav")
        Path(text_file).write_text(text, encoding="utf-8")  # -f: no text on the command line
        subprocess.run(say_command(text_file, out, sample_rate, voice), check=True, timeout=timeout_s,
                       stdin=subprocess.DEVNULL, capture_output=True)
        rate, pcm = read_wav_int16(out)
    if rate != sample_rate:
        raise RuntimeError(f"say wrote {rate} Hz, asked for {sample_rate} Hz")
    return pcm
