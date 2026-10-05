"""Audio levels and test signals for s16le mono PCM (pure Python, no audio library)."""

import math
from array import array

SILENCE_DBFS = -120.0


def dbfs(pcm: bytes) -> float:
    """RMS level of s16le PCM in dBFS (`SILENCE_DBFS` for digital silence)."""
    samples = array("h", pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return SILENCE_DBFS
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768
    return 20 * math.log10(rms) if rms > 1e-6 else SILENCE_DBFS


def tone(hz: float, seconds: float, rate: int, level_dbfs: float = -12.0) -> bytes:
    """A sine tone as s16le mono, with 10 ms fades so it starts and ends without a click."""
    n = int(seconds * rate)
    amplitude = 32767 * 10 ** (level_dbfs / 20) * math.sqrt(2)
    fade = max(1, rate // 100)
    out = array("h", bytes(2 * n))
    for i in range(n):
        gain = min(1.0, i / fade, (n - 1 - i) / fade)
        out[i] = int(
            max(-32768, min(32767, amplitude * gain * math.sin(2 * math.pi * hz * i / rate)))
        )
    return out.tobytes()
