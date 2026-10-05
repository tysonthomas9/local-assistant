"""Small audio helpers shared by every model environment (numpy, scipy, soundfile only)."""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REC_SR = 24000  # recording rate for both channels


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or x.size == 0:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load_mono(path: str, sr: int) -> np.ndarray:
    x, s = sf.read(path, dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), s, sr)


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * np.log10(max(r, 1e-6))


def frame_db(x: np.ndarray, sr: int, hop_s: float = 0.02) -> np.ndarray:
    hop = int(sr * hop_s)
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    f = x[: n * hop].reshape(n, hop).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(f * f, axis=1)), 1e-6))


def speech_segments(x: np.ndarray, sr: int, thr_db: float = -42.0, min_gap: float = 0.35,
                    min_len: float = 0.12, hop_s: float = 0.02) -> list[tuple[float, float]]:
    """Energy-based talk spurts: [(start_s, end_s)], merging gaps shorter than min_gap."""
    db = frame_db(x, sr, hop_s)
    on = db > thr_db
    segs: list[list[float]] = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            s, e = i * hop_s, j * hop_s
            if segs and s - segs[-1][1] < min_gap:
                segs[-1][1] = e
            else:
                segs.append([s, e])
            i = j
        else:
            i += 1
    return [(s, e) for s, e in segs if e - s >= min_len]


def write_stereo(path: str, left: np.ndarray, right: np.ndarray, sr: int = REC_SR) -> None:
    n = max(len(left), len(right))
    st = np.zeros((n, 2), dtype=np.float32)
    st[: len(left), 0] = left
    st[: len(right), 1] = right
    sf.write(path, np.clip(st, -1.0, 1.0), sr, subtype="PCM_16")
