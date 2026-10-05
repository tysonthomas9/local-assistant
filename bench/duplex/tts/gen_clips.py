"""Generate the DX-POC scenario clips (24 kHz mono WAV, one per clip id).

Voice: Kokoro-82M stock voice ``am_michael`` (synthetic; no real-person voice cloning).
The cough is synthesised from shaped noise, not recorded.

    uv run --project tts python tts/gen_clips.py OUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxbench.scenarios import CLIPS  # noqa: E402

SR = 24000
VOICE = "am_michael"
TARGET_DB = -20.0


def trim(x: np.ndarray, thr: float = 0.01, pad: float = 0.05) -> np.ndarray:
    idx = np.nonzero(np.abs(x) > thr)[0]
    if idx.size == 0:
        return x
    p = int(pad * SR)
    return x[max(0, idx[0] - p): idx[-1] + p]


def norm(x: np.ndarray) -> np.ndarray:
    r = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x * (10 ** (TARGET_DB / 20) / r)).astype(np.float32)


def cough(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for dur, f0 in ((0.28, 180.0), (0.22, 160.0)):
        n = int(dur * SR)
        t = np.arange(n) / SR
        env = (1 - np.exp(-t / 0.008)) * np.exp(-t / (dur / 3.5))
        noise = sosfilt(butter(4, [250, 3500], btype="band", fs=SR, output="sos"), rng.standard_normal(n))
        voiced = 0.25 * np.sign(np.sin(2 * np.pi * f0 * t)) * np.exp(-t / 0.05)
        parts += [(noise + voiced) * env, np.zeros(int(0.14 * SR))]
    return norm(np.concatenate(parts[:-1]))


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from kokoro import KPipeline

    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    for cid, text in CLIPS.items():
        if cid == "cough":
            x = cough()
        else:
            audio = [r.audio.numpy() for r in pipe(text, voice=VOICE, speed=1.0)]
            x = norm(trim(np.concatenate(audio)))
        sf.write(out / f"{cid}.wav", x, SR, subtype="PCM_16")
        print(f"{cid}: {len(x) / SR:.2f}s  {text!r}")


if __name__ == "__main__":
    main()
