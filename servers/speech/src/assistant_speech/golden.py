"""Make the golden WAVs (tests/fixtures/audio/*.wav) with the speech server's own TTS.

    python -m assistant_speech.golden --url http://127.0.0.1:8772 --out tests/fixtures/audio

Each entry of `<out>/golden.toml` (`[<name>] text = "...", voice = "aiden"`) becomes
`<out>/<name>.wav`: 16 kHz mono s16le, the synthetic voice (no personal recordings) with
`lead_s` of digital silence before and `tail_s` after, so an edge's voice-activity detector
sees a clear start and end. Run once; the WAVs are committed. Existing files are kept unless
`--force` is given.
"""

import argparse
import json
import tomllib
import urllib.request
import wave
from pathlib import Path

import numpy as np

from assistant_speech.server import STT_RATE, TTS_RATE, pcm16_to_float, resample


def synthesize(url: str, text: str, voice: str) -> np.ndarray:
    request = urllib.request.Request(
        f"{url}/v1/audio/speech",
        data=json.dumps({"input": text, "voice": voice}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return pcm16_to_float(response.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_speech.golden")
    parser.add_argument("--url", default="http://127.0.0.1:8772")
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/audio"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    golden = tomllib.loads((args.out / "golden.toml").read_text())
    for name, entry in golden.items():
        path = args.out / f"{name}.wav"
        if path.exists() and not args.force:
            print(f"kept {path}")
            continue
        audio = resample(synthesize(args.url, entry["text"], entry["voice"]), TTS_RATE, STT_RATE)
        peak = float(np.abs(audio).max()) or 1.0
        audio = audio * (0.5 / peak)  # -6 dBFS peak
        lead = np.zeros(int(STT_RATE * entry.get("lead_s", 0.3)), dtype=np.float32)
        tail = np.zeros(int(STT_RATE * entry.get("tail_s", 1.0)), dtype=np.float32)
        pcm = (np.concatenate([lead, audio, tail]) * 32767).astype("<i2").tobytes()
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(STT_RATE)
            wav.writeframes(pcm)
        print(f"wrote {path}: {len(pcm) / 2 / STT_RATE:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
