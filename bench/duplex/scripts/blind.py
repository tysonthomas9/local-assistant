"""Blind-label DX-POC recordings and render them as small MP4s for listening.

Each model gets one random letter (the same letter in every scenario). Every take becomes
an MP4: a title card with the scenario and the letter, the whole take's split-channel waveform
(top = user, bottom = model) with a moving playhead, and the stereo audio (left = user,
right = model).

    python scripts/blind.py RUNS_DIR OUT_DIR [TAKE]

Writes OUT_DIR/key.json (letter -> model), or reuses it if present so every take keeps the same
letters. Keep the key away from listeners.
Needs ffmpeg with drawtext on PATH.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxbench.scenarios import SCENARIOS  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def render(wav: Path, out: Path, title: str, sub: str) -> None:
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(wav)], capture_output=True, text=True, check=True).stdout)
    pic = out.with_suffix(".png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-filter_complex",
                    "showwavespic=s=1280x360:split_channels=1:colors=0x4f9dff|0xffb347:scale=sqrt",
                    "-frames:v", "1", str(pic)], check=True)
    vf = (
        "color=c=0x14161a:s=1280x540:r=10[bg];color=c=white:s=3x360:r=10[bar];[bg][1:v]overlay=0:170[o];"
        f"[o][bar]overlay=x='1280*t/{dur:.3f}':y=170:shortest=0,"
        f"drawtext=fontfile={FONT}:text='{esc(title)}':fontcolor=white:fontsize=40:x=40:y=40,"
        f"drawtext=fontfile={FONT}:text='{esc(sub)}':fontcolor=0xb0b0b0:fontsize=24:x=40:y=105,"
        f"drawtext=fontfile={FONT}:text='%{{pts\\:hms}}':fontcolor=0xb0b0b0:fontsize=22:x=1110:y=105[v]"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-loop", "1", "-i", str(pic),
         "-filter_complex", vf, "-map", "[v]", "-map", "0:a", "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-r", "10",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)],
        check=True,
    )
    pic.unlink()


def main() -> None:
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    take = sys.argv[3] if len(sys.argv) > 3 else "t1"
    out.mkdir(parents=True, exist_ok=True)
    models = sorted(p.name for p in runs.iterdir() if p.is_dir() and any(p.glob(f"*/{take}.wav")))
    kf = out / "key.json"
    if kf.exists():  # reuse the letters from an earlier take
        key = json.loads(kf.read_text())
    else:
        letters = [chr(ord("A") + i) for i in range(len(models))]
        order = list(models)
        for i in range(len(order) - 1, 0, -1):  # Fisher-Yates with a CSPRNG
            j = secrets.randbelow(i + 1)
            order[i], order[j] = order[j], order[i]
        key = dict(zip(letters, order))
        kf.write_text(json.dumps(key, indent=1))
    for n, sc in enumerate(SCENARIOS, 1):
        for letter, model in key.items():
            wav = runs / model / sc.id / f"{take}.wav"
            if wav.exists():
                render(wav, out / f"{sc.id}_{letter}_{take}.mp4", f"S{n}. {sc.title}  -  Model {letter}  ({take})",
                       "top / left: user (TTS)    bottom / right: model")
                print(out / f"{sc.id}_{letter}_{take}.mp4")


if __name__ == "__main__":
    main()
