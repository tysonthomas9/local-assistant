"""Score DX-POC recordings.

For each run dir (runs/<model>/<scenario>/<take>.{wav,json}) this transcribes the model
channel (right) with Whisper on CPU, finds model talk spurts by energy, and computes the
scorecard's timing metrics. Judgement calls (is the answer right, is the English good) are
printed alongside the transcript for a human to confirm.

    uv run --project analysis python analysis/score.py RUNS_DIR MODEL [TAKE]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dxbench.audio import speech_segments  # noqa: E402
from dxbench.scenarios import by_id  # noqa: E402

_W = None


def whisper():
    global _W
    if _W is None:
        from faster_whisper import WhisperModel
        _W = WhisperModel("medium.en", device="cpu", compute_type="int8", cpu_threads=12)
    return _W


def transcribe(x: np.ndarray, sr: int) -> list[dict]:
    from dxbench.audio import resample
    y = resample(x.astype(np.float32), sr, 16000)
    segs, _ = whisper().transcribe(y, language="en", vad_filter=True, word_timestamps=True,
                                   condition_on_previous_text=False)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]


def talking_frac(segs, a: float, b: float) -> float:
    if b <= a:
        return 0.0
    tot = sum(max(0.0, min(e, b) - max(s, a)) for s, e in segs)
    return tot / (b - a)


def onset_after(segs, t_end: float, t_start: float) -> float | None:
    """Model reply onset for a user clip [t_start, t_end]: first spurt that starts after the
    clip starts and still runs after the clip ends (so a reply, not an earlier greeting)."""
    for s, e in segs:
        if s > t_start + 0.2 and e > t_end + 0.2:
            return s
    return None


def text_between(tr, a: float, b: float = 1e9) -> str:
    return " ".join(t["text"] for t in tr if t["end"] > a and t["start"] < b)


def score(run: Path, sid: str, take: str) -> dict:
    sc = by_id(sid)
    x, sr = sf.read(run / sid / f"{take}.wav")
    log = json.loads((run / sid / f"{take}.json").read_text())
    model = x[:, 1]
    segs = speech_segments(model, sr)
    tr = transcribe(model, sr)
    clips = log["clips"]
    r: dict = {"scenario": sid, "segments": [(round(s, 2), round(e, 2)) for s, e in segs], "transcript": tr,
               "tools": log.get("tools", []), "flushes": log.get("flushes", []), "notes": log.get("trigger_notes", []),
               "peak_mib": log.get("peak_mib"), "behind": log.get("behind_realtime_s")}
    q = [c for c in clips if c["role"] in ("question", "turn")]
    if sid == "s1_pause":
        a, b, c = clips
        pauses = [(a["end"], b["start"]), (b["end"], c["start"])]
        r["pause_talk_s"] = round(sum(talking_frac(segs, p, q2) * (q2 - p) for p, q2 in pauses), 2)
        r["takes_over_in_pause"] = r["pause_talk_s"] >= 0.3
        on = onset_after(segs, c["end"], c["start"] - 0.2)
        r["latency_s"] = None if on is None else round(on - c["end"], 2)
        r["answer"] = text_between(tr, a["start"])
    elif sid in ("s2_yeah", "s3_interrupt", "s4_cough"):
        qc, ov = clips[0], clips[1]
        on = onset_after(segs, qc["end"], qc["start"])
        r["latency_s"] = None if on is None else round(on - qc["end"], 2)
        # 1 s look-back: chunked models (MiniCPM-o, 1 s units) leave short gaps between chunks
        r["talking_at_overlap"] = talking_frac(segs, ov["start"] - 1.0, ov["start"]) > 0.3
        # first silence >= 0.5 s that starts at/after the overlap onset
        stop = None
        for s, e in segs:
            if e >= ov["start"] - 0.05:
                stop = e
                break
        r["stop_after_overlap_s"] = None if stop is None else round(stop - ov["start"], 2)
        r["stop_after_overlap_end_s"] = None if stop is None else round(stop - ov["end"], 2)
        r["talk_frac_overlap_to_+2s"] = round(talking_frac(segs, ov["start"], ov["end"] + 2.0), 2)
        if sid == "s3_interrupt":
            r["stops"] = bool(r["talking_at_overlap"] and stop is not None and stop - ov["start"] <= 2.0)
            r["after_interrupt"] = text_between(tr, ov["start"] + 0.3)
        else:
            r["keeps_talking"] = bool(r["talking_at_overlap"] and r["talk_frac_overlap_to_+2s"] >= 0.6)
            r["text_around_overlap"] = text_between(tr, ov["start"] - 2.0, ov["end"] + 3.0)
        r["answer"] = text_between(tr, qc["start"])
    else:
        lat = []
        for c in q:
            on = onset_after(segs, c["end"], c["start"])
            lat.append(None if on is None else round(on - c["end"], 2))
        r["latency_s"] = lat[0] if len(lat) == 1 else lat
        r["answer"] = text_between(tr, q[0]["start"])
        if sid == "s8_multiturn":
            r["final_answer"] = text_between(tr, q[-1]["start"])
    hay = r.get("final_answer", r.get("after_interrupt", r["answer"]))
    r["expect_hit"] = bool(re.search(sc.expect, hay, re.I)) if sc.expect else None
    return r


def main() -> None:
    runs, model = Path(sys.argv[1]), sys.argv[2]
    take = sys.argv[3] if len(sys.argv) > 3 else "t1"
    run = runs / model
    out = {}
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        if (d / f"{take}.wav").exists():
            out[d.name] = score(run, d.name, take)
            r = out[d.name]
            keys = [k for k in r if k not in ("transcript", "segments")]
            print(json.dumps({k: r[k] for k in keys}, ensure_ascii=False))
    (run / f"score_{take}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
