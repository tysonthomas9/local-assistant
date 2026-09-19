"""Compare speech-to-text models on the robot-microphone clips in tests/fixtures/stt/.

Run with the speech-to-speech venv, on the GPU the speech server uses (start_local_backend.sh
prints it: "Starting speech server (..., GPU N)"):

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N third_party/speech-to-speech/.venv/bin/python local_backend/stt_benchmark.py

Prints a per-clip table and appends a summary to local_backend/logs/stt_benchmark.jsonl.
"""

import json
import re
import subprocess
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CLIPS = sorted((HERE / "tests" / "fixtures" / "stt").glob("*.wav"))
REFERENCE = {
    "en_reachy_joke": "hey reachy tell me a joke",
    "en_reachy_look_dance": "reachy look to your left and then dance",
    "en_weather_paris": "what's the weather like in paris today",
}
REPEATS = 3


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def gpu_mib_of_this_process() -> int:
    import os
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(m) for p, m in (l.split(",") for l in out.strip().splitlines()) if int(p) == os.getpid())


class Parakeet:
    name = "parakeet-tdt-0.6b-v3"

    def __init__(self):
        from nano_parakeet import from_pretrained
        self.model = from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3", device="cuda")

    def __call__(self, audio):
        out = self.model.transcribe(audio)
        text = out if isinstance(out, str) else getattr(out, "text", None) or (out[0] if out else "")
        return (text if isinstance(text, str) else str(text)), "-"


class FasterWhisper:
    def __init__(self, model: str, hotwords: str | None = None):
        from faster_whisper import WhisperModel
        self.name = f"whisper-{model}" + (" +hotword" if hotwords else "")
        self.model = WhisperModel(model, device="cuda", compute_type="float16")
        self.hotwords = hotwords

    def __call__(self, audio):
        segs, info = self.model.transcribe(audio, beam_size=1, vad_filter=False, hotwords=self.hotwords,
                                           condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs), f"{info.language} {info.language_probability:.2f}"


def bench(factory) -> dict:
    before = gpu_mib_of_this_process()
    t0 = time.time(); engine = factory(); load_s = time.time() - t0
    engine(load(CLIPS[0]))  # warm-up
    mem = gpu_mib_of_this_process() - before
    rows = []
    for clip in CLIPS:
        audio = load(clip)
        times, text, lang = [], "", ""
        for _ in range(REPEATS):
            t = time.time(); text, lang = engine(audio); times.append(time.time() - t)
        rows.append({"clip": clip.stem, "ms": round(1000 * sorted(times)[REPEATS // 2]), "lang": lang, "text": text.strip(),
                     "wer": round(wer(REFERENCE[clip.stem], text), 2) if clip.stem in REFERENCE else None})
    del engine
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    return {"engine": getattr(factory, "label", None), "load_s": round(load_s, 1), "gpu_mib": mem, "rows": rows}


def main():
    engines = [
        ("parakeet-tdt-0.6b-v3", Parakeet),
        ("whisper-large-v3", lambda: FasterWhisper("large-v3")),
        ("whisper-large-v3 +hotword", lambda: FasterWhisper("large-v3", hotwords="Reachy")),
        ("whisper-large-v3-turbo", lambda: FasterWhisper("large-v3-turbo")),
        ("whisper-large-v3-turbo +hotword", lambda: FasterWhisper("large-v3-turbo", hotwords="Reachy")),
    ]
    results = []
    for label, factory in engines:
        factory.label = label if not hasattr(factory, "label") else factory.label
        r = bench(factory); r["engine"] = label; results.append(r)
        known = [x["wer"] for x in r["rows"] if x["wer"] is not None]
        print(f"\n=== {label}: load {r['load_s']} s, GPU +{r['gpu_mib']} MiB, "
              f"median latency {int(np.median([x['ms'] for x in r['rows']]))} ms, WER on known clips {np.mean(known):.2f}")
        for x in r["rows"]:
            print(f"  {x['clip']:22s} {x['ms']:5d} ms  [{x['lang']:>7s}]  wer={x['wer']!s:5s} {x['text'][:90]!r}")
    log = HERE / "logs" / "stt_benchmark.jsonl"
    log.parent.mkdir(exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "results": results}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
