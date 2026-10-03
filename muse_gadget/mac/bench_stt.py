"""Speech-to-text benchmark on the Mac: Nemotron 3.5 ASR and Qwen3-ASR (mlx-audio) vs parakeet-mlx (today's engine).

Phase A of "Nemotron 3.5 ASR vs Parakeet on the Mac": a benchmark only, nothing in the app changes.
Run it on the Mac in its own venv (mlx-audio needs a newer transformers than the app pins):

    B=~/assistant-edge/muse-app/stt-bench
    uv venv --python 3.12 $B/.venv
    uv pip install --python $B/.venv/bin/python mlx==0.32.3 "mlx-audio[stt]==0.5.7" parakeet-mlx==0.5.3
    HF_HOME=~/assistant-edge/muse-app/hf $B/.venv/bin/python bench_stt.py clips --out $B/clips
    HF_HOME=~/assistant-edge/muse-app/hf $B/.venv/bin/python bench_stt.py all --clips $B/clips --out $B/results

`clips` writes 12 clips with macOS `say` (16 kHz mono): 10 clean questions and 2 with background
noise. Each clip is 0.3 s of silence, the speech, then 0.8 s of silence (MuseHandler's VAD ends a
turn after 0.8 s of quiet, so that is the audio the STT gets today).
`all` runs every engine in its own process (clean load time and peak memory) and prints a table.

Metrics per engine:
  WER          word error rate against the known text (lower case, no punctuation)
  load         model load from the local cache (offline), plus the first warm-up call
  after end    time from the end of the clip (= when the VAD hands the turn over) to the final
               text. Offline engines transcribe the whole clip then; streaming engines were fed
               the clip in real time, chunk by chunk, so only the flush is left.
  first part.  streaming: time from the start of speech to the first partial text
  last word    streaming: when all the final words were there (ignoring case and punctuation),
               relative to the end of speech (negative = before the speaker stopped)
  RTF          offline: compute time / audio length
  peak mem     mx.get_peak_memory() (MLX allocations) and the process's max RSS
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

SR = 16000
LEAD_S = 0.3
TAIL_S = 0.8  # muse_vad.Utterance silence_s
NEMOTRON = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
PARAKEET = "mlx-community/parakeet-tdt-0.6b-v3"
QWEN3 = {"qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-bf16", "qwen3-asr-1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit"}

# (text, voice). No digits, so WER doesn't depend on number formatting.
QUESTIONS = [
    ("What's the weather like in Paris tomorrow morning?", "Samantha"),
    ("Can you set a timer for ten minutes?", "Daniel"),
    ("Who wrote the novel Pride and Prejudice?", "Karen"),
    ("How far is the moon from the earth?", "Moira"),
    ("Tell me a short joke about robots.", "Samantha"),
    ("What time does the bakery on Main Street open?", "Daniel"),
    ("Could you remind me to call my sister tonight?", "Karen"),
    ("Why is the sky blue during the day?", "Moira"),
    ("Play some relaxing jazz music in the kitchen.", "Samantha"),
    ("What is the capital city of Australia?", "Daniel"),
]
NOISY = [
    ("Turn off the lights in the living room please.", "Samantha", "pink", 5.0),
    ("How many cups are in a gallon of milk?", "Daniel", "babble", 8.0),
]


# ----------------------------------------------------------------------------- audio helpers
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def say(text: str, voice: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "s.wav"
        cmd = ["say", "-v", voice, "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@16000", text]
        if subprocess.run(cmd, capture_output=True).returncode != 0:  # voice missing: default voice
            subprocess.run(cmd[:1] + cmd[3:], check=True, capture_output=True)
        return read_wav(out)


def trim(audio: np.ndarray, thresh: float = 0.01) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) > thresh)
    return audio[idx[0] : idx[-1] + 1] if idx.size else audio


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.maximum(np.arange(spec.size), 1))
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-9)


def babble_noise(n: int) -> np.ndarray:
    lines = [
        ("I think the meeting moved to Thursday afternoon, didn't it?", "Karen"),
        ("We should probably order more paper for the printer soon.", "Moira"),
        ("The game last night went into overtime again.", "Fred"),
    ]
    x = np.zeros(n, dtype=np.float32)
    for i, (t, v) in enumerate(lines):
        s = trim(say(t, v))
        s = np.tile(s, n // len(s) + 1)[i * 2000 : i * 2000 + n]
        x += s / (np.std(s) + 1e-9)
    return x / (np.std(x) + 1e-9)


def make_clips(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    manifest = []
    items = [(t, v, None, None) for t, v in QUESTIONS] + list(NOISY)
    for i, (text, voice, noise, snr_db) in enumerate(items):
        speech = trim(say(text, voice))
        speech = 0.5 * speech / (np.max(np.abs(speech)) + 1e-9)
        clip = np.concatenate([np.zeros(int(LEAD_S * SR)), speech, np.zeros(int(TAIL_S * SR))]).astype(np.float32)
        if noise:
            n = pink_noise(len(clip), rng) if noise == "pink" else babble_noise(len(clip))
            gain = np.std(speech) / (10 ** (snr_db / 20))
            clip = (clip + gain * n).astype(np.float32)
            clip *= 0.9 / max(0.9, float(np.max(np.abs(clip))))
        name = f"q{i:02d}{'_' + noise if noise else ''}.wav"
        write_wav(out / name, clip)
        manifest.append({
            "file": name, "text": text, "voice": voice, "noise": noise, "snr_db": snr_db,
            "speech_start": LEAD_S, "speech_end": LEAD_S + len(speech) / SR, "duration": len(clip) / SR,
        })
        print(f"{name}  {len(clip) / SR:4.1f}s  {voice:9s} {noise or 'clean':6s} {text}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


# ----------------------------------------------------------------------------- WER
def words(s: str) -> list[str]:
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", s).split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


# ----------------------------------------------------------------------------- engines
class Offline:
    """Transcribe the whole clip once it has ended (what MuseHandler does today)."""

    streaming = False

    def __init__(self, name: str):
        import mlx.core as mx

        self.mx = mx
        if name == "parakeet":
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel

            model = from_pretrained(PARAKEET)
            cfg = model.preprocessor_config
            # Same call as muse_stt.parakeet_transcriber
            self.fn = lambda a: model.generate(get_logmel(mx.array(a), cfg))[0].text.strip()
        elif name in QWEN3:  # Qwen3-ASR: not a streaming model, offline only
            from mlx_audio.stt import load

            model = load(QWEN3[name])
            self.fn = lambda a: model.generate(mx.array(a), language="English").text.strip()
        else:  # nemotron-offline: default look-ahead [56,13], best offline accuracy
            from mlx_audio.stt import load

            model = load(NEMOTRON)
            self.fn = lambda a: model.generate(mx.array(a), language="en-US").text.strip()

    def warm(self) -> None:
        self.fn(np.zeros(SR, dtype=np.float32))

    def run(self, audio: np.ndarray, meta: dict) -> dict:
        t = time.perf_counter()
        text = self.fn(audio)
        dt = time.perf_counter() - t
        return {"text": text, "after_end": dt, "rtf": dt / meta["duration"]}


class Streaming:
    """Feed the clip in real time (chunk_s pieces at audio pace), decode as it arrives."""

    streaming = True

    def __init__(self, name: str, chunk_s: float):
        import mlx.core as mx

        self.mx, self.name, self.chunk = mx, name, int(chunk_s * SR)
        if name.startswith("nemotron"):
            from mlx_audio.stt import load

            self.model = load(NEMOTRON)
            # nemotron-stream-<right>: trained look-aheads [56,0] 80 ms, [56,3] 320 ms, [56,6] 560 ms
            right = int(name.rsplit("-", 1)[1])
            self.model.default_att_context_size = [56, right]
        else:
            from parakeet_mlx import from_pretrained

            self.model = from_pretrained(PARAKEET)

    def warm(self) -> None:
        self.run(np.zeros(SR, dtype=np.float32), {"duration": 1.0, "speech_start": 0, "speech_end": 1}, realtime=False)

    def _session(self):
        if self.name.startswith("nemotron"):
            s = self.model.create_streaming_session(language="en-US")
            text: list[str] = []

            def drain():
                while True:
                    text.extend(s.step(max_decode_tokens=64))
                    if s.done or (not s._encoded and s._queued == 0):
                        return "".join(text).strip()

            def feed(a):
                s.feed(a)
                return drain()

            def finish():
                s.close()
                while not s.done:
                    text.extend(s.step(max_decode_tokens=64))
                return "".join(text).strip()

            return feed, finish, lambda: None
        # parakeet-mlx's own streaming mode (local attention, default 256/256 context)
        ctx = self.model.transcribe_stream(context_size=(256, 256))
        st = ctx.__enter__()

        def feed(a):
            st.add_audio(self.mx.array(a))
            return st.result.text.strip()

        def finish():  # parakeet has no flush: the draft tokens are the final text
            return st.result.text.strip()

        return feed, finish, lambda: ctx.__exit__(None, None, None)

    def run(self, audio: np.ndarray, meta: dict, realtime: bool = True) -> dict:
        feed, finish, close = self._session()
        events = []  # (seconds since the clip started, text so far)
        t0 = time.perf_counter()
        behind = 0.0
        try:
            starts = list(range(0, len(audio), self.chunk))
            if len(starts) > 1 and len(audio) - starts[-1] < SR // 25:
                starts.pop()  # fold a tail under 40 ms into the previous piece (parakeet's STFT needs >= n_fft)
            for k, i in enumerate(starts):
                piece = audio[i : starts[k + 1] if k + 1 < len(starts) else len(audio)]
                avail = (i + len(piece)) / SR  # the moment this piece has been "spoken"
                if realtime:
                    wait = t0 + avail - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                text = feed(piece)
                now = time.perf_counter() - t0
                behind = max(behind, now - avail)
                if not events or text != events[-1][1]:
                    events.append((now, text))
            end = len(audio) / SR if realtime else time.perf_counter() - t0
            text = finish()
            done = time.perf_counter() - t0
        finally:
            close()
        if not events or text != events[-1][1]:
            events.append((done, text))
        first = next((t for t, x in events if x), None)
        # all the final words first seen (punctuation and case can still change at the flush)
        last_change = next(t for t, x in events if words(x) == words(text))
        return {
            "text": text,
            "after_end": done - end,
            "first_partial": None if first is None else first - meta["speech_start"],
            "last_word": last_change - meta["speech_end"],
            "max_behind": behind,
        }


def make_engine(name: str, chunk_s: float):
    return Offline(name) if name in ("parakeet", "nemotron-offline", *QWEN3) else Streaming(name, chunk_s)


# ----------------------------------------------------------------------------- runner
def run_engine(name: str, clips: Path, chunk_s: float, out: Path) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import mlx.core as mx

    manifest = json.loads((clips / "manifest.json").read_text())
    mx.reset_peak_memory()
    t = time.perf_counter()
    eng = make_engine(name, chunk_s)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    eng.warm()
    warm_s = time.perf_counter() - t
    rows = []
    for m in manifest:
        r = eng.run(read_wav(clips / m["file"]), m)
        ref = words(m["text"])
        r.update(file=m["file"], noise=m["noise"], ref=m["text"], errors=edit_distance(ref, words(r["text"])), ref_words=len(ref))
        rows.append(r)
        print(f"{name:20s} {m['file']:16s} err={r['errors']} after_end={r['after_end'] * 1000:6.0f}ms  {r['text']}", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 1024
    result = {
        "engine": name, "chunk_s": chunk_s if eng.streaming else None, "streaming": eng.streaming,
        "load_s": load_s, "warm_s": warm_s, "peak_mlx_mb": mx.get_peak_memory() / 2**20, "max_rss_mb": rss_mb,
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=1))


def summarize(results: list[dict]) -> str:
    def ms(v):
        return "–" if v is None else f"{v * 1000:.0f}"

    head = ("| engine | WER clean | WER noisy | after end, median (max) ms | first partial ms | last word vs end of speech ms "
            "| behind real time, max ms | RTF | load + warm-up s | peak MLX / RSS MB |")
    lines = [head, "|" + "---|" * 10]
    for r in results:
        rows = r["rows"]
        clean = [x for x in rows if not x["noise"]]
        noisy = [x for x in rows if x["noise"]]
        wer = lambda xs: 100 * sum(x["errors"] for x in xs) / max(1, sum(x["ref_words"] for x in xs))
        ae = [x["after_end"] for x in rows]
        med = lambda k: (float(np.median([x[k] for x in rows if x.get(k) is not None])) if any(x.get(k) is not None for x in rows) else None)
        name = r["engine"] + (f" ({int(r['chunk_s'] * 1000)} ms feed)" if r["streaming"] else "")
        lines.append(
            f"| {name} | {wer(clean):.1f}% | {wer(noisy):.1f}% | {ms(float(np.median(ae)))} ({ms(max(ae))}) "
            f"| {ms(med('first_partial'))} | {ms(med('last_word'))} | {ms(med('max_behind')) if r['streaming'] else '–'} "
            f"| {'–' if r['streaming'] else format(med('rtf'), '.3f')} "
            f"| {r['load_s']:.1f} + {r['warm_s']:.1f} | {r['peak_mlx_mb']:.0f} / {r['max_rss_mb']:.0f} |"
        )
    return "\n".join(lines)


ENGINES = ["parakeet", "nemotron-offline", "nemotron-stream-0", "nemotron-stream-3", "nemotron-stream-6", "parakeet-stream",
           "qwen3-asr-0.6b", "qwen3-asr-1.7b"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clips", help="synthesize the test clips with `say`")
    c.add_argument("--out", type=Path, required=True)
    for n in ("run", "all"):
        s = sub.add_parser(n)
        s.add_argument("--clips", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--chunk", type=float, default=0.08, help="streaming feed size in seconds (default 0.08)")
        if n == "run":
            s.add_argument("engine", choices=ENGINES)
        else:
            s.add_argument("--engines", default=",".join(ENGINES))
    s = sub.add_parser("table", help="print the table from saved results")
    s.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    if a.cmd == "clips":
        make_clips(a.out)
    elif a.cmd == "run":
        run_engine(a.engine, a.clips, a.chunk, a.out)
    elif a.cmd == "all":
        names = a.engines.split(",")
        for n in names:  # one process per engine: clean load time and peak memory
            subprocess.run([sys.executable, __file__, "run", n, "--clips", str(a.clips), "--out", str(a.out),
                            "--chunk", str(a.chunk)], check=True)
        print(summarize([json.loads((a.out / f"{n}.json").read_text()) for n in names]))
    else:
        names = [n for n in ENGINES if (a.out / f"{n}.json").exists()]
        print(summarize([json.loads((a.out / f"{n}.json").read_text()) for n in names]))


if __name__ == "__main__":
    main()
