"""Speech-model bake-off (task G1): VRAM, latency and word error rate per STT/TTS candidate.

Each candidate is a real speech server (`python -m assistant_speech` with `--stt/--tts/...`),
started on its own, measured over HTTP exactly as the brain uses it, and stopped:

    python -m assistant_speech.bakeoff --out DIR [--only NAME ...] [--repeats 3]

- VRAM: the server process's GPU memory (nvidia-smi, sampled every 0.2 s): after load and the
  peak while measuring.
- STT: each golden clip of tests/fixtures/audio, plus the reference TTS clips (1.7B BF16, ryan
  and eric), sent whole after the speech ended (as the brain does): wall time per request and
  word error rate (lower case, no punctuation) against the expected text.
- TTS: each sentence in ryan's and eric's voice: time to first audio, synthesis real-time
  factor, and the round-trip word error rate (the reference Parakeet transcribes it). The audio
  is saved as WAV for listening (`DIR/clips/<candidate>/`).

Kokoro-82M (reference only: no Ryan/Eric voices) runs in-process with `--kokoro` from an env
that has the `kokoro` package; its round trip uses `--stt-url` (a running reference server).

Results: `DIR/results.json` (one entry per candidate) and a Markdown table on stdout.
"""

import argparse
import contextlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
GOLDEN = ROOT / "tests/fixtures/audio"
SPEECH_GOLDEN = [
    "hello",
    "capital_of_france",
    "weather_in_paris",
    "what_time_is_it",
    "tell_me_a_story",
    "whats_your_name",
    "hey_jarvis_whats_your_name",
    "story_start",
    "story_end",
    "what_time_is_it_in_noise",
]
"""Golden clips with speech (stt_golden's five first); the noise-only clips are left out."""
GOLDEN_TEXT_OVERRIDE = {"what_time_is_it_in_noise": "What time is it?"}
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Good morning! Your first meeting is with the design team.",
    "Paris is the capital of France.",
    "It is twenty past three in the afternoon.",
    "Sure, I can help with that. What would you like to know?",
    "Once upon a time, a little robot found a box of paints and began to draw the sky.",
]
VOICES = ["ryan", "eric"]
STT_RATE = 16000
TTS_RATE = 24000

CANDIDATES: dict[str, dict[str, Any]] = {
    # TTS candidates (STT: the current Parakeet default)
    "tts-1.7b-bf16": {"args": ["--tts", "1.7b", "--tts-quant", "BF16"], "measure": "tts"},
    "tts-1.7b-q8": {"args": ["--tts", "1.7b", "--tts-quant", "Q8_0"], "measure": "tts"},
    "tts-0.6b-bf16": {"args": ["--tts", "0.6b", "--tts-quant", "BF16"], "measure": "tts"},
    "tts-0.6b-q8": {"args": ["--tts", "0.6b", "--tts-quant", "Q8_0"], "measure": "tts"},
    # STT candidates (TTS: the current 1.7B BF16)
    "stt-parakeet-bf16": {"args": ["--stt", "parakeet", "--stt-dtype", "bf16"], "measure": "stt"},
    "stt-parakeet-fp16": {"args": ["--stt", "parakeet", "--stt-dtype", "fp16"], "measure": "stt"},
    "stt-parakeet-fp32": {"args": ["--stt", "parakeet", "--stt-dtype", "fp32"], "measure": "stt"},
    "stt-parakeet-cpu": {
        "args": ["--stt", "parakeet", "--stt-device", "cpu", "--stt-dtype", "fp32"],
        "measure": "stt",
    },
    "stt-moonshine-small": {"args": ["--stt", "moonshine-small"], "measure": "stt"},
    "stt-moonshine-medium": {"args": ["--stt", "moonshine-medium"], "measure": "stt"},
}


# ---------------------------------------------------------------- helpers


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("-", " ")).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return float(bool(h))
    row = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, row[0] = row[0], i
        for j, hw in enumerate(h, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (rw != hw))
    return row[-1] / len(r)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    errors = sum(wer(r, h) * len(words(r)) for r, h in pairs)
    return errors / max(1, sum(len(words(r)) for r, _ in pairs))


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes()), w.getframerate()


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def wav_bytes(pcm: bytes, rate: int) -> bytes:
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(url: str, wav: bytes) -> tuple[str, float]:
    request = urllib.request.Request(f"{url}/v1/audio/transcriptions", data=wav, method="POST")
    request.add_header("Content-Type", "audio/wav")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    return str(body["text"]), (time.monotonic() - started) * 1000


def synthesize(url: str, text: str, voice: str) -> tuple[bytes, float, float]:
    """(pcm, ms to the first audio byte, ms to the end) of one streamed TTS request."""
    body = json.dumps({"input": text, "voice": voice, "response_format": "pcm"}).encode()
    request = urllib.request.Request(f"{url}/v1/audio/speech", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    started = time.monotonic()
    first = None
    chunks = []
    with urllib.request.urlopen(request, timeout=300) as response:
        while chunk := response.read1(65536):
            if first is None:
                first = (time.monotonic() - started) * 1000
            chunks.append(chunk)
    return b"".join(chunks), first or 0.0, (time.monotonic() - started) * 1000


def to_16k(pcm: bytes, rate: int) -> bytes:
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n = round(len(audio) * STT_RATE / rate)
    out = np.interp(np.arange(n) * (rate / STT_RATE), np.arange(len(audio)), audio)
    return out.astype("<i2").tobytes()


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


class GpuSampler:
    """GPU memory of one process tree (nvidia-smi) and of the whole GPU, sampled in a thread."""

    def __init__(self, pids: set[int] | None = None) -> None:
        self.pids = pids or set()
        self.peak_proc = 0
        self.peak_total = 0
        self.last_proc = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self) -> tuple[int, int]:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        proc = 0
        for line in apps.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in self.pids:
                with contextlib.suppress(ValueError):
                    proc += int(parts[1])
        total = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        return proc, int(total[0]) if total else 0

    def _run(self) -> None:
        while not self._stop.is_set():
            proc, total = self.sample()
            self.last_proc = proc
            self.peak_proc = max(self.peak_proc, proc)
            self.peak_total = max(self.peak_total, total)
            self._stop.wait(0.2)

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def golden_items() -> list[tuple[str, bytes, str]]:
    texts = tomllib.loads((GOLDEN / "golden.toml").read_text())
    items = []
    for name in SPEECH_GOLDEN:
        text = GOLDEN_TEXT_OVERRIDE.get(name) or texts[name]["text"]
        pcm, rate = read_wav(GOLDEN / f"{name}.wav")
        items.append((f"golden/{name}", wav_bytes(pcm, rate), text))
    return items


def reference_items(out: Path) -> list[tuple[str, bytes, str]]:
    """The reference TTS clips (saved by the tts-1.7b-bf16 run), as 16 kHz STT input."""
    items = []
    for i, text in enumerate(SENTENCES):
        for voice in VOICES:
            path = out / "clips/tts-1.7b-bf16" / f"{voice}-{i}.wav"
            if path.exists():
                pcm, rate = read_wav(path)
                items.append((f"tts/{voice}-{i}", wav_bytes(to_16k(pcm, rate), STT_RATE), text))
    return items


# ---------------------------------------------------------------- measuring


def measure_stt(url: str, items: list[tuple[str, bytes, str]], repeats: int) -> dict[str, Any]:
    golden_lat, other_lat, golden_pairs, other_pairs, bad = [], [], [], [], []
    for _ in range(repeats):
        for name, wav, text in items:
            hyp, ms = transcribe(url, wav)
            (golden_lat if name.startswith("golden/") else other_lat).append(ms)
            pairs = golden_pairs if name.startswith("golden/") else other_pairs
            pairs.append((text, hyp))
            if wer(text, hyp) > 0.2:
                bad.append({"clip": name, "expected": text, "got": hyp})
    lat = golden_lat + other_lat
    return {
        "stt_ms_p50": statistics.median(lat),
        "stt_ms_p90": pct(lat, 0.9),
        "stt_ms_max": max(lat),
        "golden_ms_p50": statistics.median(golden_lat),
        "wer_golden": corpus_wer(golden_pairs),
        "wer_tts_clips": corpus_wer(other_pairs) if other_pairs else None,
        "requests": len(lat),
        "over_0_2": bad[: 3 * len(items)],
    }


def measure_tts(url: str, ref_stt_url: str, clips: Path, repeats: int) -> dict[str, Any]:
    ttfa, rtf, pairs = [], [], []
    durations: dict[tuple[int, str], list[float]] = {}
    for r in range(repeats):
        for i, text in enumerate(SENTENCES):
            for voice in VOICES:
                pcm, first_ms, total_ms = synthesize(url, text, voice)
                seconds = len(pcm) / 2 / TTS_RATE
                durations.setdefault((i, voice), []).append(seconds)
                ttfa.append(first_ms)
                rtf.append(total_ms / 1000 / max(seconds, 1e-3))
                if r == 0:
                    write_wav(clips / f"{voice}-{i}.wav", pcm, TTS_RATE)
                    hyp, _ = transcribe(ref_stt_url, wav_bytes(to_16k(pcm, TTS_RATE), STT_RATE))
                    pairs.append((text, hyp))
    return {
        "ttfa_ms_p50": statistics.median(ttfa),
        "ttfa_ms_p90": pct(ttfa, 0.9),
        "rtf_p50": statistics.median(rtf),
        "wer_roundtrip": corpus_wer(pairs),
        "requests": len(ttfa),
        **runaways(durations),
        "roundtrip": [{"expected": t, "got": h} for t, h in pairs if wer(t, h) > 0],
    }


def runaways(durations: dict[tuple[int, str], list[float]]) -> dict[str, Any]:
    """Generations far longer than the text needs: more than 1.6x the median of the same
    sentence and voice, or more than 0.12 s per character (a calm reading is about 0.07)."""
    found = []
    for (i, voice), values in durations.items():
        median = statistics.median(values)
        limit = min(1.6 * median, 0.12 * len(SENTENCES[i]) + 1.0)
        found += [f"{voice}-{i}: {v:.1f} s (median {median:.1f})" for v in values if v > limit]
    total = sum(len(v) for v in durations.values())
    return {"runaways": len(found), "runaway_of": total, "runaway_examples": found[:10]}


def run_candidate(
    name: str, spec: dict[str, Any], out: Path, port: int, repeats: int, python: str
) -> dict[str, Any]:
    log = out / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [python, "-m", "assistant_speech", "--port", str(port), *spec["args"]]
    print(f"== {name}: {' '.join(argv[1:])}", flush=True)
    started = time.monotonic()
    with log.open("w") as fh, GpuSampler() as gpu:
        proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        gpu.pids = {proc.pid}
        try:
            while "READY " not in log.read_text():
                if proc.poll() is not None or time.monotonic() - started > 600:
                    raise RuntimeError(f"{name} did not start:\n{log.read_text()[-3000:]}")
                time.sleep(0.5)
            ready_s = time.monotonic() - started
            time.sleep(1.0)
            after_load, _ = gpu.sample()
            url = f"http://127.0.0.1:{port}"
            health = json.loads(urllib.request.urlopen(f"{url}/health", timeout=10).read())
            if spec["measure"] == "stt":
                items = golden_items() + reference_items(out)
                result = measure_stt(url, items, repeats)
            else:
                result = measure_tts(url, url, out / "clips" / name, repeats)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(30)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    return {
        "name": name,
        "args": spec["args"],
        "ready_s": round(ready_s, 1),
        "vram_mib_loaded": after_load,
        "vram_mib_peak": gpu.peak_proc,
        "health": {
            k: health.get(k)
            for k in ("stt_model", "stt_device", "stt_dtype", "tts_model", "tts_quant", "gpu")
        },
        **result,
    }


def run_kokoro(out: Path, stt_url: str, repeats: int) -> dict[str, Any]:
    """Kokoro-82M in-process on the GPU (reference only: American male voices)."""
    import torch
    from kokoro import KPipeline

    voices = {"ryan": "am_michael", "eric": "am_adam"}
    with GpuSampler({os.getpid()}) as gpu:
        started = time.monotonic()
        pipeline = KPipeline(lang_code="a", device="cuda", repo_id="hexgrad/Kokoro-82M")
        for _ in pipeline("Ready.", voice="am_michael"):
            pass
        ready_s = time.monotonic() - started
        torch.cuda.synchronize()
        time.sleep(1.0)
        after_load, _ = gpu.sample()
        ttfa, rtf, pairs = [], [], []
        durations: dict[tuple[int, str], list[float]] = {}
        for r in range(repeats):
            for i, text in enumerate(SENTENCES):
                for voice, kvoice in voices.items():
                    t0 = time.monotonic()
                    first, parts = None, []
                    for result in pipeline(text, voice=kvoice):
                        audio = result.audio.detach().cpu().numpy()
                        if first is None:
                            first = (time.monotonic() - t0) * 1000
                        parts.append(audio)
                    total = time.monotonic() - t0
                    audio = np.concatenate(parts)
                    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                    ttfa.append(first or 0.0)
                    durations.setdefault((i, voice), []).append(len(audio) / TTS_RATE)
                    rtf.append(total / max(len(audio) / TTS_RATE, 1e-3))
                    if r == 0:
                        write_wav(out / "clips/tts-kokoro" / f"{kvoice}-{i}.wav", pcm, TTS_RATE)
                        hyp, _ = transcribe(stt_url, wav_bytes(to_16k(pcm, TTS_RATE), STT_RATE))
                        pairs.append((text, hyp))
    return {
        "name": "tts-kokoro-82m",
        "args": ["kokoro (in-process)", "voices am_michael/am_adam"],
        "ready_s": round(ready_s, 1),
        "vram_mib_loaded": after_load,
        "vram_mib_peak": gpu.peak_proc,
        "ttfa_ms_p50": statistics.median(ttfa),
        "ttfa_ms_p90": pct(ttfa, 0.9),
        "rtf_p50": statistics.median(rtf),
        "wer_roundtrip": corpus_wer(pairs),
        "requests": len(ttfa),
        **runaways(durations),
        "roundtrip": [{"expected": t, "got": h} for t, h in pairs if wer(t, h) > 0],
    }


def table(results: list[dict[str, Any]]) -> str:
    rows = [
        "| candidate | VRAM loaded / peak (MiB) | STT p50 / p90 (ms) | WER golden | WER TTS clips "
        "| TTS first audio p50 / p90 (ms) | TTS RTF | round-trip WER | runaways |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def f(v: Any, fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    for r in results:
        runaway = f"{r['runaways']}/{r['runaway_of']}" if "runaways" in r else "-"
        rows.append(
            f"| {r['name']} | {r['vram_mib_loaded']} / {r['vram_mib_peak']} "
            f"| {f(r.get('stt_ms_p50'), '.0f')} / {f(r.get('stt_ms_p90'), '.0f')} "
            f"| {f(r.get('wer_golden'), '.3f')} | {f(r.get('wer_tts_clips'), '.3f')} "
            f"| {f(r.get('ttfa_ms_p50'), '.0f')} / {f(r.get('ttfa_ms_p90'), '.0f')} "
            f"| {f(r.get('rtf_p50'), '.2f')} | {f(r.get('wer_roundtrip'), '.3f')} "
            f"| {runaway} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_speech.bakeoff")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", default=None, help="candidate names (default: all)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--kokoro", action="store_true", help="measure Kokoro-82M in-process")
    parser.add_argument("--stt-url", default="", help="reference STT for Kokoro's round trip")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    results: dict[str, Any] = json.loads(results_path.read_text()) if results_path.exists() else {}
    if args.kokoro:
        results["tts-kokoro-82m"] = run_kokoro(args.out, args.stt_url, args.repeats)
    else:
        names = args.only or list(CANDIDATES)
        # The reference TTS clips feed the STT candidates: make them first.
        names.sort(key=lambda n: (n != "tts-1.7b-bf16", CANDIDATES[n]["measure"] != "tts"))
        for name in names:
            try:
                results[name] = run_candidate(
                    name, CANDIDATES[name], args.out, args.port, args.repeats, sys.executable
                )
            except Exception as exc:  # one broken candidate must not lose the others
                print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)
                results[name] = {"name": name, "error": f"{type(exc).__name__}: {exc}"[:2000]}
            results_path.write_text(json.dumps(results, indent=2))
    results_path.write_text(json.dumps(results, indent=2))
    print(table([r for r in results.values() if "error" not in r]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
