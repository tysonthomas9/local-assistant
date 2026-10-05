"""Speech steps: the real speech server (Parakeet STT + Qwen3-TTS on its GPU) and spoken turns.

The speech server is `servers/speech` (`python -m assistant_speech`, its own venv
`servers/speech/.venv`), on loopback. `speech_server_running` uses the one already serving on
127.0.0.1:8772 (the gate starts it there) and otherwise starts one of the scenario's own on a
free port on the speech GPU (`[gpu]`: GPU1 in layout `two`, `gpu_index` in layout `one`);
`start_speech_server` always starts the scenario's own (so a fault can kill it for real).
Either way the server must run on a GPU: no GPU, no free memory on the speech GPU or no model
weights is a failure, never a skip. On one card (layout `one`) a second speech server next to
the stack's vLLM does not fit: such features fail with the GPU's processes named.

Voice input is the golden WAVs of `tests/fixtures/audio` (synthetic speech, see `golden.toml`)
fed as real 20 ms mic frames at the edge's mic input point (`/feed`); spoken replies are what
the edge's speaker was given (`--record-dir`), transcribed again by the real STT. Timings of a
turn are collected while the steps run and written by `timings_recorded` as a JSON artifact in
`$ASSISTANT_ARTIFACTS_DIR` (default `<repo>/artifacts`).
"""

import asyncio
import base64
import contextlib
import io
import json
import math
import os
import re
import signal
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
import wave
from array import array
from pathlib import Path
from typing import Any

from assistant_testing import llm_server
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.processes import RemoteProcess
from assistant_testing.steps import edge_host as edge_host_steps
from assistant_testing.steps.brain import LLM_MODEL, admin, ensure_llm_server
from assistant_testing.steps.link import SERVER, _client_name, _expect, _free_port, _get_lines

SPEECH = "speech"
SHARED_SPEECH_URL = "http://127.0.0.1:8772"
MIN_FREE_GPU_MIB = int(os.environ.get("ASSISTANT_SPEECH_MIN_FREE_MIB", "8000"))
"""What the speech server needs on its GPU (the default one uses about 6.8 GB; with
`ASSISTANT_SPEECH_ARGS="--tts 0.6b --tts-quant Q8_0"` about 4.1 GB: lower it to match)."""
READY_TIMEOUT_S = 300.0
GOLDEN_DIR = "tests/fixtures/audio"
SILENCE_DBFS = -45.0
"""A 20 ms frame quieter than this counts as silence."""


# ---------------------------------------------------------------- helpers


def _speech(ctx: ScenarioContext) -> dict[str, Any]:
    speech = ctx.state.get("speech")
    if speech is None:
        raise AssertionError("no speech server; use speech_server_running first")
    return speech


def speech_url(ctx: ScenarioContext) -> str:
    """The speech server of this scenario (for `start_brain: {speech: true}`)."""
    return _speech(ctx)["url"]


def _request(
    method: str,
    url: str,
    body: bytes | None = None,
    content_type: str = "application/json",
    timeout_s: float = 60.0,
) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        raise AssertionError(f"{method} {url}: {exc.code} {exc.read().decode()[:300]}") from exc


def _health(url: str, timeout_s: float = 2.0) -> dict[str, Any] | None:
    try:
        data, _ = _request("GET", f"{url}/health", timeout_s=timeout_s)
    except (OSError, AssertionError):
        return None
    with contextlib.suppress(json.JSONDecodeError):
        health = json.loads(data)
        if isinstance(health, dict) and health.get("ok"):
            return health
    return None


def _gpu_free_mib(index: str) -> int | None:
    try:
        done = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    with contextlib.suppress(ValueError):
        return int(done.stdout.strip().splitlines()[0])
    return None


def _gpu_apps(index: str) -> str:
    """The compute processes on GPU `index` (nvidia-smi numbering), e.g. `ollama/llama-server
    pid 123 13142 MiB`."""
    query = ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"]  # fmt: skip
    bus = ["nvidia-smi", f"--id={index}", "--query-gpu=pci.bus_id", "--format=csv,noheader"]
    try:
        apps = subprocess.run(query, capture_output=True, text=True, timeout=20, check=False)
        gpu = subprocess.run(bus, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    found = []
    for line in apps.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0] == gpu.stdout.strip():
            found.append(f"{parts[2]} pid {parts[1]} {parts[3]} MiB")
    return ", ".join(found)


def _check_health(health: dict[str, Any]) -> None:
    print(
        f"speech server: STT {health.get('stt_model')}, TTS {health.get('tts_model')}, "
        f"device {health.get('device')} ({health.get('gpu')}, CUDA_VISIBLE_DEVICES="
        f"{health.get('cuda_visible_devices')}), {health.get('gpu_memory_mib')} MiB on the GPU"
    )
    assert str(health.get("device", "")).startswith("cuda"), f"not on a GPU: {health}"


def _levels(samples: "array[int]", step: int) -> list[float]:
    """The level (dBFS) of each `step` samples."""
    levels: list[float] = []
    for offset in range(0, len(samples), step):
        chunk = samples[offset : offset + step]
        if not chunk:
            continue
        rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
        levels.append(20 * math.log10(rms / 32768.0) if rms > 0 else -120.0)
    return levels


def _wav(pcm: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


def _read_wav(data: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(data)) as wav:
        assert wav.getnchannels() == 1, "need mono audio"
        assert wav.getsampwidth() == 2, "need 16-bit audio"
        return wav.readframes(wav.getnframes()), wav.getframerate()


async def transcribe(ctx: ScenarioContext, wav: bytes) -> dict[str, Any]:
    """The speech server's transcript of a WAV: `{id, text, audio_ms, ms}`."""
    url = f"{speech_url(ctx)}/v1/audio/transcriptions"
    started = time.monotonic()
    data, _ = await asyncio.to_thread(_request, "POST", url, wav, "audio/wav")
    answer = json.loads(data)
    answer["round_trip_ms"] = round((time.monotonic() - started) * 1000, 1)
    return answer


def words(text: str) -> list[str]:
    """Lower-case words without punctuation (numbers kept), for the word error rate."""
    return re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'"))


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / len(ref)


def _golden(ctx: ScenarioContext, name: str) -> tuple[Path, str]:
    directory = ctx.repo_root / GOLDEN_DIR
    entries = tomllib.loads((directory / "golden.toml").read_text())
    assert name in entries, f"no golden WAV {name!r} in {GOLDEN_DIR}/golden.toml"
    return directory / f"{name}.wav", str(entries[name]["text"])


def _timings(ctx: ScenarioContext) -> dict[str, Any]:
    return ctx.state.setdefault("timings", {})


def _line_time(ctx: ScenarioContext, process: str, index: int) -> float:
    return ctx.processes.get(process).line_times[index]


# ---------------------------------------------------------------- the speech server


def speech_gpu() -> str:
    """The speech server's GPU (nvidia-smi numbering) in the configured `[gpu]` layout."""
    return str(llm_server.gpu_config().speech_gpu)


async def _start_own(ctx: ScenarioContext) -> dict[str, Any]:
    gpu = speech_gpu()
    free = _gpu_free_mib(gpu)
    assert free is not None, f"nvidia-smi cannot read GPU{gpu}: the speech server needs it"
    print(f"GPU{gpu}: {free} MiB free")
    if free < MIN_FREE_GPU_MIB:
        raise AssertionError(
            f"GPU{gpu} has only {free} MiB free (the speech server needs "
            f"{MIN_FREE_GPU_MIB}); on it: {_gpu_apps(gpu) or 'unknown'}"
        )
    root = ctx.repo_root
    venv = root / "servers/speech/.venv"
    env = {"UV_PROJECT_ENVIRONMENT": str(venv), "VIRTUAL_ENV": ""}
    if not (venv / "bin/python").exists():
        print("syncing servers/speech/.venv (uv sync --locked --project servers/speech)")
        done = await ctx.processes.run(
            "speech-sync",
            ["uv", "sync", "--locked", "--project", "servers/speech"],
            timeout_s=1800,
            env=env,
        )
        assert done.returncode == 0, f"uv sync of servers/speech failed:\n{done.output[-2000:]}"
    port = _free_port()
    # ASSISTANT_SPEECH_ARGS: extra server options (the gate's too), e.g. a smaller TTS on one card.
    extra = os.environ.get("ASSISTANT_SPEECH_ARGS", "").split()
    argv = [str(venv / "bin/python"), "-m", "assistant_speech", "--port", str(port), *extra]
    proc = await ctx.processes.start(
        SPEECH,
        argv,
        env={
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONUNBUFFERED": "1",
        },
        ready_line=r"^READY ",
        ready_timeout=READY_TIMEOUT_S,
    )
    ready = next(line for line in proc.lines if line.startswith("READY "))
    print(ready)
    url = f"http://127.0.0.1:{port}"
    health = await asyncio.to_thread(_health, url, 10)
    assert health is not None, f"the speech server at {url} printed READY but does not answer"
    _check_health(health)
    return {"url": url, "owned": True, "port": port, "health": health}


@step("speech_server_running")
async def speech_server_running(ctx: ScenarioContext) -> None:
    """A real speech server is serving on a GPU: the shared one on 127.0.0.1:8772 if it
    answers (the gate starts it), else one of the scenario's own on the speech GPU (stopped at
    the end). No GPU, a busy speech GPU or missing weights fail the step."""
    health = await asyncio.to_thread(_health, SHARED_SPEECH_URL)
    if health is not None:
        print(f"using the speech server at {SHARED_SPEECH_URL}")
        _check_health(health)
        ctx.state["speech"] = {"url": SHARED_SPEECH_URL, "owned": False, "health": health}
        return
    ctx.state["speech"] = await _start_own(ctx)


@step("start_speech_server")
async def start_speech_server(ctx: ScenarioContext) -> None:
    """Start a speech server of the scenario's own on the speech GPU and a free loopback port
    (it can be killed for real, `kill_speech_server`, without touching a shared one)."""
    ctx.state["speech"] = await _start_own(ctx)


async def _gone(pid: int, within_s: float) -> bool:
    deadline = time.monotonic() + within_s
    while Path(f"/proc/{pid}").exists():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.05)
    return True


@step("kill_speech_server")
async def kill_speech_server(ctx: ScenarioContext) -> None:
    """Kill the scenario's speech server (SIGKILL); it no longer answers and its GPU memory
    is released (its model worker dies with its supervisor)."""
    speech = _speech(ctx)
    assert speech.get("owned"), "the speech server is shared, not the scenario's: not killed"
    before = await asyncio.to_thread(_health, speech["url"])
    proc = ctx.processes.get(SPEECH)
    proc.send_signal(signal.SIGKILL)
    await proc.wait(15)
    if before is not None and before.get("pid"):
        worker = int(before["pid"])
        assert await _gone(worker, 15), f"the speech worker (pid {worker}) outlived its supervisor"
    health = await asyncio.to_thread(_health, speech["url"])
    assert health is None, f"the speech server at {speech['url']} still answers after the kill"
    print(f"speech server killed (exit {proc.proc.returncode})")


@step("crash_speech_worker")
async def crash_speech_worker(
    ctx: ScenarioContext, during: str = "tts", within_s: float = 60.0
) -> None:
    """Kill the scenario's speech model worker for real (SIGKILL: what a native crash such as
    GGML's CUDA abort does) while a TTS stream is sending audio (`during: tts`) or at once
    (`during: idle`). Its supervisor, the server process the scenario started, stays."""
    speech = _speech(ctx)
    assert speech.get("owned"), "the speech server is shared, not the scenario's: not crashed"
    health = await asyncio.to_thread(_health, speech["url"])
    assert health is not None, f"the speech server at {speech['url']} does not answer"
    supervisor = health.get("supervisor_pid")
    assert supervisor, "the speech server runs without its supervisor (--no-restart?)"
    worker = int(health["pid"])
    assert during in ("tts", "idle"), f"during: tts or idle, not {during!r}"
    if during == "tts":
        deadline = time.monotonic() + within_s
        while True:
            data, _ = await asyncio.to_thread(
                _request, "GET", f"{speech['url']}/requests", None, timeout_s=5
            )
            running = [
                r for r in json.loads(data)
                if r.get("kind") == "tts" and r.get("outcome") == "running" and "ttfa_ms" in r
            ]  # fmt: skip
            if running:
                print(f"TTS {running[-1]['id']} is streaming ({running[-1]['chars']} chars)")
                break
            assert time.monotonic() < deadline, f"no TTS stream within {within_s:.0f} s"
            await asyncio.sleep(0.02)
    os.kill(worker, signal.SIGKILL)
    assert await _gone(worker, 15), f"the speech worker (pid {worker}) survived SIGKILL"
    assert ctx.processes.get(SPEECH).proc.returncode is None, "the supervisor died too"
    speech["crashed_worker"] = {"pid": worker, "restarts": int(health.get("restarts", 0))}
    print(f"speech worker pid {worker} killed (supervisor pid {supervisor} still runs)")


@step("speech_server_restarted")
async def speech_server_restarted(ctx: ScenarioContext, within_s: float = 120.0) -> None:
    """After `crash_speech_worker`: the supervisor reported the death (WORKER-DIED) and a new
    worker answers on the same URL (a new pid, one more restart), with the same models."""
    speech = _speech(ctx)
    crashed = speech.get("crashed_worker")
    assert crashed is not None, "no worker crashed; use crash_speech_worker first"
    started = time.monotonic()
    deadline = started + within_s
    while True:
        health = await asyncio.to_thread(_health, speech["url"])
        if health is not None and health.get("pid") != crashed["pid"]:
            break
        assert time.monotonic() < deadline, f"no new speech worker within {within_s:.0f} s"
        await asyncio.sleep(0.5)
    assert int(health.get("restarts", 0)) == crashed["restarts"] + 1, f"restarts: {health}"
    _check_health(health)
    for key in ("stt_model", "tts_model", "tts_quant"):
        assert health[key] == speech["health"][key], f"{key}: {health[key]} after the restart"
    died = [x for x in ctx.processes.get(SPEECH).lines if x.startswith("WORKER-DIED ")]
    assert died, "the supervisor did not report the worker's death"
    print(died[-1])
    print(f"new worker pid {health['pid']} (restart {health['restarts']}) answers "
          f"{time.monotonic() - started:.1f} s after the crash was seen")  # fmt: skip


@step("stop_speech_server")
async def stop_speech_server(ctx: ScenarioContext) -> None:
    """Stop the scenario's speech server (SIGTERM, then it exits); it no longer answers."""
    speech = _speech(ctx)
    assert speech.get("owned"), "the speech server is shared, not the scenario's: not stopped"
    code = await ctx.processes.get(SPEECH).stop(grace_s=20)
    health = await asyncio.to_thread(_health, speech["url"])
    assert health is None, f"the speech server at {speech['url']} still answers after stopping"
    print(f"speech server stopped (exit {code})")


@step("restart_speech_server")
async def restart_speech_server(ctx: ScenarioContext) -> None:
    """Start the scenario's stopped or killed speech server again, on the same port."""
    speech = _speech(ctx)
    await ctx.processes.restart(SPEECH, ready_line=r"^READY ", ready_timeout=READY_TIMEOUT_S)
    health = await asyncio.to_thread(_health, speech["url"], 10)
    assert health is not None, f"the restarted speech server at {speech['url']} does not answer"
    _check_health(health)


@step("llm_serves")
async def llm_serves(
    ctx: ScenarioContext, server: str | None = None, model: str = LLM_MODEL
) -> None:
    """The stack's LLM server (`brain.ensure_llm_server`: the one on 127.0.0.1:8773, else the
    scenario's own) answers, is the configured kind (`server` if given must match it: vllm or
    ollama), serves `model` (default reachy-gemma4) and holds it entirely on the LLM's GPU
    (layout `two`: GPU0, so GPU1 stays the speech server's)."""
    url = await ensure_llm_server(ctx)
    kind = ctx.state["llm"]["server"]
    if server is not None:
        assert kind == server, f"the stack's LLM server is {kind}, want {server}"
    try:
        data, _ = await asyncio.to_thread(_request, "GET", f"{url}/v1/models", None)
    except OSError as exc:
        raise AssertionError(f"the LLM server at {url} does not answer: {exc}") from exc
    names = [m.get("id", "") for m in json.loads(data).get("data", [])]
    found = [n for n in names if n == model or n.split(":")[0] == model]
    assert found, f"the LLM server at {url} has no {model!r} (it has {names})"
    print(f"the LLM server at {url} ({kind}) serves {found[0]}")


# ---------------------------------------------------------------- STT and TTS directly


@step("transcribe_golden_wav")
async def transcribe_golden_wav(ctx: ScenarioContext, name: str) -> None:
    """The speech server transcribes the golden WAV `tests/fixtures/audio/<name>.wav`."""
    path, text = _golden(ctx, name)
    answer = await transcribe(ctx, path.read_bytes())
    ctx.state["transcript"] = {"text": answer["text"], "reference": text}
    _timings(ctx).setdefault("stt", []).append(
        {"wav": name, "audio_ms": answer.get("audio_ms"), "server_ms": answer.get("ms"),
         "round_trip_ms": answer["round_trip_ms"]}
    )  # fmt: skip
    print(
        f"STT of {name}.wav ({answer.get('audio_ms')} ms of audio): {answer['text']!r} "
        f"in {answer.get('ms')} ms (round trip {answer['round_trip_ms']} ms)"
    )


@step("transcript_matches")
async def transcript_matches(
    ctx: ScenarioContext, text: str | None = None, max_wer: float = 0.2
) -> None:
    """The last transcript matches `text` (default: what the golden WAV or the TTS said) with
    a word error rate of at most `max_wer` (lower case, no punctuation)."""
    transcript = ctx.state.get("transcript")
    assert transcript is not None, "nothing transcribed yet"
    reference = text if text is not None else transcript["reference"]
    wer = word_error_rate(reference, transcript["text"])
    print(f"WER {wer:.3f}: {transcript['text']!r} vs {reference!r}")
    assert wer <= max_wer, f"word error rate {wer:.3f} > {max_wer}"


@step("tts_gives_audio")
async def tts_gives_audio(
    ctx: ScenarioContext,
    text: str,
    voice: str = "ryan",
    min_s: float = 0.5,
    max_s: float = 30.0,
    min_voiced: float = 0.4,
) -> None:
    """The speech server speaks `text` in `voice`: between `min_s` and `max_s` seconds of
    audio, streamed (first audio before the end), and not silent: at least `min_voiced` of its
    20 ms frames are above -45 dBFS."""
    body = json.dumps({"input": text, "voice": voice, "response_format": "pcm"}).encode()
    url = f"{speech_url(ctx)}/v1/audio/speech"

    def fetch() -> tuple[bytes, int, float, float]:
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        started = time.monotonic()
        first: float | None = None
        chunks: list[bytes] = []
        with urllib.request.urlopen(request, timeout=120) as response:
            rate = int(response.headers.get("X-Sample-Rate", "24000"))
            while chunk := response.read1(65536):
                first = first if first is not None else time.monotonic() - started
                chunks.append(chunk)
        return b"".join(chunks), rate, (first or 0) * 1000, (time.monotonic() - started) * 1000

    pcm, rate, first_ms, total_ms = await asyncio.to_thread(fetch)
    seconds = len(pcm) / 2 / rate
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    levels = _levels(samples, rate // 50)
    voiced = sum(level > SILENCE_DBFS for level in levels) / max(1, len(levels))
    print(
        f"TTS {voice} {text!r}: {seconds:.2f} s at {rate} Hz, first audio {first_ms:.0f} ms, "
        f"all in {total_ms:.0f} ms, {voiced:.0%} voiced, peak {max(levels, default=-120):.1f} dBFS"
    )
    _timings(ctx).setdefault("tts", []).append(
        {"voice": voice, "chars": len(text), "audio_s": round(seconds, 2),
         "first_audio_ms": round(first_ms, 1), "total_ms": round(total_ms, 1)}
    )  # fmt: skip
    assert min_s <= seconds <= max_s, f"{seconds:.2f} s of audio, want {min_s}-{max_s} s"
    assert voiced >= min_voiced, f"only {voiced:.0%} of the audio is above {SILENCE_DBFS} dBFS"
    assert first_ms < total_ms, "the audio was not streamed"
    ctx.state["tts_audio"] = {"wav": _wav(pcm, rate), "text": text}


@step("transcribe_tts_audio")
async def transcribe_tts_audio(ctx: ScenarioContext) -> None:
    """The speech server transcribes the audio `tts_gives_audio` produced (a round trip)."""
    audio = ctx.state.get("tts_audio")
    assert audio is not None, "no TTS audio yet; use tts_gives_audio first"
    answer = await transcribe(ctx, audio["wav"])
    ctx.state["transcript"] = {"text": answer["text"], "reference": audio["text"]}
    print(f"STT of the TTS audio: {answer['text']!r} in {answer.get('ms')} ms")


@step("speech_server_used_voice")
async def speech_server_used_voice(ctx: ScenarioContext, voice: str) -> None:
    """The speech server's latest speech request (its `/requests` log) used `voice`."""
    data, _ = await asyncio.to_thread(_request, "GET", f"{speech_url(ctx)}/requests", None)
    spoken = [r for r in json.loads(data) if r.get("kind") == "tts"]
    assert spoken, "the speech server spoke nothing"
    latest = spoken[-1]
    print(f"latest TTS request: {json.dumps(latest)}")
    assert latest.get("voice") == voice, f"voice {latest.get('voice')!r}, want {voice!r}"


# ---------------------------------------------------------------- spoken turns at the edge


@step("feed_golden_wav")
async def feed_golden_wav(ctx: ScenarioContext, client: str, name: str) -> None:
    """Feed the golden WAV `<name>` into the edge agent as real 20 ms mic frames at its mic
    input point (`/feed`), in real time; waits until the agent started it (FEED)."""
    agent = _client_name(client)
    await ctx.processes.get(agent).write_line(f"/feed {GOLDEN_DIR}/{name}.wav")
    line = await _expect(ctx, agent, {"FEED", "CONSOLE-ERROR"}, 15, what="FEED")
    assert line.tag == "FEED", f"the agent did not feed {name}.wav: {line.text}"
    ctx.state["fed"] = {"name": name, "index": line.index, "wall": line.fields.get("wall")}
    print(line.text)


@step("room_level_measured")
async def room_level_measured(
    ctx: ScenarioContext,
    client: str,
    seconds: float = 2.0,
    key: str = "room_level",
    min_mean_dbfs: float | None = None,
) -> None:
    """Measure the room on the edge agent's real microphone for `seconds` (`/level`, before a
    feed) and put it in the timings (`room_level`, or `key`): the microphone must deliver
    frames; the level is recorded, not judged (an ordinary room may sit above the energy
    trigger), unless `min_mean_dbfs` asks for a room at least that loud (noise played)."""
    agent = _client_name(client)
    await ctx.processes.get(agent).write_line(f"/level {seconds}")
    line = await _expect(ctx, agent, {"LEVEL", "CONSOLE-ERROR"}, seconds + 15, what="LEVEL")
    assert line.tag == "LEVEL", f"the agent did not measure the room: {line.text}"
    frames = int(line.fields.get("frames", 0))
    expected = int(seconds * 1000 / 20)
    assert frames >= expected // 2, (
        f"the real microphone delivered {frames} frames in {seconds} s (want about {expected})"
    )
    mean, peak = float(line.fields["mean_dbfs"]), float(line.fields["max_dbfs"])
    trigger = (_timings(ctx).get("energy_trigger") or {}).get("dbfs")
    _timings(ctx)[key] = {
        "seconds": seconds,
        "frames": frames,
        "mean_dbfs": mean,
        "max_dbfs": peak,
        "above_energy_trigger": None if trigger is None else mean > trigger,
    }
    print(f"room level on the real microphone: mean {mean} dBFS, peak {peak} dBFS over "
          f"{seconds} s ({frames} frames); energy trigger {trigger} dBFS")  # fmt: skip
    if min_mean_dbfs is not None:
        assert mean >= min_mean_dbfs, f"the room is at {mean} dBFS (want >= {min_mean_dbfs})"


def _turn_input_time(ctx: ScenarioContext, client: str) -> tuple[float, str] | None:
    """When the user's input ended at the edge: the mic window closing after a fed utterance
    (end of speech or push-to-talk released) or the typed text.input (SENT), whichever came
    last (a scenario may type after a voice turn)."""
    agent = ctx.processes.get(_client_name(client))
    closes = [
        line
        for line in _get_lines(agent, "MIC-CLOSE")
        if line.index > ctx.state.get("fed", {}).get("index", 10**9)
    ]
    typed = [line for line in _get_lines(agent, "SENT") if line.fields.get("type") == "text.input"]
    if closes and not (typed and typed[-1].index > closes[-1].index):
        reason = closes[-1].fields.get("reason")
        return agent.line_times[closes[-1].index], f"end of speech (MIC-CLOSE {reason})"
    if typed:
        return agent.line_times[typed[-1].index], "typed text (SENT text.input)"
    return None


@step("golden_wav_fed")
async def golden_wav_fed(ctx: ScenarioContext, client: str, within_s: float = 30.0) -> None:
    """The golden WAV being fed reached its end (FED)."""
    line = await _expect(ctx, _client_name(client), {"FED"}, within_s, what="FED")
    print(line.text)


@step("playback_progress_at_least")
async def playback_progress_at_least(
    ctx: ScenarioContext, client: str, ms: int, within_s: float = 60.0
) -> None:
    """The reply being spoken (`robot_speaks_reply`) has played at least `ms` on the edge's
    speaker (its playback clock's `progress`)."""
    speaking = ctx.state.get("speaking")
    assert speaking is not None, "no reply spoken yet; use robot_speaks_reply first"
    stream = str(speaking["stream"])
    agent = _client_name(client)
    deadline = time.monotonic() + within_s
    while True:
        line = await _expect(
            ctx, agent, {"PLAYBACK"}, max(0.1, deadline - time.monotonic()),
            fields={"stream": stream}, what=f"playback of stream {stream} reaching {ms} ms",
        )  # fmt: skip
        state = line.fields.get("state")
        if state not in ("progress", "started"):
            proc = ctx.processes.get(agent)
            why = [x.text for tag in ("MIC-OPEN", "BARGE-IN") for x in _get_lines(proc, tag)]
            raise AssertionError(f"stream {stream} ended early: {line.text}; " + "; ".join(why))
        if int(line.fields.get("played_ms", "0")) >= ms:
            print(line.text)
            return


@step("robot_speaks_reply")
async def robot_speaks_reply(
    ctx: ScenarioContext,
    client: str,
    min_ms: int = 300,
    within_s: float = 120.0,
    wait_done: bool = True,
) -> None:
    """The brain's reply reaches the edge as speech: `speak.begin` (with its first sentence)
    then 0x02 audio played by the edge's speaker, its playback clock reporting `started`,
    `progress` and (with `wait_done`) `done` with at least `min_ms` played. Records the time
    from the user's input to the speaker's first audio."""
    agent = _client_name(client)
    # The next stream that starts playing (a text-only stream, e.g. an apology, never does).
    started = await _expect(
        ctx, agent, {"PLAYBACK"}, within_s, fields={"state": "started"}, what="playback started"
    )
    stream = started.fields["stream"]
    begins = [
        line
        for line in _get_lines(ctx.processes.get(agent), "RECV")
        if line.fields.get("type") == "speak.begin"
        and str((line.payload or {}).get("stream_id")) == stream
        and line.index < started.index
    ]
    assert begins, f"stream {stream} played without a speak.begin"
    begin = begins[-1]
    ctx.state["link"].consumed.setdefault(agent, set()).add(begin.index)
    text = (begin.payload or {}).get("text")
    ctx.state["speaking"] = {"stream": int(stream), "begin": begin.index, "text": text}
    print(f"speak.begin stream {stream}: {text!r}")
    first_audio = _line_time(ctx, agent, started.index)
    timings = _timings(ctx)
    source = _turn_input_time(ctx, client)
    if source is not None:
        ms = (first_audio - source[0]) * 1000
        timings["input_to_first_audio_ms"] = round(ms, 1)
        timings["input_to_first_audio_from"] = source[1]
        print(f"{source[1]} -> the speaker's first audio: {ms:.0f} ms (seen from this PC)")
    await _expect(
        ctx, agent, {"PLAYBACK"}, 30, fields={"stream": stream, "state": "progress"},
        what=f"playback progress (stream {stream})",
    )  # fmt: skip
    if not wait_done:
        return
    done = await _expect(
        ctx, agent, {"PLAYBACK"}, within_s, fields={"stream": stream, "state": "done"},
        what=f"playback done (stream {stream})",
    )  # fmt: skip
    played = int(done.fields.get("played_ms", "0"))
    took = (_line_time(ctx, agent, done.index) - first_audio) * 1000
    timings["reply_played_ms"] = played
    print(f"stream {stream}: {played} ms played, started -> done in {took:.0f} ms")
    assert played >= min_ms, f"only {played} ms of the reply played (want >= {min_ms})"


async def _fetch_recording(ctx: ScenarioContext, client: str, path: str) -> bytes:
    agent = ctx.processes.get(_client_name(client))
    if isinstance(agent, RemoteProcess):
        host = edge_host_steps.host_of(ctx)
        quoted = path.replace("'", "")
        script = (
            f"cd {agent.remote_cwd or '.'} && base64 < '{quoted}' && rm -f '{quoted}' "
            f"&& rmdir \"$(dirname '{quoted}')\" .recordings 2>/dev/null; true"
        )
        done = await asyncio.to_thread(host.run, script, 60)
        assert done.returncode == 0, f"cannot read {path} on {host.label}: {done.stderr[-300:]}"
        assert done.stdout.strip(), f"{path} on {host.label} is empty or missing"
        return base64.b64decode(done.stdout)
    local = ctx.repo_root / path
    data = local.read_bytes()
    local.unlink()
    for directory in (local.parent, local.parent.parent):
        with contextlib.suppress(OSError):
            directory.rmdir()  # only if empty
    return data


@step("recorded_reply_transcript_not_empty")
async def recorded_reply_transcript_not_empty(
    ctx: ScenarioContext, client: str, within_s: float = 120.0
) -> None:
    """What the edge's speaker was given for the reply (the agent's `--record-dir` WAV of the
    stream) is real speech: the speech server transcribes it to non-empty text (printed with
    its word error rate against the reply text the brain logged)."""
    speaking = ctx.state.get("speaking")
    assert speaking is not None, "no reply spoken yet; use robot_speaks_reply first"
    stream = str(speaking["stream"])
    line = await _expect(
        ctx, _client_name(client), {"RECORDED"}, within_s, fields={"stream": stream},
        what=f"RECORDED stream {stream} (start the agent with record_dir)",
    )  # fmt: skip
    data = await _fetch_recording(ctx, client, line.fields["path"])
    pcm, rate = _read_wav(data)
    answer = await transcribe(ctx, _wav(pcm, rate))
    turns = await admin(ctx, "/turns")
    reply = next((t["reply_text"] for t in reversed(turns) if t.get("reply_text")), "")
    wer = word_error_rate(reply, answer["text"]) if reply else float("nan")
    print(
        f"recorded reply ({len(pcm) * 1000 // (2 * rate)} ms at {rate} Hz) transcribed: "
        f"{answer['text']!r}; the brain's reply text: {reply!r} (WER {wer:.2f})"
    )
    assert answer["text"].strip(), "the recorded reply transcribes to nothing"


def _without_wake(text: str, wake: str) -> str:
    """`text` without a leading wake word (all its words, or its last ones) and the punctuation
    after it, compared as lower-case letters and digits."""

    def norm(word: str) -> str:
        return "".join(ch for ch in word.lower() if ch.isalnum())

    words, wanted = text.split(), [norm(w) for w in wake.split()]
    for start in range(len(wanted)):
        tail = wanted[start:]
        if [norm(w) for w in words[: len(tail)]] == tail:
            return " ".join(words[len(tail) :]).lstrip(" ,.;:!?-")
    return text.strip()


@step("voice_turn_transcribed")
async def voice_turn_transcribed(
    ctx: ScenarioContext,
    client: str,
    text: str | None = None,
    max_wer: float = 0.2,
    within_s: float = 60.0,
) -> None:
    """The brain transcribed the fed utterance (its TRANSCRIPT line): the golden WAV's text
    (or `text`) with a word error rate of at most `max_wer`. Prints the edge's mic windows.
    After a wake (`wake_detected`) the transcript, what the LLM gets, must not start with the
    wake word (or its last word), and the reference is the golden text without it."""
    line = await _expect(ctx, SERVER, {"TRANSCRIPT"}, within_s, what="TRANSCRIPT")
    agent = ctx.processes.get(_client_name(client))
    for window in _get_lines(agent, "MIC-OPEN") + _get_lines(agent, "MIC-CLOSE"):
        print(f"  edge: {window.text}")
    heard = str((line.payload or {}).get("text") or "")
    fed = ctx.state.get("fed")
    reference = text if text is not None else (_golden(ctx, fed["name"])[1] if fed else "")
    wake = ctx.state.pop("wake_word", None)
    if wake is not None:
        reference = _without_wake(reference, wake)
        assert _without_wake(heard, wake) == heard.strip(), (
            f"the wake word {wake!r} reached the brain's transcript: {heard!r}"
        )
    wer = word_error_rate(reference, heard)
    print(f"the brain heard {heard!r} (WER {wer:.3f} against {reference!r})")
    # The real microphone is live until the feed starts: a room sound above the energy trigger
    # may open (and fill) a mic window before the golden WAV arrives.
    early = [w.text for w in _get_lines(agent, "MIC-OPEN") if fed and w.index < fed["index"]]
    assert wer <= max_wer, f"word error rate {wer:.3f} > {max_wer}" + (
        f"; the real microphone opened a mic window before the feed: {early}" if early else ""
    )


@step("barge_in_stops_speech")
async def barge_in_stops_speech(
    ctx: ScenarioContext, client: str, min_ms: float = 250, max_ms: float = 500
) -> None:
    """A barge-in stopped the edge's speech the way a person stops talking: the edge sent
    `vad{start, barge_in, stream_id, played_ms}` for the playing stream at once (BARGE-IN), the
    speech went on and then stopped with a fade between `min_ms` and `max_ms` after the
    barge-in (FLUSHED local=true stop_ms, on the edge's playback clock), at the `played_ms` it
    told the brain."""
    agent = _client_name(client)
    barge = await _expect(ctx, agent, {"BARGE-IN"}, 15, what="BARGE-IN")
    print(barge.text)
    vad = await _expect(
        ctx, agent, {"SENT"}, 10, fields={"type": "vad"}, payload={"barge_in": True},
        what="vad barge_in",
    )  # fmt: skip
    payload = vad.payload or {}
    flushed = await _expect(
        ctx, agent, {"FLUSHED"}, 10, fields={"local": "true"}, what="FLUSHED local=true"
    )
    print(flushed.text)
    stop_ms = float(flushed.fields.get("stop_ms", flushed.fields.get("took_ms", "0")))
    speaking = ctx.state.get("speaking") or {}
    ctx.state["barge_in"] = {"played_ms": payload.get("played_ms"), "stop_ms": stop_ms,
                             "wall": float(barge.fields.get("wall", "0"))}  # fmt: skip
    timings = _timings(ctx)
    timings["barge_in_stop_ms"] = stop_ms
    timings["barge_in_played_ms"] = payload.get("played_ms")
    timings["barge_in_fade_ms"] = int(barge.fields.get("fade_ms", "0"))
    print(f"barge-in: the speech stopped {stop_ms:.0f} ms after it, at {payload.get('played_ms')}"
          f" ms played (fade {barge.fields.get('fade_ms')} ms)")  # fmt: skip
    if speaking:
        assert payload.get("stream_id") == speaking["stream"], f"barge-in on {payload}"
    assert int(flushed.fields["played_ms"]) == payload.get("played_ms"), (
        f"stopped at {flushed.fields['played_ms']} ms, told the brain {payload.get('played_ms')}"
    )
    assert min_ms <= stop_ms <= max_ms, (
        f"the speech stopped {stop_ms:.0f} ms after the barge-in (want {min_ms}-{max_ms} ms)"
    )


@step("turn_truncated_at_played_ms")
async def turn_truncated_at_played_ms(ctx: ScenarioContext, within_s: float = 15.0) -> None:
    """The interrupted turn's log entry shows the reply cut where the edge stopped: `truncated`
    with the edge's `played_ms`, the heard text a strict beginning of what was spoken (whole
    sentences, then the words of the cut one in proportion to how much of it played), and
    less audio heard than was sent."""
    barge = ctx.state.get("barge_in")
    assert barge is not None, "no barge-in yet; use barge_in_stops_speech first"
    deadline = time.monotonic() + within_s
    turn: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        turns = await admin(ctx, "/turns")
        turn = next((t for t in reversed(turns) if t.get("outcome") == "interrupted"), None)
        if turn is not None:
            break
        await asyncio.sleep(0.2)
    assert turn is not None, "no interrupted turn in the turn log"
    cut = turn.get("truncated")
    print(f"interrupted turn {turn['turn_id']}: truncated {json.dumps(cut)}")
    assert cut, f"the interrupted turn logs no truncation: {json.dumps(turn)}"
    assert cut["played_ms"] == barge["played_ms"], (
        f"truncated at {cut['played_ms']} ms, the edge stopped at {barge['played_ms']} ms"
    )
    assert cut["played_ms"] < cut["sent_audio_ms"], f"all the sent audio played: {cut}"
    heard, spoken = words(str(cut["heard_text"])), words(str(cut["spoken_text"]))
    assert len(heard) < len(spoken), f"nothing was cut: {cut}"
    assert spoken[: len(heard)] == heard, f"the heard text is not the start of the reply: {cut}"
    line = await _expect(ctx, SERVER, {"TURN-TRUNCATED"}, 5, what="TURN-TRUNCATED")
    print(line.text)


@step("timings_recorded")
async def timings_recorded(ctx: ScenarioContext, name: str) -> None:
    """Write the scenario's timings as `<artifacts>/timings-<name>.json` (`-sim.json` on the
    simulated robot, so it never replaces the physical robot's), with the latest
    finished turn's from the turn log (STT, LLM first token, TTS first audio, the reply's
    first audio, total), and print them. `$ASSISTANT_ARTIFACTS_DIR` (default
    `<repo>/artifacts`) is the directory."""
    timings = dict(_timings(ctx))
    if "brain" in ctx.state:
        turns = await admin(ctx, "/turns")
        done = [t for t in turns if t.get("outcome") in ("finished", "interrupted")]
        timings["turns"] = [
            {"turn_id": t["turn_id"], "kind": t["kind"], "outcome": t["outcome"],
             "input": t.get("input_text"), "stt_ms": t["speech"].get("stt_ms"),
             "llm_ttft_ms": t["llm"].get("ttft_ms"), "llm_total_ms": t["llm"].get("total_ms"),
             "tts_first_audio_ms": t["speech"].get("tts_first_audio_ms"),
             "first_audio_ms": t["speech"].get("first_audio_ms"),
             "voice": t["speech"].get("voice"), "total_ms": t.get("total_ms"),
             "reply_audio_ms": t.get("reply_audio_ms")}
            for t in done
        ]  # fmt: skip
    llm = (ctx.state.get("brain") or {}).get("llm")
    if llm is not None:
        gpu = await asyncio.to_thread(llm_server.server_gpu_mib, llm["url"])
        timings["llm_server"] = {"server": llm["server"], "url": llm["url"], "gpu_mib": gpu}
    speech = ctx.state.get("speech")
    if speech is not None:
        health = await asyncio.to_thread(_health, speech["url"])
        if health is not None:
            timings["speech_server"] = {
                k: health.get(k) for k in ("gpu", "gpu_memory_mib", "stt_model", "tts_model")
            }
    timings["feature"] = ctx.feature_path.stem
    if ctx.robot is not None:
        timings["robot"] = ctx.robot
    directory = Path(os.environ.get("ASSISTANT_ARTIFACTS_DIR") or ctx.repo_root / "artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"timings-{name}{'-sim' if ctx.sim else ''}.json"
    path.write_text(json.dumps(timings, indent=2) + "\n")
    print(f"timings -> {path.name}:\n{json.dumps(timings, indent=2)}")
    turns_logged = timings.get("turns") or []
    if "brain" in ctx.state:
        assert turns_logged, "no finished turn to record timings of"
