"""The minimal speech server (skeleton task S8; multi-session arrives in phase 2).

    CUDA_VISIBLE_DEVICES=1 python -m assistant_speech --port 8772

One process holds both models on one GPU (GPU1 on the brain PC, chosen by the caller with
`CUDA_VISIBLE_DEVICES`):

- STT: NVIDIA Parakeet TDT 0.6B v3 (`nano-parakeet`, PyTorch), 16 kHz mono; or (`--stt`)
  Moonshine v2 streaming small/medium (transformers, English only). `--stt-device cpu` keeps
  the STT off the GPU.
- TTS: Qwen3-TTS 12 Hz CustomVoice, 1.7B (default) or 0.6B (`--tts`), at `--tts-quant`
  (`faster-qwen3-tts`, GGML backend), 24 kHz mono, with its preset voices (`ryan`, `eric`,
  `aiden`, ...).

Weights come from the local Hugging Face cache; nothing is downloaded unless `--online` is
given. It binds to loopback only. HTTP API (OpenAI-shaped where that is simple):

    GET  /health                    {"ok", "stt_model", "tts_model", "voices", "rate", "gpu", ...}
    POST /v1/audio/transcriptions   body: a WAV file (Content-Type audio/wav), or raw s16le mono
                                    PCM (audio/pcm or application/octet-stream, `?rate=16000`)
                                    -> {"id", "text", "audio_ms", "ms"}
    POST /v1/audio/speech           {"input": text, "voice": "ryan", "language"?: "english"}
                                    -> streamed raw s16le mono PCM at 24 kHz
                                       (headers X-Sample-Rate, X-Request-Id)
    GET  /requests                  the recent request log (kind, voice, timings, outcome)

It serves one session: model work runs on one worker thread, in arrival order. A client that
goes away mid-stream stops its TTS at the next chunk. The first line on stdout is
`READY url=http://127.0.0.1:<port> ...` once both models are loaded and warmed up; `STOPPED`
is the last.

The models run in a worker process under a small supervisor (this process, which binds the
port and never loads a model). A native crash in a model (GGML aborts the process on a CUDA
error) kills only the worker: the request in flight fails (its stream ends incomplete), the
supervisor prints `WORKER-DIED ...` and starts a new worker on the same socket, which prints
`READY` again (`restarts=1`, and `/health` `pid`, `supervisor_pid`, `restarts`). Requests made
meanwhile wait in the socket's backlog (about 10-15 s on a GPU) instead of being refused. A
worker that dies before READY is not restarted (the supervisor exits with its code), and more
than `MAX_RESTARTS` deaths in `RESTART_WINDOW_S` end the supervisor (`GIVING-UP`). SIGTERM
and SIGINT go to the worker; a killed supervisor takes the worker with it (Linux
PR_SET_PDEATHSIG). `--no-restart` loads the models in this one process, as before.
"""

import argparse
import asyncio
import contextlib
import io
import os
import socket
import sys
import threading
import time
import uuid
import wave
from collections import deque
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

STT_MODELS = {
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "moonshine-small": "moonshine-ai/moonshine-streaming-small",
    "moonshine-medium": "moonshine-ai/moonshine-streaming-medium",
}
"""`--stt` choices. Moonshine v2 streaming is English only (transformers, already a dependency)."""
TTS_MODELS = {
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}
"""`--tts` choices: the same Qwen3-TTS CustomVoice family and preset voices, two sizes."""
STT_DTYPES = ("auto", "bf16", "fp16", "fp32")
STT_RATE = 16000
TTS_RATE = 24000
TTS_CHUNK_FRAMES = 8
"""Codec frames per streamed TTS chunk (12.5 frames/s: about 0.64 s of audio per chunk)."""
TTS_TOKENS_PER_S = 12.5
REQUEST_LOG_SIZE = 200
MAX_RESTARTS = 5
RESTART_WINDOW_S = 600.0
"""The supervisor gives up after more than MAX_RESTARTS worker deaths in RESTART_WINDOW_S."""
MAX_TEXT_CHARS = 2000


def line(tag: str, **fields: object) -> None:
    sys.stdout.write(" ".join([tag, *(f"{k}={v}" for k, v in fields.items())]) + "\n")
    sys.stdout.flush()


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resampling (enough for speech recognition)."""
    if src == dst or not len(audio):
        return audio
    n = round(len(audio) * dst / src)
    return np.interp(np.arange(n) * (src / dst), np.arange(len(audio)), audio).astype(np.float32)


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """A 16-bit PCM WAV (mono, or stereo mixed down) as float32 and its rate."""
    with wave.open(io.BytesIO(data)) as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        rate, channels = wav.getframerate(), wav.getnchannels()
        audio = pcm16_to_float(wav.readframes(wav.getnframes()))
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


class Models:
    """The STT and the TTS model, loaded once. Only the worker thread calls into them."""

    def __init__(
        self,
        device: str,
        tts_quant: str,
        stt: str = "parakeet",
        tts: str = "1.7b",
        stt_device: str | None = None,
        stt_dtype: str = "auto",
    ) -> None:
        import torch
        from faster_qwen3_tts.ggml_backend import GGMLQwen3TTS

        stt_device = stt_device or device
        for dev in (device, stt_device):
            if dev.startswith("cuda") and not torch.cuda.is_available():
                raise SystemExit(
                    "CUDA is not available (check CUDA_VISIBLE_DEVICES and the driver)"
                )
        self.device = device
        self.stt_device = stt_device
        self.gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.stt_name = STT_MODELS[stt]
        self.tts_name = TTS_MODELS[tts]
        self.tts_quant = tts_quant
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtypes.get(stt_dtype)
        if stt == "parakeet":
            from nano_parakeet import from_pretrained

            self.stt = from_pretrained(model_name=self.stt_name, device=stt_device, dtype=dtype)
            self.moonshine = None
        else:
            from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

            if dtype is None:
                dtype = torch.float16 if stt_device.startswith("cuda") else torch.float32
            model = MoonshineStreamingForConditionalGeneration.from_pretrained(self.stt_name)
            self.moonshine = (
                model.to(stt_device).to(dtype).eval(),
                AutoProcessor.from_pretrained(self.stt_name),
                dtype,
            )
        self.stt_dtype = str(dtype or "auto").removeprefix("torch.")
        self.tts = GGMLQwen3TTS.from_pretrained(
            self.tts_name, quant=tts_quant, local_files_only=True
        )
        self.voices = sorted(v.lower() for v in self.tts.get_supported_speakers())

    def transcribe(self, audio: np.ndarray) -> str:
        if self.moonshine is not None:
            return self._moonshine(audio)
        out: Any = self.stt.transcribe(audio)
        if isinstance(out, str):
            text = out
        else:
            text = getattr(out, "text", None) or (out[0] if out else "")
        return str(text).strip()

    def _moonshine(self, audio: np.ndarray) -> str:
        import torch

        model, processor, dtype = self.moonshine  # type: ignore[misc]
        inputs = processor(audio, return_tensors="pt", sampling_rate=STT_RATE)
        inputs = inputs.to(self.stt_device, dtype)
        # About 6.5 tokens per second of audio is the model card's limit for a transcript.
        max_length = max(8, int(inputs.attention_mask.sum().item() * 6.5 / STT_RATE))
        with torch.inference_mode():
            ids = model.generate(**inputs, max_length=max_length)
        return str(processor.decode(ids[0], skip_special_tokens=True)).strip()

    def speak(
        self, text: str, voice: str, language: str, stop: threading.Event
    ) -> Iterator[np.ndarray]:
        # Enough codec tokens for the text at a slow speaking rate, plus a margin; a runaway
        # generation cannot go on for minutes.
        seconds = len(text) / 10.0 + 2.0
        max_tokens = int(min(2048, seconds * TTS_TOKENS_PER_S * 1.5 + 24))
        for chunk, sr, _info in self.tts.generate_custom_voice_streaming(
            text=text,
            speaker=voice,
            language=language,
            chunk_size=TTS_CHUNK_FRAMES,
            max_new_tokens=max_tokens,
        ):
            if stop.is_set():
                return
            audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
            yield resample(audio, int(sr), TTS_RATE)

    def gpu_memory_mib(self) -> int | None:
        """PyTorch's share only (the STT); the TTS (GGML) is not counted: see nvidia-smi."""
        if not self.stt_device.startswith("cuda"):
            return None
        import torch

        return int(torch.cuda.memory_reserved() // (1024 * 1024))


class SpeechService:
    def __init__(self, models: Models, language: str) -> None:
        self.models = models
        self.language = language
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech")
        self.requests: deque[dict[str, Any]] = deque(maxlen=REQUEST_LOG_SIZE)

    async def transcribe(self, audio: np.ndarray, rate: int) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": f"stt-{uuid.uuid4().hex[:10]}",
            "kind": "stt",
            "audio_ms": int(len(audio) * 1000 / rate),
            "outcome": "running",
        }
        self.requests.append(entry)
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(
                self.worker, self.models.transcribe, resample(audio, rate, STT_RATE)
            )
        except Exception as exc:
            entry |= {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"}
            raise
        entry |= {"outcome": "ok", "ms": round((time.monotonic() - started) * 1000, 1)}
        entry["chars"] = len(text)
        return {"id": entry["id"], "text": text, "audio_ms": entry["audio_ms"], "ms": entry["ms"]}

    async def speak(self, text: str, voice: str, entry: dict[str, Any]) -> AsyncIterator[bytes]:
        """Stream the TTS audio as s16le PCM chunks; stops the model when the caller goes."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        stop = threading.Event()

        def put(kind: str, item: Any) -> None:
            with contextlib.suppress(RuntimeError):  # the loop is gone at shutdown
                loop.call_soon_threadsafe(queue.put_nowait, (kind, item))

        def work() -> None:
            try:
                for chunk in self.models.speak(text, voice, self.language, stop):
                    put("chunk", chunk)
                put("end", None)
            except BaseException as exc:
                put("error", exc)

        started = time.monotonic()
        loop.run_in_executor(self.worker, work)
        samples = 0
        entry["outcome"] = "running"
        try:
            while True:
                kind, item = await queue.get()
                if kind == "error":
                    entry |= {"outcome": "error", "error": f"{type(item).__name__}: {item}"}
                    return
                if kind == "end":
                    entry["outcome"] = "ok"
                    return
                if samples == 0:
                    entry["ttfa_ms"] = round((time.monotonic() - started) * 1000, 1)
                samples += len(item)
                yield float_to_pcm16(item)
        finally:
            stop.set()
            if entry["outcome"] == "running":
                entry["outcome"] = "cancelled"
            entry["audio_ms"] = int(samples * 1000 / TTS_RATE)
            entry["ms"] = round((time.monotonic() - started) * 1000, 1)


def build_app(service: SpeechService, port_ref: dict[str, int]) -> Any:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="assistant-speech", docs_url=None, redoc_url=None, openapi_url=None)
    models = service.models

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "stt_model": models.stt_name,
            "stt_device": models.stt_device,
            "stt_dtype": models.stt_dtype,
            "tts_model": models.tts_name,
            "tts_quant": models.tts_quant,
            "voices": models.voices,
            "rate": TTS_RATE,
            "language": service.language,
            "device": models.device,
            "gpu": models.gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_memory_mib": models.gpu_memory_mib(),
            "pid": os.getpid(),
            "supervisor_pid": int(os.environ.get("ASSISTANT_SPEECH_SUPERVISOR", "0")) or None,
            "restarts": int(os.environ.get("ASSISTANT_SPEECH_RESTARTS", "0")),
            "port": port_ref["port"],
        }

    @app.get("/requests")
    async def requests() -> list[dict[str, Any]]:
        return list(service.requests)

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request) -> dict[str, Any]:
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty body: send a WAV file or raw s16le PCM")
        kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
        try:
            if kind in ("audio/wav", "audio/x-wav", "audio/wave"):
                audio, rate = read_wav(body)
            else:
                rate = int(request.query_params.get("rate", STT_RATE))
                audio = pcm16_to_float(body)
        except (ValueError, wave.Error, EOFError) as exc:
            raise HTTPException(400, f"bad audio: {exc}") from exc
        if not 8000 <= rate <= 48000:
            raise HTTPException(400, f"unsupported rate {rate}")
        return await service.transcribe(audio, rate)

    @app.post("/v1/audio/speech")
    async def speech(request: Request) -> StreamingResponse:
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(400, f"bad JSON: {exc}") from exc
        text = str(payload.get("input") or "").strip()
        voice = str(payload.get("voice") or "").strip().lower()
        if not text or len(text) > MAX_TEXT_CHARS:
            raise HTTPException(400, f"input must be 1-{MAX_TEXT_CHARS} characters")
        if voice not in models.voices:
            raise HTTPException(400, f"unknown voice {voice!r}; voices: {', '.join(models.voices)}")
        if payload.get("response_format", "pcm") != "pcm":
            raise HTTPException(400, "only response_format pcm (s16le, 24 kHz mono) is supported")
        entry: dict[str, Any] = {
            "id": f"tts-{uuid.uuid4().hex[:10]}",
            "kind": "tts",
            "voice": voice,
            "chars": len(text),
        }
        service.requests.append(entry)
        return StreamingResponse(
            service.speak(text, voice, entry),
            media_type="audio/pcm",
            headers={"X-Sample-Rate": str(TTS_RATE), "X-Request-Id": entry["id"]},
        )

    return app


def warm_up(models: Models) -> float:
    """One TTS sentence and its transcription, so the first real request is fast."""
    started = time.monotonic()
    stop = threading.Event()
    audio = np.concatenate(list(models.speak("Ready.", "ryan", "english", stop)))
    models.transcribe(resample(audio, TTS_RATE, STT_RATE))
    return time.monotonic() - started


def _die_with_parent() -> None:
    """In the worker, before exec: SIGKILL when the supervisor dies (Linux), so a killed
    supervisor never leaves the models on the GPU."""
    import ctypes
    import signal

    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid() == 1:  # the supervisor was already gone
        os._exit(1)


def supervise(sock: socket.socket, argv: list[str]) -> int:
    """Run the models in a worker process on the supervisor's listening socket, and start a
    new worker when one dies after it was ready (a native TTS crash takes the whole process
    with it: it cannot be caught in Python). Requests made meanwhile wait in the socket's
    backlog for the new worker. A worker that dies before READY is not restarted (bad options,
    no GPU, no weights): its exit code is the supervisor's. SIGTERM/SIGINT go to the worker."""
    import signal
    import subprocess

    worker: subprocess.Popen[bytes] | None = None
    stopping = False

    def forward(signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        if worker is not None and worker.poll() is None:
            worker.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    restarts: list[float] = []
    while True:
        ready_r, ready_w = os.pipe()
        env = os.environ | {
            "ASSISTANT_SPEECH_RESTARTS": str(len(restarts)),
            "ASSISTANT_SPEECH_SUPERVISOR": str(os.getpid()),
        }
        command = [sys.executable, "-m", "assistant_speech", *argv]
        command += ["--worker-fd", str(sock.fileno()), "--ready-fd", str(ready_w)]
        worker = subprocess.Popen(
            command,
            pass_fds=(sock.fileno(), ready_w),
            env=env,
            preexec_fn=_die_with_parent,
            start_new_session=True,  # a terminal's Ctrl-C reaches the worker once, via forward()
        )
        os.close(ready_w)
        code = worker.wait()
        os.set_blocking(ready_r, False)
        try:
            was_ready = bool(os.read(ready_r, 1))
        except BlockingIOError:
            was_ready = False
        os.close(ready_r)
        if stopping or not was_ready:
            return code if code >= 0 else 128 - code
        now = time.monotonic()
        restarts = [t for t in restarts if now - t < RESTART_WINDOW_S] + [now]
        how = f"signal {-code}" if code < 0 else f"exit {code}"
        line("WORKER-DIED", pid=worker.pid, how=how.replace(" ", "_"), restarts=len(restarts))
        if len(restarts) > MAX_RESTARTS:
            line("GIVING-UP", restarts=len(restarts), window_s=RESTART_WINDOW_S)
            return 1


def _without_port(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--port":
            skip = True
        elif not arg.startswith("--port="):
            out.append(arg)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_speech")
    parser.add_argument("--host", default="127.0.0.1", help="loopback only")
    parser.add_argument("--port", type=int, default=8772, help="0 picks a free port")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tts", default="1.7b", choices=sorted(TTS_MODELS))
    parser.add_argument("--tts-quant", default="BF16", choices=["BF16", "Q8_0", "Q4_K_M", "F32"])
    parser.add_argument("--stt", default="parakeet", choices=sorted(STT_MODELS))
    parser.add_argument("--stt-device", default=None, help="default: --device (cpu: STT on CPU)")
    parser.add_argument(
        "--stt-dtype",
        default="auto",
        choices=STT_DTYPES,
        help="auto: bf16 on Ampere+ GPUs (Parakeet), fp16 (Moonshine), fp32 on CPU",
    )
    parser.add_argument("--language", default="english")
    parser.add_argument("--online", action="store_true", help="allow model downloads")
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="load the models in this process (no supervisor: a native crash ends the server)",
    )
    parser.add_argument("--worker-fd", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ready-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        parser.error("the speech server binds to loopback only")
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"

    if args.worker_fd is not None:
        sock = socket.socket(fileno=args.worker_fd)
    else:
        sock = socket.socket(socket.AF_INET6 if ":" in args.host else socket.AF_INET)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
        # From here on a request waits (in the backlog) for a worker instead of being refused.
        sock.listen(128)
        if not args.no_restart:
            port = sock.getsockname()[1]
            given = sys.argv[1:] if argv is None else list(argv)
            return supervise(sock, [*_without_port(given), "--port", str(port)])
    port = sock.getsockname()[1]

    started = time.monotonic()
    models = Models(
        args.device,
        args.tts_quant,
        stt=args.stt,
        tts=args.tts,
        stt_device=args.stt_device,
        stt_dtype=args.stt_dtype,
    )
    load_s = time.monotonic() - started
    warm_s = warm_up(models)
    service = SpeechService(models, args.language)
    app = build_app(service, {"port": port})

    import uvicorn

    config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="off")
    server = uvicorn.Server(config)
    line(
        "READY",
        url=f"http://{args.host}:{port}",
        gpu=models.gpu.replace(" ", "_"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        load_s=f"{load_s:.1f}",
        warmup_s=f"{warm_s:.1f}",
        gpu_memory_mib=models.gpu_memory_mib(),
        stt=models.stt_name,
        stt_device=models.stt_device,
        stt_dtype=models.stt_dtype,
        tts=models.tts_name,
        tts_quant=models.tts_quant,
        pid=os.getpid(),
        restarts=os.environ.get("ASSISTANT_SPEECH_RESTARTS", "0"),
    )
    if args.ready_fd is not None:
        os.write(args.ready_fd, b"R")
        os.close(args.ready_fd)
    try:
        server.run(sockets=[sock])
    finally:
        service.worker.shutdown(wait=False, cancel_futures=True)
        line("STOPPED")
    return 0
