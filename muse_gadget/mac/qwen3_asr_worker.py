"""Qwen3-ASR speech-to-text worker (MLX, Apple silicon), run in the Kokoro venv by muse_stt.Qwen3AsrEngine.

    <kokoro venv>/bin/python qwen3_asr_worker.py [--model mlx-community/Qwen3-ASR-0.6B-bf16]

The Kokoro venv already has mlx-audio 0.5.7 (the app venv's pins are too old for it), the same
version bench_stt.py measured Qwen3-ASR with. The model is loaded once from HF_HOME (the app runs
with HF_HUB_OFFLINE=1), warmed up with one second of silence, then `ready 16000` is printed.

Protocol (stdin/stdout, one request at a time):
    -> audio <samples>\\n then <samples> float32 little-endian mono samples at 16 kHz
    <- text <bytes>\\n then <bytes> of UTF-8 text
    <- err <ExceptionType>\\n      (instead of text)
Audio and text only travel over the pipes: never on a command line, never logged. Anything the
libraries print goes to stderr, which the parent discards unless MUSE_TTS_DEBUG=1 (shared with the TTS workers).
"""

from __future__ import annotations

import argparse
import os
import sys

MODEL = "mlx-community/Qwen3-ASR-0.6B-bf16"
RATE = 16000


def read_exact(stream, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError("stdin closed")
        data += chunk
    return data


def main() -> int:
    out = os.fdopen(os.dup(1), "wb", buffering=0)  # the protocol channel
    os.dup2(2, 1)  # library print()s go to stderr
    sys.stdout = sys.stderr
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    try:
        import mlx.core as mx
        import numpy as np
        from mlx_audio.stt import load

        model = load(args.model)

        def transcribe(audio: "np.ndarray") -> str:  # same call as bench_stt.py
            return model.generate(mx.array(audio), language="English").text.strip()

        transcribe(np.zeros(RATE, dtype=np.float32))
    except Exception as e:  # noqa: BLE001  (reported to the parent, which falls back)
        out.write(f"err {type(e).__name__}: {e}\n".encode())
        return 1
    out.write(f"ready {RATE}\n".encode())

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        try:
            header = line.decode().split()
            if len(header) != 2 or header[0] != "audio":
                raise ValueError("bad header")
            n = int(header[1])
            audio = np.frombuffer(read_exact(stdin, 4 * n), dtype="<f4").astype(np.float32)
            text = transcribe(audio).encode()
            out.write(f"text {len(text)}\n".encode() + text)
        except EOFError:
            return 0
        except Exception as e:  # noqa: BLE001
            out.write(f"err {type(e).__name__}\n".encode())


if __name__ == "__main__":
    raise SystemExit(main())
