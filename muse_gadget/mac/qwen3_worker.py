"""Qwen3-TTS speech worker (MLX, Apple silicon), run in the Kokoro venv by muse_tts.Qwen3Engine.

    <kokoro venv>/bin/python qwen3_worker.py [--repo R] [--revision SHA] [--speaker Aiden]

Qwen3-TTS 1.7B CustomVoice (8-bit) runs through mlx-audio, which is already in the Kokoro venv
(install.sh), so it needs no venv of its own. It loads the model once from HF_HOME (cached by
install.sh; local_files_only, so it never downloads at run time), warms up with one short phrase,
then prints `ready <rate>`. Audio is streamed: the first chunk comes after about 0.3 s and the
model renders faster than real time (RTF ~0.6 on an M4), so playback can start right away.

Protocol (stdin/stdout, one request at a time):
    -> {"text": "...", "speaker": "Aiden", "instruct": "..." or null}\\n
    <- chunk <samples>\\n then <samples> float32 little-endian mono samples at <rate> Hz (repeated)
    <- end\\n                       (the reply is complete)
    <- err <ExceptionType>\\n       (instead of end; chunks may come before it)
`instruct` only changes the delivery (style, emotion), never the words. Text arrives only on
stdin: it is never on a command line and never logged. Anything the libraries print goes to
stderr, which the parent discards unless MUSE_TTS_DEBUG=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
REVISION = "41d3337e8b7f2843a75841595fc14e4b9a7a4b96"
STREAMING_INTERVAL_S = 0.5  # seconds of audio per streamed chunk


def main() -> int:
    out = os.fdopen(os.dup(1), "wb", buffering=0)  # the protocol channel
    os.dup2(2, 1)  # library print()s go to stderr
    sys.stdout = sys.stderr
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=REPO)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--speaker", default="Aiden", help="speaker to warm up")
    args = p.parse_args()
    try:
        import numpy as np
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model

        model_dir = snapshot_download(args.repo, revision=args.revision, local_files_only=True)
        model = load_model(model_dir)
        speakers = {s.lower(): s for s in model.supported_speakers}

        def stream(text: str, speaker: str, instruct: "str | None"):
            if speaker.lower() not in speakers:
                raise LookupError("unknown speaker")
            for r in model.generate(text=text, voice=speaker, instruct=instruct or None, lang_code="English",
                                    stream=True, streaming_interval=STREAMING_INTERVAL_S, verbose=False):
                audio = np.asarray(r.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield audio

        for _ in stream("Hello.", args.speaker, None):
            pass
        rate = int(model.sample_rate)
    except Exception as e:  # noqa: BLE001  (reported to the parent, which falls back)
        out.write(f"err {type(e).__name__}: {e}\n".encode())
        return 1
    out.write(f"ready {rate}\n".encode())

    for line in sys.stdin:
        try:
            req = json.loads(line)
            for audio in stream(str(req["text"]), str(req.get("speaker") or args.speaker), req.get("instruct")):
                out.write(f"chunk {audio.size}\n".encode() + audio.astype("<f4").tobytes())
            out.write(b"end\n")
        except Exception as e:  # noqa: BLE001
            out.write(f"err {type(e).__name__}\n".encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
