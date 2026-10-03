"""Kokoro-82M speech worker (MLX, Apple silicon), run in its own venv by muse_tts.KokoroEngine.

    <kokoro venv>/bin/python kokoro_worker.py [--repo R] [--revision SHA] [--voice af_heart]

mlx-audio needs torch, transformers and spaCy, which don't fit the conversation app's pinned
venv, so it runs here, in ~/assistant-edge/muse-app/kokoro/.venv (install.sh), as a child of the
app. It loads the model once from HF_HOME (cached by install.sh; loaded with local_files_only, so
it never downloads at run time), warms up with one short phrase, then prints `ready <rate>`.

Protocol (stdin/stdout, one request at a time):
    -> {"text": "...", "voice": "af_heart", "speed": 1.0}\\n
    <- ok <samples>\\n then <samples> float32 little-endian mono samples at <rate> Hz
    <- err <ExceptionType>\\n
Text arrives only on stdin: it is never on a command line and never logged. Anything the
libraries print goes to stderr, which the parent discards unless MUSE_TTS_DEBUG=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = "mlx-community/Kokoro-82M-bf16"
REVISION = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
SAMPLE_RATE = 24000
FILES = ["config.json", "*.safetensors"]  # the weights and voices/*.safetensors (install.sh caches these)


def main() -> int:
    out = os.fdopen(os.dup(1), "wb", buffering=0)  # the protocol channel
    os.dup2(2, 1)  # library print()s (e.g. "Creating new KokoroPipeline") go to stderr
    sys.stdout = sys.stderr
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=REPO)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--voice", default="af_heart", help="voice to warm up")
    args = p.parse_args()
    try:
        import numpy as np
        from huggingface_hub import snapshot_download
        from mlx_audio.tts.utils import load_model

        model_dir = snapshot_download(args.repo, revision=args.revision, allow_patterns=FILES,
                                      local_files_only=True)
        model = load_model(model_dir, model_type="kokoro")  # a cache path, so name it

        def synth(text: str, voice: str, speed: float) -> "np.ndarray":
            path = os.path.join(model_dir, "voices", f"{voice}.safetensors")
            if not voice.replace("_", "").isalnum() or not os.path.isfile(path):
                raise LookupError("unknown voice")
            parts = [
                np.asarray(r.audio, dtype=np.float32).reshape(-1)
                for r in model.generate(text=text, voice=path, speed=speed, lang_code=voice[0],
                                        split_pattern=r"\n+", verbose=False)
            ]
            return np.concatenate(parts) if parts else np.zeros(0, np.float32)

        synth("Hello.", args.voice, 1.0)
    except Exception as e:  # noqa: BLE001  (reported to the parent, which falls back to `say`)
        out.write(f"err {type(e).__name__}: {e}\n".encode())
        return 1
    out.write(f"ready {SAMPLE_RATE}\n".encode())

    for line in sys.stdin:
        try:
            req = json.loads(line)
            audio = synth(str(req["text"]), str(req.get("voice") or args.voice), float(req.get("speed") or 1.0))
            out.write(f"ok {audio.size}\n".encode() + audio.astype("<f4").tobytes())
        except Exception as e:  # noqa: BLE001
            out.write(f"err {type(e).__name__}\n".encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
