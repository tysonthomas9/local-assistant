"""Weight-memory calculator for models we don't load: sums safetensors tensor bytes by module prefix.

Reads only the safetensors headers, from a local checkpoint dir or from a Hugging Face repo at a
pinned revision (HTTP range requests, no weights downloaded). Bytes are as stored; a module that
the loader casts (e.g. fp32 -> bf16) is reported separately with its cast size.

    python analysis/fit_calc.py DIR_OR_REPO [--rev SHA] [--depth N] [--cast PREFIX=DTYPE ...]

Example (VoiceChat: LLM cast to bf16, rest left in fp32 as NVIDIA's loader does):
    python analysis/fit_calc.py ~/assistant-dxpoc/models/VoiceChat-11B --depth 1 --cast llm=bf16
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import requests

SIZE = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
        "F8_E4M3": 1, "F8_E5M2": 1}
CAST = {"bf16": 2, "fp16": 2, "fp32": 4}


def headers_local(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.safetensors")):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            out.append(json.loads(f.read(n)))
    return out


def headers_remote(repo: str, rev: str) -> list[dict]:
    files = requests.get(f"https://huggingface.co/api/models/{repo}/revision/{rev}", timeout=30).json()["siblings"]
    out = []
    for s in files:
        if not s["rfilename"].endswith(".safetensors"):
            continue
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{s['rfilename']}"
        n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content)[0]
        h = requests.get(url, headers={"Range": f"bytes=8-{7 + n}"}, timeout=60).content
        out.append(json.loads(h))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--rev", default="main")
    ap.add_argument("--depth", type=int, default=1, help="prefix depth to group by (dot-separated)")
    ap.add_argument("--cast", nargs="*", default=[], help="PREFIX=bf16|fp16|fp32 (float tensors only)")
    a = ap.parse_args()
    casts = dict(c.split("=", 1) for c in a.cast)
    src = Path(a.src).expanduser()
    hs = headers_local(src) if src.is_dir() else headers_remote(a.src, a.rev)
    stored: dict[str, int] = collections.Counter()
    loaded: dict[str, int] = collections.Counter()
    for h in hs:
        for name, t in h.items():
            if name == "__metadata__":
                continue
            numel = 1
            for s in t["shape"]:
                numel *= s
            key = ".".join(name.split(".")[: a.depth])
            b = numel * SIZE[t["dtype"]]
            stored[key] += b
            c = next((casts[p] for p in casts if name.startswith(p)), None)
            loaded[key] += numel * CAST[c] if c and t["dtype"] in ("F32", "BF16", "F16") else b
    print(f"{'module':40s} {'stored GB':>10s} {'loaded GB':>10s}")
    for k in sorted(stored, key=stored.get, reverse=True):
        print(f"{k:40s} {stored[k] / 1e9:10.2f} {loaded[k] / 1e9:10.2f}")
    ts, tl = sum(stored.values()), sum(loaded.values())
    print(f"{'TOTAL':40s} {ts / 1e9:10.2f} {tl / 1e9:10.2f}   (loaded = {tl / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
