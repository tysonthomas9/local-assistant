"""The stack's LLM server as the tests and the gate see it: vLLM (the default) or Ollama.

Both are started by `scripts/llm_server.sh` on the LLM's GPU (`[gpu]`: GPU0 in layout `two`,
`gpu_index` in layout `one`) and serve the OpenAI API with the model `reachy-gemma4`. This
module tells them apart by their own endpoints (Ollama `/api/version`, vLLM `/version`), loads
the model with a one-token request and checks it sits entirely on that GPU: Ollama's `/api/ps`
(all of it in VRAM) and, for both, the GPU memory of the server's own processes (found from
the port's listener and its children) is on that GPU and nowhere else. A sleeping vLLM (put to
sleep by a scenario's own server, see the launcher) is woken first, once the GPU has room for
it again.

    python -m assistant_testing.llm_server check [--url URL] [--server vllm|ollama]
    python -m assistant_testing.llm_server server     # the configured server (profile ci)
    python -m assistant_testing.llm_server gpu        # "<layout> <llm gpu> <speech gpu>"
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from assistant_core.config import GpuConfig, load_config

ServerKind = Literal["vllm", "ollama"]
STACK_URL = "http://127.0.0.1:8773"
"""The stack's LLM server (the gate starts it; `scripts/llm_server.sh`)."""
MODEL = "reachy-gemma4"
SCRIPT = "scripts/llm_server.sh"
REPO_ROOT = Path(__file__).resolve().parents[4]
LOAD_TIMEOUT_S = 180.0
START_TIMEOUT_S: dict[ServerKind, float] = {"vllm": 600.0, "ollama": 30.0}
"""vLLM loads the weights, compiles (cached after the first run) and captures CUDA graphs."""
MIN_GPU_MIB: dict[ServerKind, int] = {"vllm": 15000, "ollama": 15000}
"""The model's weights alone take about 15.6 GiB (vLLM) / 17 GB (Ollama) of GPU memory."""
WAKE_FREE_MIB = {"two": 19800, "one": 16500}
"""GPU memory a sleeping vLLM needs back (weights, KV cache, CUDA graphs), per layout (on one
card its KV cache is sized to what the speech server leaves)."""
WAKE_WAIT_S = 60.0


class LlmServerError(AssertionError):
    """The LLM server is missing, of the wrong kind, or not entirely on its GPU."""


def http(method: str, url: str, body: Any = None, timeout_s: float = 10.0) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise LlmServerError(f"{method} {url}: {exc.code} {exc.read().decode()[:300]}") from exc
    return json.loads(raw) if raw.strip() else None


def kind(url: str) -> ServerKind | None:
    """Which server answers at `url` (None: nothing does)."""
    try:
        if "version" in (http("GET", f"{url}/api/version", None, 2) or {}):
            return "ollama"
    except (OSError, LlmServerError, ValueError):
        pass
    try:
        if "version" in (http("GET", f"{url}/version", None, 2) or {}):
            return "vllm"
    except (OSError, LlmServerError, ValueError):
        pass
    return None


def up(url: str) -> bool:
    return kind(url) is not None


def configured_server(repo_root: Path, environ: Mapping[str, str] | None = None) -> ServerKind:
    """`[llm] server` of the ci profile (the one the tests' brain runs with); the env var
    ASSISTANT__LLM__SERVER overrides it."""
    env = os.environ if environ is None else environ
    keep = {k: v for k, v in env.items() if k == "ASSISTANT__LLM__SERVER"}
    return load_config(repo_root / "config", profile="ci", environ=keep).llm.server


def gpu_config(repo_root: Path = REPO_ROOT, environ: Mapping[str, str] | None = None) -> GpuConfig:
    """`[gpu]` of the ci profile; the env vars ASSISTANT__GPU__* override it (GATE_GPU sets
    ASSISTANT__GPU__LAYOUT)."""
    env = os.environ if environ is None else environ
    keep = {k: v for k, v in env.items() if k.upper().startswith("ASSISTANT__GPU__")}
    return load_config(repo_root / "config", profile="ci", environ=keep).gpu


def llm_gpu() -> str:
    """The LLM server's GPU (nvidia-smi numbering) in the configured layout."""
    return str(gpu_config().llm_gpu)


def _run(argv: list[str]) -> str:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout


def _listener_pids(port: int) -> set[int]:
    pids: set[int] = set()
    for line in _run(["ss", "-ltnpH", f"sport = :{port}"]).splitlines():
        for part in line.split("pid=")[1:]:
            digits = part.split(",")[0]
            if digits.isdigit():
                pids.add(int(digits))
    return pids


def _process_tree(roots: set[int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
    found, todo = set(roots), list(roots)
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in found:
                found.add(child)
                todo.append(child)
    return found


def gpu_apps() -> list[dict[str, Any]]:
    """Every compute process on the GPUs: index, pid, name, MiB."""
    index_of = {}
    for line in _run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"]
    ).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            index_of[parts[1]] = parts[0]
    query = ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"]  # fmt: skip
    apps = []
    for line in _run(query).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[1].isdigit() and parts[3].isdigit():
            apps.append({"gpu": index_of.get(parts[0], "?"), "pid": int(parts[1]),
                         "name": parts[2], "mib": int(parts[3])})  # fmt: skip
    return apps


def gpu_free_mib(index: str | None = None) -> int | None:
    index = llm_gpu() if index is None else index
    out = _run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"]).strip()  # fmt: skip
    return int(out.splitlines()[0]) if out and out.splitlines()[0].isdigit() else None


def server_gpu_mib(url: str) -> dict[str, int]:
    """GPU memory (MiB) of the processes of the server listening at `url`, per GPU index."""
    port = int(url.rsplit(":", 1)[1].split("/")[0])
    tree = _process_tree(_listener_pids(port))
    usage: dict[str, int] = {}
    for app in gpu_apps():
        if app["pid"] in tree:
            usage[app["gpu"]] = usage.get(app["gpu"], 0) + app["mib"]
    return usage


def is_sleeping(url: str) -> bool:
    try:
        return bool((http("GET", f"{url}/is_sleeping", None, 5) or {}).get("is_sleeping"))
    except (OSError, LlmServerError, ValueError):
        return False


def wake(url: str) -> bool:
    """Wake a sleeping vLLM at `url` once its GPU has room; True if it was asleep."""
    if not is_sleeping(url):
        return False
    gpu, needs = llm_gpu(), WAKE_FREE_MIB[gpu_config().layout]
    deadline = time.monotonic() + WAKE_WAIT_S
    free = gpu_free_mib(gpu)
    while (free is None or free < needs) and time.monotonic() < deadline:
        time.sleep(1)
        free = gpu_free_mib(gpu)
    if free is None or free < needs:
        others = ", ".join(f"{a['name']} pid {a['pid']} {a['mib']} MiB" for a in gpu_apps()
                           if a["gpu"] == gpu)  # fmt: skip
        raise LlmServerError(
            f"the vLLM at {url} sleeps and GPU{gpu} has only {free} MiB free (it needs "
            f"{needs}): {others or 'no compute process listed'}"
        )
    http("POST", f"{url}/wake_up", None, 120)
    if is_sleeping(url):
        raise LlmServerError(f"the vLLM at {url} is still asleep after /wake_up")
    return True


def sleep(url: str) -> None:
    http("POST", f"{url}/sleep?level=1", None, 120)


def load_and_check(url: str, model: str = MODEL, server: ServerKind | None = None) -> str:
    """Load `model` at `url` (a one-token request; a sleeping vLLM is woken first) and check it
    sits entirely on the LLM's GPU. `server`: also require that kind. Returns a line for the
    log."""
    found_kind = kind(url)
    if found_kind is None:
        raise LlmServerError(f"no LLM server answers at {url}")
    if server is not None and found_kind != server:
        raise LlmServerError(
            f"the LLM server at {url} is {found_kind}, but [llm] server is {server!r} "
            f"(scripts/llm_server.sh --server {server}, or set [llm] server = {found_kind!r})"
        )
    woke = found_kind == "vllm" and wake(url)
    models = (http("GET", f"{url}/v1/models", None, 10) or {}).get("data", [])
    ids = [m.get("id", "") for m in models]
    if not any(i == model or i.split(":")[0] == model for i in ids):
        raise LlmServerError(f"the LLM server at {url} has no {model!r} (it has {ids})")
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    http("POST", f"{url}/v1/chat/completions", body, LOAD_TIMEOUT_S)
    detail = ""
    if found_kind == "ollama":
        loaded = (http("GET", f"{url}/api/ps", None, 10) or {}).get("models", [])
        mine = [m for m in loaded if m.get("name", "").split(":")[0] == model.split(":")[0]]
        if not mine:
            raise LlmServerError(f"{model} is not loaded at {url} after a request: {loaded}")
        size, vram = int(mine[0].get("size", 0)), int(mine[0].get("size_vram", 0))
        if size <= 0 or vram < size:
            raise LlmServerError(
                f"{model} at {url} is not entirely on the GPU ({vram} of {size} bytes in GPU "
                f"memory): GPU{llm_gpu()} is short of memory"
            )
        detail = f"context {mine[0].get('context_length')}"
    else:
        info = models[0] if models else {}
        detail = f"max_model_len {info.get('max_model_len')}{', woken from sleep' if woke else ''}"
    gpu = llm_gpu()
    usage = server_gpu_mib(url)
    elsewhere = {g: mib for g, mib in usage.items() if g != gpu}
    if elsewhere:
        raise LlmServerError(f"the {found_kind} at {url} also uses GPU(s) {elsewhere} MiB")
    on_gpu = usage.get(gpu, 0)
    if on_gpu < MIN_GPU_MIB[found_kind]:
        raise LlmServerError(
            f"the {found_kind} at {url} holds only {on_gpu} MiB on GPU{gpu} (the model needs "
            f"at least {MIN_GPU_MIB[found_kind]}): it is not entirely on the GPU ({usage})"
        )
    return (
        f"{found_kind} at {url}: {model} entirely on GPU{gpu} ({on_gpu} MiB, nothing on "
        f"other GPUs, layout {gpu_config().layout}), {detail}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant_testing.llm_server")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="load the model and check it is entirely on its GPU")
    check.add_argument("--url", default=STACK_URL)
    check.add_argument("--model", default=MODEL)
    check.add_argument("--server", choices=["vllm", "ollama"])
    sub.add_parser("server", help="print the configured [llm] server (profile ci)")
    sub.add_parser("gpu", help="print the [gpu] layout, the LLM's GPU and the speech GPU")
    args = parser.parse_args(argv)
    if args.command == "server":
        print(configured_server(Path.cwd()))
        return 0
    if args.command == "gpu":
        gpu = gpu_config(Path.cwd())
        print(gpu.layout, gpu.llm_gpu, gpu.speech_gpu)
        return 0
    try:
        print(load_and_check(args.url, args.model, args.server))
    except (LlmServerError, OSError) as exc:
        print(f"LLM server: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
