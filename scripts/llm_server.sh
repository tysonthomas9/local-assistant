#!/usr/bin/env bash
# The stack's own LLM server on 127.0.0.1:8773 (the LLM port), on GPU0 only, so the speech
# server always has GPU1 whatever loads first. vLLM by default; Ollama is the fallback.
#
#   scripts/llm_server.sh [--server vllm|ollama] [--port N]   # foreground until SIGTERM/Ctrl-C
#
# The server is `--server`, else ASSISTANT__LLM__SERVER, else `[llm] server` of the config
# (profile $ASSISTANT_PROFILE, default dev). Both serve the OpenAI API (`/v1`) with the model
# `reachy-gemma4`, so switching is that one value.
#
# Common: GPU0 only (CUDA_VISIBLE_DEVICES=0, CUDA_DEVICE_ORDER=PCI_BUS_ID: nvidia-smi
# numbering). Before serving, the model is freed from the other copies that may hold GPU0,
# through their APIs: unloaded from an Ollama (the system service on 127.0.0.1:11434 and the
# stack's on 8773; keep_alive 0) and a vLLM on 8773 is put to sleep (level 1: its weights move
# to CPU memory, about 0.8 GB stays on the GPU; the test steps wake it again). Nothing else is
# touched: other models and the system Ollama service itself are left alone.
#
# vLLM: servers/vllm/.venv (pinned, synced here with `uv sync --locked`), serving
#   cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit from the local HF cache (offline: no downloads) as
#   `reachy-gemma4` (alias `gemma4-26b`), text only. Flags: max-model-len 32768,
#   gpu-memory-utilization 0.90 (about 21 GB of GPU0: 15.6 GB weights, 3.9 GB KV cache =
#   80k tokens), max-num-seqs 8, prefix caching, priority scheduling (the brain's voice turns
#   go first), gemma4 tool-call and reasoning parsers, sleep mode. ASSISTANT_VLLM_* override
#   the sizes. FlashInfer's sampler is off (its JIT build needs a newer nvcc than the PC's).
# Ollama: `ollama serve` with the system Ollama's model store read-only (OLLAMA_MODELS, default
#   /usr/share/ollama/.ollama/models; ASSISTANT_OLLAMA_MODELS overrides it), never pruned, no
#   downloads (`reachy-gemma4` must be there), OLLAMA_NUM_PARALLEL=2 (about 20 GB of GPU0),
#   and OLLAMA_VULKAN=0: Ollama also drives the GPUs through Vulkan, which ignores
#   CUDA_VISIBLE_DEVICES.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=8773
SERVER="${ASSISTANT__LLM__SERVER:-}"
MODEL="reachy-gemma4"
VLLM_WEIGHTS="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --server) SERVER="$2"; shift 2 ;;
        *) echo "usage: $0 [--server vllm|ollama] [--port N]" >&2; exit 2 ;;
    esac
done
if [[ -z "$SERVER" ]]; then
    SERVER="$(cd "$ROOT" && UV_PROJECT_ENVIRONMENT=.venv-assistant uv run --locked -q python -c \
        'import os, sys; from assistant_core.config import load_config
print(load_config("config", profile=os.environ.get("ASSISTANT_PROFILE", "dev")).llm.server)')"
fi
case "$SERVER" in
    vllm|ollama) ;;
    *) echo "llm server: unknown server '$SERVER' (vllm or ollama)" >&2; exit 2 ;;
esac

# free_gpu0: the model must not sit on GPU0 twice.
free_gpu0() {
    local url
    for url in http://127.0.0.1:11434 http://127.0.0.1:8773; do
        [[ "$url" == "http://127.0.0.1:$PORT" ]] && continue
        if curl -sf -m 3 "$url/api/ps" 2>/dev/null | grep -q "\"name\":\"$MODEL"; then
            curl -sf -m 60 "$url/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" \
                >/dev/null || true
            echo "llm server: unloaded $MODEL from the Ollama at $url"
        elif curl -sf -m 3 "$url/is_sleeping" 2>/dev/null | grep -q '"is_sleeping":false'; then
            curl -sf -m 120 -X POST "$url/sleep?level=1" >/dev/null || true
            echo "llm server: put the vLLM at $url to sleep"
        fi
    done
}
free_gpu0

export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ "$SERVER" == "ollama" ]]; then
    OLLAMA="$(command -v ollama || echo /usr/local/bin/ollama)"
    [[ -x "$OLLAMA" ]] || { echo "llm server: ollama is not installed" >&2; exit 1; }
    export OLLAMA_HOST="127.0.0.1:$PORT"
    export OLLAMA_MODELS="${ASSISTANT_OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
    export OLLAMA_NOPRUNE=1
    export OLLAMA_NUM_PARALLEL="${ASSISTANT_LLM_NUM_PARALLEL:-2}"
    export OLLAMA_KEEP_ALIVE="${ASSISTANT_LLM_KEEP_ALIVE:-30m}"
    export OLLAMA_VULKAN=0
    echo "llm server: ollama serve on $OLLAMA_HOST, GPU0, models $OLLAMA_MODELS (read-only)"
    exec "$OLLAMA" serve
fi

VENV="$ROOT/servers/vllm/.venv"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --locked -q --project "$ROOT/servers/vllm"
export PATH="$VENV/bin:$PATH"      # its own ninja first (another one on PATH may be a wrapper)
export HF_HUB_OFFLINE=1            # the weights are in the local HF cache; never download
export VLLM_SERVER_DEV_MODE=1      # /sleep, /wake_up, /is_sleeping
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
ARGS=(
    serve "$VLLM_WEIGHTS"
    --host 127.0.0.1 --port "$PORT"
    --served-model-name "$MODEL" gemma4-26b
    --max-model-len "${ASSISTANT_VLLM_MAX_MODEL_LEN:-32768}"
    --gpu-memory-utilization "${ASSISTANT_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
    --max-num-seqs "${ASSISTANT_VLLM_MAX_NUM_SEQS:-8}"
    --enable-prefix-caching
    --scheduling-policy priority
    --enable-auto-tool-choice --tool-call-parser gemma4
    --reasoning-parser gemma4
    --language-model-only
    --enable-sleep-mode
)
echo "llm server: vllm $("$VENV/bin/python" -c 'import vllm; print(vllm.__version__)') on 127.0.0.1:$PORT, GPU0, $VLLM_WEIGHTS as $MODEL"
echo "llm server: vllm ${ARGS[*]:2}"
exec "$VENV/bin/vllm" "${ARGS[@]}"
