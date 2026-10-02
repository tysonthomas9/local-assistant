#!/usr/bin/env bash
# The stack's own LLM server: `ollama serve` on 127.0.0.1:8773 (the LLM port; vLLM takes it
# later), on GPU0 only, so the speech server always has GPU1 whatever loads first.
#
#   scripts/llm_server.sh [--port N]     # runs in the foreground until stopped (SIGTERM/Ctrl-C)
#
# - GPU0 only: CUDA_VISIBLE_DEVICES=0 with CUDA_DEVICE_ORDER=PCI_BUS_ID (nvidia-smi numbering),
#   and OLLAMA_VULKAN=0: Ollama also drives the GPUs through Vulkan, which ignores
#   CUDA_VISIBLE_DEVICES.
# - Models: the system Ollama's store, read-only (OLLAMA_MODELS, default
#   /usr/share/ollama/.ollama/models; ASSISTANT_OLLAMA_MODELS overrides it), never pruned. No
#   downloads: `reachy-gemma4` must already be there.
# - OLLAMA_NUM_PARALLEL=2 (reachy-gemma4 at its 32k context: about 20 GB of GPU0's 24 GB).
# - Before serving, reachy-gemma4 is unloaded (keep_alive 0, through their API) from the
#   system Ollama service (127.0.0.1:11434) and from our own server on 8773 when this one runs
#   on another port, so a second copy never pushes this one off the GPU. Nothing else is
#   touched: other models and the system service itself are left alone.
set -euo pipefail

PORT=8773
MODEL="reachy-gemma4"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        *) echo "usage: $0 [--port N]" >&2; exit 2 ;;
    esac
done

OLLAMA="$(command -v ollama || echo /usr/local/bin/ollama)"
[[ -x "$OLLAMA" ]] || { echo "llm server: ollama is not installed" >&2; exit 1; }

unload_elsewhere() {
    local url
    for url in http://127.0.0.1:11434 http://127.0.0.1:8773; do
        [[ "$url" == "http://127.0.0.1:$PORT" ]] && continue
        if curl -sf -m 3 "$url/api/ps" 2>/dev/null | grep -q "\"name\":\"$MODEL"; then
            curl -sf -m 60 "$url/api/generate" -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" \
                >/dev/null || true
            echo "llm server: unloaded $MODEL from the Ollama at $url"
        fi
    done
}
unload_elsewhere

export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_MODELS="${ASSISTANT_OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}"
export OLLAMA_NOPRUNE=1
export OLLAMA_NUM_PARALLEL="${ASSISTANT_LLM_NUM_PARALLEL:-2}"
export OLLAMA_KEEP_ALIVE="${ASSISTANT_LLM_KEEP_ALIVE:-30m}"
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OLLAMA_VULKAN=0
echo "llm server: ollama serve on $OLLAMA_HOST, GPU0, models $OLLAMA_MODELS (read-only)"
exec "$OLLAMA" serve
