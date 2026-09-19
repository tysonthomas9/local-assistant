#!/usr/bin/env bash
# Start the fully local voice backend for the Reachy Mini conversation app:
#   - loads the Ollama model and keeps it loaded (it takes ~10 s to load, and Ollama
#     unloads after 5 idle minutes; every /v1 request resets that timer to 5 min),
#   - runs the speech-to-speech Realtime server at ws://127.0.0.1:8765/v1/realtime, on whichever
#     GPU has the most free memory once the LLM is loaded.
#
# Settings (environment variables):
#   REACHY_LLM=reachy-gemma4        Ollama model (created from Modelfile.<name> if missing)
#   REACHY_STT=parakeet             parakeet (fast, English + 24 European languages)
#                                   | whisper (large-v3-turbo, multilingual, auto language, ~0.3 s slower)
#   REACHY_VAD_THRESH=0.45          voice-activity threshold (speech-to-speech default 0.6). Lower picks up
#                                   quieter/farther speech, but also more TV/background audio.
#   REACHY_VAD_MIN_SPEECH_MS=256    shortest speech kept (default 384); shorter segments are discarded
#   REACHY_STREAM_SENTENCES=1       sentences to collect before starting TTS (default 3); 1 starts speaking
#                                   after the first sentence instead of the whole short reply
#   REACHY_NET_AUDIT=1              log non-loopback DNS lookups / connections made from *Python code*
#                                   (with stack traces) to local_backend/logs/net_audit.log. Native code
#                                   (C++/Rust libraries) is invisible to it; use strace for that.
#   REACHY_SPEECH_GPU=<index>       GPU for speech-to-text and text-to-speech. Default: whichever GPU has
#                                   the most free memory after the LLM is loaded (Ollama picks the LLM's
#                                   GPU itself, and it isn't always the same one).
#
# Logs go to local_backend/logs/speech.log. Ctrl+C stops everything this script started.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
S2S="$ROOT/third_party/speech-to-speech"
LOGS="$HERE/logs"
mkdir -p "$LOGS"

LLM="${REACHY_LLM:-reachy-gemma4}"
VAD_THRESH="${REACHY_VAD_THRESH:-0.45}"
VAD_MIN_SPEECH_MS="${REACHY_VAD_MIN_SPEECH_MS:-256}"
STREAM_SENTENCES="${REACHY_STREAM_SENTENCES:-1}"
STT="${REACHY_STT:-parakeet}"
OLLAMA=http://127.0.0.1:11434
# nvidia-smi numbers GPUs in PCI bus order; make CUDA use the same numbering.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if ss -ltn | grep -q ':8765 '; then
    echo "Port 8765 is already in use; is the speech server already running?" >&2
    exit 1
fi

# 1. Ollama model: create it if needed, load it, and keep it loaded.
if ! ollama list | awk '{print $1}' | grep -qx "$LLM\(:latest\)\?"; then
    echo "Creating Ollama model $LLM from $HERE/Modelfile.$LLM"
    ollama create "$LLM" -f "$HERE/Modelfile.$LLM"
fi
echo "Loading $LLM into GPU memory..."
if ! curl -sf "$OLLAMA/api/generate" -d "{\"model\":\"$LLM\",\"keep_alive\":\"10m\"}" >/dev/null; then
    echo "Ollama could not load $LLM (is ollama running, and does the model's FROM base exist?)" >&2
    exit 1
fi
(
    while sleep 120; do
        curl -sf "$OLLAMA/api/generate" -d "{\"model\":\"$LLM\",\"keep_alive\":\"10m\"}" >/dev/null || true
    done
) &
KEEPALIVE_PID=$!

# Speech needs ~7 GB. Put it on the GPU with the most free memory now that the LLM is loaded.
GPU="${REACHY_SPEECH_GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1)}"
FREE_MIB=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE_MIB" -lt 8000 ]; then
    echo "Warning: GPU $GPU has only $FREE_MIB MiB free; the speech server needs ~7 GB." >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv >&2
fi

# 2. Speech-to-text choice.
case "$STT" in
    parakeet) STT_ARGS=(--stt parakeet-tdt) ;;
    whisper)  STT_ARGS=(--stt faster-whisper --faster_whisper_stt_model_name large-v3-turbo
                        --faster_whisper_stt_compute_type float16 --faster_whisper_stt_gen_language auto) ;;
    *) echo "Unknown REACHY_STT=$STT (use parakeet or whisper)" >&2; kill "$KEEPALIVE_PID"; exit 1 ;;
esac

cleanup() {
    pkill -P "$KEEPALIVE_PID" 2>/dev/null || true   # the in-flight `sleep 120`
    kill "$KEEPALIVE_PID" 2>/dev/null || true
    [ -n "${SPEECH_PID:-}" ] && kill -INT "$SPEECH_PID" 2>/dev/null && wait "$SPEECH_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "${REACHY_NET_AUDIT:-0}" = 1 ]; then
    export PYTHONPATH="$HERE/netaudit${PYTHONPATH:+:$PYTHONPATH}"
    echo "Network audit on: $LOGS/net_audit.log"
fi

# 3. Realtime speech server.
#    HF_HUB_OFFLINE: models are already cached; nothing is downloaded.
#    ORT_DISABLE_TELEMETRY: onnxruntime 1.30 (used for the Smart Turn model) otherwise uploads
#    telemetry to https://mobile.events.data.microsoft.com/OneCollector/1.0 on Linux.
echo "Starting speech server (STT=$STT, LLM=$LLM, GPU $GPU, VAD thresh=$VAD_THRESH min_speech=${VAD_MIN_SPEECH_MS}ms, stream_sentences=$STREAM_SENTENCES); log: $LOGS/speech.log"
cd "$S2S"
LOG_START=$(( $(wc -l < "$LOGS/speech.log" 2>/dev/null || echo 0) + 1 ))
HF_HUB_OFFLINE=1 ORT_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/speech-to-speech serve \
    --device cuda "${STT_ARGS[@]}" \
    --llm_backend chat-completions --model_name "$LLM" \
    --responses_api_base_url "$OLLAMA/v1" --responses_api_api_key ollama \
    --responses_api_reasoning_effort none --responses_api_stream \
    --tts qwen3 --qwen3_tts_backend ggml --qwen3_tts_speaker Aiden \
    --thresh "$VAD_THRESH" --min_speech_ms "$VAD_MIN_SPEECH_MS" --stream_batch_sentences "$STREAM_SENTENCES" \
    --enable_live_transcription --port 8765 \
    >>"$LOGS/speech.log" 2>&1 &
SPEECH_PID=$!

READY=0
for _ in $(seq 1 90); do
    # Only look at lines this run wrote: speech.log is appended to, and holds older "ready" lines.
    if tail -n +"$LOG_START" "$LOGS/speech.log" | grep -q "Uvicorn running on http://127.0.0.1:8765"; then
        echo "Ready: ws://127.0.0.1:8765/v1/realtime (speech server pid $SPEECH_PID)"
        READY=1
        break
    fi
    kill -0 "$SPEECH_PID" 2>/dev/null || { echo "Speech server exited; see $LOGS/speech.log" >&2; exit 1; }
    sleep 2
done
[ "$READY" = 1 ] || echo "Warning: speech server not ready after 180 s; still waiting. See $LOGS/speech.log" >&2
wait "$SPEECH_PID"
