#!/usr/bin/env bash
# Start the Reachy Mini conversation app. The daemon must already be running (./start_daemon.sh).
#   ./start_conversation.sh --ui                 hosted Hugging Face backend (cloud)
#   ./start_conversation.sh --local --ui         fully local backend (run local_backend/start_local_backend.sh first)
#   ./start_conversation.sh --local --web --ui   local speech + LLM, plus online tools (weather, web search via the
#                                                local SearXNG, tech news); run local_backend/start_searxng.sh first
#   add --wake                                   only listen after the wake word ("Hey Marvin" by default;
#                                                REACHY_WAKE_WORD="<any phrase>" or hey_jarvis|hey_mycroft|hey_marvin|alexa); local modes only
# Other arguments are passed through to the app, e.g. --no-camera, --debug.
# The app runs via local_backend/run_app.py, which binds the --ui web page to 127.0.0.1 instead of
# 0.0.0.0 (set REACHY_UI_HOST=0.0.0.0 to open it to the LAN).
set -euo pipefail

# Until you log in again after joining the groups, re-run under `sg` (see start_daemon.sh for why
# unusual characters in arguments are refused).
for g in audio video; do
    if ! id -nG | grep -qw "$g"; then
        for a in "$@"; do
            if LC_ALL=C grep -q '[^[:print:]]' <<<"$a"; then
                echo "Argument contains control or non-ASCII characters; log in again so sg isn't needed: $a" >&2
                exit 2
            fi
        done
        exec sg "$g" -c "$(printf '%q ' "$0" "$@")"
    fi
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
# --web implies --local.
local=0; wake=0; profile=local_reachy; args=()
for a in "$@"; do
    case "$a" in
        --local) local=1 ;;
        --web)   local=1; profile=local_reachy_web ;;
        --wake)  wake=1 ;;
        *)       args+=("$a") ;;
    esac
done
if [ "$wake" = 1 ]; then
    if [ "$local" = 1 ]; then
        export REACHY_WAKE_WORD="${REACHY_WAKE_WORD:-hey_marvin}"
    else
        echo "Note: --wake only works with --local/--web; ignoring it." >&2
    fi
fi
if [ "$local" = 1 ]; then
    export HF_REALTIME_CONNECTION_MODE=local
    export HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
    export REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY="$ROOT/local_backend/profiles"
    export REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY="$ROOT/local_backend/tools"
    export REACHY_MINI_CUSTOM_PROFILE="$profile"
    export HF_HUB_OFFLINE=1
    export ORT_DISABLE_TELEMETRY=1   # harmless for onnxruntime 1.27 (no telemetry); guards future upgrades
    if [ "${REACHY_NET_AUDIT:-0}" = 1 ]; then
        export PYTHONPATH="$ROOT/local_backend/netaudit${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if [ "$profile" = local_reachy_web ] && ! curl -sf -o /dev/null "http://127.0.0.1:8888/search?q=ping&format=json"; then
        echo "Warning: SearXNG isn't answering on 127.0.0.1:8888; web_search will fail. Run local_backend/start_searxng.sh" >&2
    fi
fi

export GST_PLUGIN_PATH="$HOME/.local/gst-plugins-rs/lib/x86_64-linux-gnu${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"
cd "$ROOT/reachy_mini_conversation_app"
exec .venv/bin/python "$ROOT/local_backend/run_app.py" "${args[@]}"
