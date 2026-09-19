#!/usr/bin/env bash
# Start the Reachy Mini daemon with the WebRTC plugin and the device groups it needs, fully offline:
#   HF_HUB_OFFLINE=1               never contact huggingface.co (models/datasets are already cached;
#                                  run local_backend/cache_models.sh once while online)
#   --dataset-update-interval 0    disable the daemon's 24-hourly dataset update check
#   local_backend/run_daemon.py    binds the WebRTC signalling server (port 8443, camera + mic
#                                  stream) to 127.0.0.1 instead of 0.0.0.0
# Extra arguments are passed to reachy-mini-daemon.
set -euo pipefail

# Until you log out and back in after `usermod -aG`, re-run this script under `sg` for each group
# the session is missing. `sg -c` runs /bin/sh (dash), which can't parse the $'...' quoting that
# printf %q emits for control or non-ASCII characters, so refuse those instead of mangling them.
for g in dialout audio video; do
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
export GST_PLUGIN_PATH="$HOME/.local/gst-plugins-rs/lib/x86_64-linux-gnu${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"
export HF_HUB_OFFLINE=1
export ORT_DISABLE_TELEMETRY=1   # onnxruntime 1.27 here has no telemetry; guards against upgrades
cd "$ROOT"
exec .venv/bin/python local_backend/run_daemon.py --dataset-update-interval 0 "$@"
