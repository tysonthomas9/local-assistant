#!/bin/bash
# Install or sync Pollen's conversation app (pinned) + MuseHandler's deps on the Mac.
# Run FROM THE PC (run_poc.sh does this):  ssh reachy-mac 'bash -s' < muse_gadget/mac/install.sh
# Our Python files are copied to ~/assistant-edge/muse-app/mac by run_poc.sh before this runs.
#
# Everything lives in ~/assistant-edge/muse-app (nothing else on the Mac changes; uv and the
# uv-managed Python 3.12 are the ones edge_host_bootstrap.sh already installed):
#   app/        git checkout of reachy_mini_conversation_app at APP_COMMIT, venv in app/.venv
#   cache/uv    uv cache;  hf/  HF_HOME with the speech-to-text model
#   installed   stamp: a second run with the same pins only validates (fast)
# Written for bash 3.2 (macOS /bin/bash).
set -euo pipefail

APP_REPO="https://github.com/pollen-robotics/reachy_mini_conversation_app"
APP_COMMIT="f58523bd32ec989c90f8aba402c7ceb8ac038299"   # app 1.0.1, same as the PC
REACHY_MINI="reachy-mini==1.10.0"                        # the daemon's version
EXTRAS="parakeet-mlx==0.5.3 mlx-whisper==0.4.3"
SILERO="silero-vad==6.2.3"                               # --no-deps: only its ONNX model is used
PARAKEET_MODEL="mlx-community/parakeet-tdt-0.6b-v3"
WHISPER_MODEL="mlx-community/whisper-large-v3-turbo"

EDGE="$HOME/assistant-edge"
M="$EDGE/muse-app"
export UV_CACHE_DIR="$M/cache/uv"
export HF_HOME="$M/hf"
say() { printf 'muse-install: %s\n' "$*"; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { say "needs an Apple-silicon Mac"; exit 2; }
UV="$(command -v uv || true)"
[ -n "$UV" ] || UV="$HOME/.local/bin/uv"
[ -x "$UV" ] || { say "uv missing: run scripts/edge_host_bootstrap.sh (branch redesign/architecture) first"; exit 3; }
PY="$("$UV" python find --managed-python 3.12)"
mkdir -p "$M" "$UV_CACHE_DIR" "$HF_HOME"
[ -f "$M/mac/run_app.py" ] || { say "MuseHandler files missing in muse-app/mac (run_poc.sh copies them)"; exit 4; }

stamp="$APP_COMMIT $REACHY_MINI $EXTRAS $SILERO $PARAKEET_MODEL $(shasum -a 256 < "$M/mac/app-constraints.txt" | cut -c1-12)"
if [ "$(cat "$M/installed" 2>/dev/null || true)" = "$stamp" ] && [ -x "$M/app/.venv/bin/python" ]; then
    say "already installed (app ${APP_COMMIT:0:7})"
else
    if [ ! -d "$M/app/.git" ]; then
        say "cloning the conversation app"
        git clone -q "$APP_REPO" "$M/app"
    fi
    if [ "$(git -C "$M/app" rev-parse HEAD)" != "$APP_COMMIT" ]; then
        git -C "$M/app" fetch -q origin
        git -C "$M/app" -c advice.detachedHead=false checkout -q "$APP_COMMIT"
    fi
    say "installing the app venv (pins from the app's uv.lock: app-constraints.txt)"
    [ -x "$M/app/.venv/bin/python" ] || "$UV" venv -q --python "$PY" "$M/app/.venv"
    APY="$M/app/.venv/bin/python"
    "$UV" pip install -q --python "$APY" -c "$M/mac/app-constraints.txt" "$M/app" "$REACHY_MINI" $EXTRAS
    "$UV" pip install -q --python "$APY" --no-deps "$SILERO"
    say "caching speech-to-text model $PARAKEET_MODEL"
    if ! "$APY" -c "from parakeet_mlx import from_pretrained; from_pretrained('$PARAKEET_MODEL')" >/dev/null 2>&1; then
        say "parakeet-mlx failed to load: caching the fallback $WHISPER_MODEL"
        "$APY" -c "import huggingface_hub as h; h.snapshot_download('$WHISPER_MODEL')" >/dev/null
    fi
    printf '%s\n' "$stamp" > "$M/installed"
fi

APY="$M/app/.venv/bin/python"
cd "$M/mac"
HF_HUB_OFFLINE=1 "$APY" - <<'EOF'
import importlib.metadata as m
import muse_vad, run_app  # noqa: F401  (imports MuseHandler's modules)
from reachy_mini_conversation_app import huggingface_realtime  # noqa: F401
vad = type(muse_vad.make_vad()).__name__
print(f"muse-install: ok app={m.version('reachy_mini_conversation_app')} reachy_mini={m.version('reachy-mini')} "
      f"parakeet_mlx={m.version('parakeet-mlx')} vad={vad}")
EOF
