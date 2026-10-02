#!/usr/bin/env bash
# Prepare a remote edge host (the machine the robot is plugged into) for the robot tests.
# It runs ON the edge host, fed over SSH from the brain PC:
#
#   ssh reachy-mac 'bash -s' < scripts/edge_host_bootstrap.sh
#
# The e2e step `edge_host_bootstrapped` (assistant_testing.edge_host) runs it the same way.
# It is idempotent: a second run only validates. Everything it creates lives in
# ~/assistant-edge/ (override with ASSISTANT_EDGE_DIR); nothing outside it is changed, except
# that uv and a uv-managed Python 3.12 are installed (in uv's usual places) when missing.
#
#   ~/assistant-edge/daemon/     venv with the pinned reachy-mini SDK + daemon (and, on macOS,
#                                its media stack: the gstreamer-bundle wheel, no Homebrew needed)
#   ~/assistant-edge/repo.git    bare repo the PC pushes the commit under test to (never GitHub)
#   ~/assistant-edge/src/        checkout of that commit; `uv sync` puts the edge packages in
#                                src/.venv-assistant (UV_PROJECT_ENVIRONMENT)
#   ~/assistant-edge/cache/uv    uv cache (kept here so the host's own cache is untouched)
#   ~/assistant-edge/hf/         HF_HOME for the daemon
#
# The last line printed is `edge-host ready: key=value ...` (parsed by the e2e step).
# Written for bash 3.2 (macOS /bin/bash) as well as Linux.
set -euo pipefail

REACHY_MINI_VERSION="1.10.0"   # same as the brain PC (legacy root .venv)
PYTHON_VERSION="3.12"

EDGE="${ASSISTANT_EDGE_DIR:-$HOME/assistant-edge}"
export UV_CACHE_DIR="$EDGE/cache/uv"
export UV_NO_PROGRESS=1
export HF_HOME="$EDGE/hf"
export HF_HUB_OFFLINE=1

say() { printf 'bootstrap: %s\n' "$*"; }

# ---------------------------------------------------------------- uv and Python
UV=""
if command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    say "uv missing: installing it (astral.sh installer, into ~/.local/bin)"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    UV="$HOME/.local/bin/uv"
fi
say "uv: $UV ($("$UV" --version))"

if ! PY="$("$UV" python find --managed-python "$PYTHON_VERSION" 2>/dev/null)"; then
    say "Python $PYTHON_VERSION missing: installing it with uv"
    "$UV" python install "$PYTHON_VERSION"
    PY="$("$UV" python find --managed-python "$PYTHON_VERSION")"
fi
say "python: $("$PY" -c 'import platform; print(platform.python_version())')"

# ---------------------------------------------------------------- layout
mkdir -p "$EDGE" "$UV_CACHE_DIR" "$HF_HOME"
if [ ! -d "$EDGE/repo.git" ]; then
    git init -q --bare "$EDGE/repo.git"
    say "created repo.git"
fi
if [ ! -d "$EDGE/src/.git" ]; then
    git clone -q --no-checkout "$EDGE/repo.git" "$EDGE/src" 2>/dev/null
    say "created src/"
fi

# ---------------------------------------------------------------- reachy-mini SDK + daemon
DPY="$EDGE/daemon/bin/python"
if [ ! -x "$DPY" ]; then
    "$UV" venv -q --python "$PY" "$EDGE/daemon"
    say "created daemon venv"
fi
installed="$("$DPY" -c 'import importlib.metadata as m; print(m.version("reachy-mini"))' 2>/dev/null || true)"
if [ "$installed" != "$REACHY_MINI_VERSION" ]; then
    say "installing reachy-mini==$REACHY_MINI_VERSION (was: ${installed:-none})"
    "$UV" pip install -q --python "$DPY" "reachy-mini==$REACHY_MINI_VERSION"
fi

# Validate: the SDK imports, the daemon entry point exists, the media stack (GStreamer through
# PyGObject) initialises.
"$DPY" - "$REACHY_MINI_VERSION" <<'PY'
import importlib.metadata as m
import sys

want = sys.argv[1]
have = m.version("reachy-mini")
if have != want:
    sys.exit(f"reachy-mini {have} installed, {want} required")
import reachy_mini  # noqa: F401
from reachy_mini.media import gstreamer_utils  # noqa: F401  (sets up gi + GStreamer)
from gi.repository import Gst

Gst.init(None)
print(f"bootstrap: reachy-mini {have}, GStreamer {Gst.version_string().split()[-1]}")
PY
if [ ! -x "$EDGE/daemon/bin/reachy-mini-daemon" ]; then
    say "reachy-mini-daemon entry point missing in $EDGE/daemon/bin"
    exit 1
fi

# ---------------------------------------------------------------- robot
robot="none"
for dev in /dev/cu.usbmodem* /dev/ttyACM*; do
    if [ -e "$dev" ]; then robot="$dev"; break; fi
done

printf 'edge-host ready: os=%s arch=%s python=%s reachy_mini=%s robot=%s\n' \
    "$(uname -s)" "$(uname -m)" "$("$DPY" -c 'import platform; print(platform.python_version())')" \
    "$REACHY_MINI_VERSION" "$robot"
