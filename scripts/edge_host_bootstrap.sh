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
#   ~/assistant-edge/hf/         HF_HOME for the daemon and the agent (Pollen's emotions dataset)
#   ~/assistant-edge/Reachy Edge.app
#                                macOS only: the app that owns the robot's microphone and camera
#                                permission (see below); built and signed once
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

# ---------------------------------------------------------------- Pollen's recorded moves
# The robot's emotions are Pollen's recorded moves (a Hugging Face dataset). The daemon and the
# agent run offline (HF_HUB_OFFLINE=1), so it is downloaded here, once, into $HF_HOME.
EMOTIONS_DATASET="pollen-robotics/reachy-mini-emotions-library"
emotions_count() {
    "$DPY" - "$EMOTIONS_DATASET" <<'PY' 2>/dev/null
import sys
from reachy_mini.motion.recorded_move import RecordedMoves
print(len(RecordedMoves(sys.argv[1]).list_moves()))
PY
}
if ! emotions="$(emotions_count)" || [ -z "$emotions" ] || [ "$emotions" = 0 ]; then
    say "downloading $EMOTIONS_DATASET into $HF_HOME (once)"
    HF_HUB_OFFLINE=0 HF_HUB_DISABLE_PROGRESS_BARS=1 "$DPY" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], repo_type="dataset")' "$EMOTIONS_DATASET"
    emotions="$(emotions_count)" || emotions=0
fi
say "emotions: $EMOTIONS_DATASET, ${emotions:-0} moves cached"

# ---------------------------------------------------------------- Reachy Edge.app (macOS)
# macOS grants microphone and camera access to the RESPONSIBLE process of whatever opens them.
# The daemon and the reachy edge agent run inside this app (scripts/edge_app_run.sh starts
# them as a LaunchAgent in the logged-in user's session), so the app is responsible: macOS
# asks once "Reachy Edge would like to access the microphone / camera", and the grant belongs
# to the app, not to the shared uv Python (which needs no permission at all).
# The app's executable is a tiny launcher compiled here: it starts
# <edge dir>/src/.venv-assistant/bin/python with its own arguments as a CHILD process (exec
# would make Python the responsible process again), passes signals on and exits with the
# child's status. Our Python code lives outside the bundle, so code syncs never touch the app.
# It is signed (ad hoc) ONCE: every re-sign changes its code hash and macOS would ask again,
# so it is rebuilt only when the launcher source or Info.plist below changes.
app="none"
if [ "$(uname -s)" = "Darwin" ]; then
    APP="$EDGE/Reachy Edge.app"
    LAUNCHER_C='#include <errno.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], python[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || realpath(exe, real) == NULL) {
        fprintf(stderr, "reachy-edge: cannot find my own path\n");
        return 70;
    }
    /* <edge>/Reachy Edge.app/Contents/MacOS/reachy-edge -> <edge> */
    for (int i = 0; i < 4; i++) {
        char *slash = strrchr(real, (int)0x2f);
        if (slash == NULL) return 70;
        *slash = 0;
    }
    snprintf(python, sizeof python, "%s/src/.venv-assistant/bin/python", real);
    char **args = calloc((size_t)argc + 1, sizeof *args);
    if (args == NULL) return 70;
    args[0] = python;
    for (int i = 1; i < argc; i++) args[i] = argv[i];
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = forward;
    int sigs[] = {SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1, SIGUSR2};
    for (unsigned i = 0; i < sizeof sigs / sizeof sigs[0]; i++) sigaction(sigs[i], &sa, NULL);
    int rc = posix_spawn(&child, python, NULL, NULL, args, environ);
    if (rc != 0) {
        fprintf(stderr, "reachy-edge: cannot start %s: %s\n", python, strerror(rc));
        return 71;
    }
    int status;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return 70;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 70;
}
'
    INFO_PLIST='<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>com.assistant.reachy-edge</string>
    <key>CFBundleName</key><string>Reachy Edge</string>
    <key>CFBundleDisplayName</key><string>Reachy Edge</string>
    <key>CFBundleExecutable</key><string>reachy-edge</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>Reachy Mini robot: voice and camera</string>
    <key>NSCameraUsageDescription</key><string>Reachy Mini robot: voice and camera</string>
</dict>
</plist>
'
    want="$(printf '%s\n%s' "$LAUNCHER_C" "$INFO_PLIST" | shasum -a 256 | cut -d' ' -f1)"
    have="$(cat "$APP/Contents/Resources/build.sha256" 2>/dev/null || true)"
    if [ "$have" = "$want" ] && codesign --verify "$APP" 2>/dev/null; then
        app="kept"
    else
        if ! xcrun --sdk macosx --find clang >/dev/null 2>&1; then
            say "Reachy Edge.app needs the Xcode command-line tools (xcode-select --install)"
            exit 1
        fi
        say "building Reachy Edge.app (macOS will ask once for microphone and camera access)"
        tmp="$(mktemp -d)"
        mkdir -p "$tmp/Reachy Edge.app/Contents/MacOS" "$tmp/Reachy Edge.app/Contents/Resources"
        printf '%s' "$LAUNCHER_C" > "$tmp/launcher.c"
        xcrun --sdk macosx clang -O2 -Wall -Werror -mmacosx-version-min=12.0 -o \
            "$tmp/Reachy Edge.app/Contents/MacOS/reachy-edge" "$tmp/launcher.c"
        printf '%s' "$INFO_PLIST" > "$tmp/Reachy Edge.app/Contents/Info.plist"
        plutil -lint -s "$tmp/Reachy Edge.app/Contents/Info.plist"
        printf '%s\n' "$want" > "$tmp/Reachy Edge.app/Contents/Resources/build.sha256"
        codesign -s - --force --identifier com.assistant.reachy-edge "$tmp/Reachy Edge.app"
        rm -rf "$APP"
        mv "$tmp/Reachy Edge.app" "$APP"
        rm -rf "$tmp"
        app="built"
    fi
    codesign --verify "$APP"
    say "Reachy Edge.app: $app ($(codesign -dv "$APP" 2>&1 | grep -E '^Identifier=' || true))"
fi

# ---------------------------------------------------------------- robot
robot="none"
for dev in /dev/cu.usbmodem* /dev/ttyACM*; do
    if [ -e "$dev" ]; then robot="$dev"; break; fi
done

printf 'edge-host ready: os=%s arch=%s python=%s reachy_mini=%s robot=%s app=%s emotions=%s\n' \
    "$(uname -s)" "$(uname -m)" "$("$DPY" -c 'import platform; print(platform.python_version())')" \
    "$REACHY_MINI_VERSION" "$robot" "$app" "${emotions:-0}"
