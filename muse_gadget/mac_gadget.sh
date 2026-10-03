#!/usr/bin/env bash
# Run the Muse gadget container on the Mac, in Podman. Run this from the PC.
#
#   muse_gadget/mac_gadget.sh start    start podman-machine-default if it is stopped
#                                      (and remember that we started it), build the
#                                      arm64 image on the Mac if the sources changed,
#                                      then run the gadget with its /turn bridge on
#                                      the Mac's 127.0.0.1:48080 only
#   muse_gadget/mac_gadget.sh stop     remove the gadget container; stop the Podman
#                                      machine only if `start` started it
#   muse_gadget/mac_gadget.sh status   machine, container, bridge health, listeners
#
# Idempotent. Never changes the Podman machine's settings and never touches
# other containers or images. The gadget doesn't use the robot (no daemon,
# motors, mic, speaker or camera), so it doesn't take the hw-run lock.
# State: reachy-mac:~/assistant-edge/muse-state (from pair_on_pc.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC="reachy-mac"
PORT="${MUSE_BRIDGE_PORT:-48080}"
action="${1:-status}"
case "$PORT" in ''|*[!0-9]*) echo "MUSE_BRIDGE_PORT must be a number" >&2; exit 2 ;; esac

say() { printf '[mac-gadget] %s\n' "$*"; }

# The build context, byte-for-byte reproducible so its hash names the sources.
context_tar() {
    tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
        --exclude=__pycache__ --exclude=.pytest_cache \
        -C "$HERE" -cf - Containerfile .containerignore gadget tests
}

# REMOTE_SCRIPT runs on the Mac (uploaded once per call, so stdin stays free
# for the build context). $HOME expands there, so no Mac path or account name
# is written in this repo.
REMOTE_SCRIPT="$(cat <<'REMOTE'
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
MACHINE=podman-machine-default
NAME=muse-gadget
IMAGE=localhost/muse-gadget:latest
BASE="$HOME/assistant-edge/muse-gadget"
STATE="$HOME/assistant-edge/muse-state"
MARK="$BASE/started-machine"
action="$1"; shift

machine_state() { podman machine inspect "$MACHINE" --format '{{.State}}' 2>/dev/null || echo missing; }
container_state() { podman container inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null || echo none; }

case "$action" in
  machine-up)
    mkdir -p "$BASE"
    st="$(machine_state)"
    [ "$st" != missing ] || { echo "no $MACHINE on the Mac" >&2; exit 1; }
    if [ "$st" != running ]; then
      podman machine start "$MACHINE" >/dev/null
      : > "$MARK"
      echo "started $MACHINE (it was $st)"
    else
      [ -e "$MARK" ] && echo "$MACHINE already running (started by us earlier)" || echo "$MACHINE already running (left as is)"
    fi
    ;;
  image-hash)
    podman image inspect "$IMAGE" --format '{{index .Config.Labels "muse.context"}}' 2>/dev/null || true
    ;;
  build)
    hash="$1"
    rm -rf "$BASE/build"; mkdir -p "$BASE/build"
    tar -C "$BASE/build" -xf -
    podman build -q --label "muse.context=$hash" -f "$BASE/build/Containerfile" -t "$IMAGE" "$BASE/build" >/dev/null
    rm -rf "$BASE/build"
    echo "built $IMAGE ($(podman image inspect "$IMAGE" --format '{{.Architecture}}'))"
    ;;
  run)
    port="$1"
    mkdir -p "$STATE"; chmod 700 "$STATE"
    want="$(podman image inspect "$IMAGE" --format '{{.Id}}')"
    if [ "$(container_state)" = running ] && [ "$(podman container inspect "$NAME" --format '{{.Image}}')" = "$want" ]; then
      echo "$NAME already running"
    else
      podman rm -f -t 10 "$NAME" >/dev/null 2>&1 || true
      if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "port $port is already in use on the Mac (another bridge?); not starting" >&2
        lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print "  " $1, $9}' >&2
        exit 1
      fi
      podman run -d --name "$NAME" --label muse.gadget=1 --hostname reachy-mini \
        --userns keep-id:uid=10001,gid=10001 \
        --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp \
        -v "$STATE:/state" \
        -p "127.0.0.1:$port:48080" \
        "$IMAGE" run >/dev/null
      echo "started $NAME"
    fi
    for _ in $(seq 1 40); do
      curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
      sleep 0.5
    done
    printf 'health: '; curl -fsS "http://127.0.0.1:$port/health" || { echo "bridge not answering"; podman logs --tail 20 "$NAME" >&2; exit 1; }
    echo
    ;;
  down)
    if [ "$(machine_state)" = running ] && [ "$(container_state)" != none ]; then
      podman rm -f -t 10 "$NAME" >/dev/null
      echo "removed $NAME"
    else
      echo "$NAME not present"
    fi
    if [ -e "$MARK" ]; then
      others=0
      [ "$(machine_state)" = running ] && others="$(podman ps -q | wc -l | tr -d ' ')"
      if [ "${others:-0}" != 0 ]; then
        echo "left $MACHINE running: $others other container(s) started since; run stop again later"
      else
        [ "$(machine_state)" = running ] && podman machine stop "$MACHINE" >/dev/null
        rm -f "$MARK"
        echo "stopped $MACHINE (we had started it)"
      fi
    else
      echo "left $MACHINE as it was ($(machine_state))"
    fi
    ;;
  status)
    port="$1"
    echo "machine: $(machine_state)$([ -e "$MARK" ] && echo ', started by mac_gadget.sh')"
    if [ "$(machine_state)" = running ]; then
      echo "container: $(container_state)"
      printf 'health: '; curl -fsS --max-time 3 "http://127.0.0.1:$port/health" 2>/dev/null || printf 'no answer'
      echo
    fi
    echo "listening on port $port:"
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print "  " $1, $9}' || true
    ;;
  *) echo "unknown remote action $action" >&2; exit 2 ;;
esac
REMOTE
)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$MAC")
# shellcheck disable=SC2016  # $HOME must expand on the Mac
REMOTE_PATH='"$HOME/assistant-edge/muse-gadget/remote.sh"'

# shellcheck disable=SC2016  # $HOME must expand on the Mac
install_remote() {
    printf '%s\n' "$REMOTE_SCRIPT" | "${SSH[@]}" \
        'umask 077; mkdir -p "$HOME/assistant-edge/muse-gadget" && cat > "$HOME/assistant-edge/muse-gadget/remote.sh"'
}

# Arguments are plain words (action, port, hash), safe to pass through ssh.
remote() { "${SSH[@]}" "/bin/bash $REMOTE_PATH $*"; }

install_remote
case "$action" in
    start)
        remote machine-up | sed 's/^/[mac-gadget] /'
        hash="$(context_tar | sha256sum | cut -c1-16)"
        if [ "$(remote image-hash)" = "$hash" ]; then
            say "image is up to date ($hash)"
        else
            say "building the arm64 image on the Mac ($hash)"
            context_tar | remote build "$hash" | sed 's/^/[mac-gadget] /'
        fi
        remote run "$PORT" | sed 's/^/[mac-gadget] /'
        ;;
    stop) remote down | sed 's/^/[mac-gadget] /' ;;
    status) remote status "$PORT" | sed 's/^/[mac-gadget] /' ;;
    *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
