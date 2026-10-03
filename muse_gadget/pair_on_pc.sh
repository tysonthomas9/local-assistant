#!/usr/bin/env bash
# Pair the Muse gadget once, on this PC, then move its state to the Mac.
#
# The SDK pairs over Bluetooth LE through Linux BlueZ, which the Mac's Podman
# VM can't reach, so pairing runs here in Docker with the host's D-Bus system
# socket. The gadget's identity and pairing are plain files, so afterwards
# they're copied to reachy-mac:~/assistant-edge/muse-state (0700/0600) and
# deleted from this PC: the gadget only ever runs in one place.
#
# The SDK token is read from a file (--token-file) or a hidden prompt, never
# from the command line, and it is never printed or logged.
#
# Usage: muse_gadget/pair_on_pc.sh [--token-file FILE] [--timeout SECONDS] [--force]
#        muse_gadget/pair_on_pc.sh --check           build, then test BlueZ and SSH access only
#        muse_gadget/pair_on_pc.sh --copy-only DIR   retry copying a paired state dir to the Mac
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${MUSE_GADGET_IMAGE:-localhost/muse-gadget:latest}"
MAC="reachy-mac"
DBUS_SOCKET="/run/dbus/system_bus_socket"
TOKEN_RE='^mgst_[A-Za-z0-9_-]{42}[AEIMQUYcgkosw048]$'
STATE_FILES=(identity.json pairing.json sdk_token)

token_file="" timeout=600 force=0 check=0 copy_only=""
while [ $# -gt 0 ]; do
    case "$1" in
        --token-file) token_file="${2:?--token-file needs a path}"; shift 2 ;;
        --timeout) timeout="${2:?--timeout needs seconds}"; shift 2 ;;
        --force) force=1; shift ;;
        --check) check=1; shift ;;
        --copy-only) copy_only="${2:?--copy-only needs a directory}"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

say() { printf '[pair] %s\n' "$*"; }
die() { printf '[pair] error: %s\n' "$*" >&2; exit 1; }

# Runs on the Mac. $HOME expands there, so no Mac path is written here.
# shellcheck disable=SC2016  # $HOME must expand on the Mac
mac_state_status() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$MAC" \
        'd="$HOME/assistant-edge/muse-state"; if [ -s "$d/pairing.json" ]; then echo paired; elif [ -d "$d" ]; then echo unpaired; else echo none; fi'
}

wipe_dir() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    find "$dir" -type f -exec shred -u {} + 2>/dev/null || find "$dir" -type f -delete
    rm -rf "$dir"
}

copy_to_mac() {
    local dir="$1"
    [ -s "$dir/identity.json" ] && [ -s "$dir/pairing.json" ] || die "$dir has no pairing to copy"
    local files=()
    for f in "${STATE_FILES[@]}"; do [ -s "$dir/$f" ] && files+=("$f"); done
    # shellcheck disable=SC2016  # $HOME must expand on the Mac
    say "copying the gadget state to $MAC (files: ${files[*]})"
    tar -C "$dir" -cf - "${files[@]}" | ssh -o BatchMode=yes "$MAC" '
        set -e; umask 077
        d="$HOME/assistant-edge/muse-state"
        mkdir -p "$d"; chmod 700 "$d"
        tar -C "$d" -xf -
        chmod 600 "$d"/*
        test -s "$d/pairing.json"'
    [ "$(mac_state_status)" = paired ] || die "the copy didn't arrive; state kept in $dir, retry with --copy-only $dir"
    say "the Mac has the pairing; deleting it from this PC"
    wipe_dir "$dir"
}

if [ -n "$copy_only" ]; then
    copy_to_mac "$copy_only"
    say "done. Start the gadget with: muse_gadget/mac_gadget.sh start"
    exit 0
fi

command -v docker >/dev/null || die "docker is not installed"
[ -S "$DBUS_SOCKET" ] || die "no D-Bus system socket at $DBUS_SOCKET"
compgen -G "/sys/class/bluetooth/hci*" >/dev/null || die "no Bluetooth adapter on this PC"

say "building $IMAGE (linux/amd64)"
docker build -q -f "$HERE/Containerfile" -t "$IMAGE" "$HERE" >/dev/null

# Docker's default AppArmor profile blocks all D-Bus traffic, so the pairing
# container (only this short-lived one) runs without it; it stays non-root
# with every capability dropped and no privilege escalation.
DOCKER_BLUEZ=(
    --user "$(id -u):$(id -g)"
    --security-opt apparmor=unconfined
    --security-opt no-new-privileges
    --cap-drop ALL
    --hostname reachy-mini
    -v "$DBUS_SOCKET:$DBUS_SOCKET"
    -e "DBUS_SYSTEM_BUS_ADDRESS=unix:path=$DBUS_SOCKET"
)

if [ "$check" = 1 ]; then
    say "checking BlueZ from inside the container"
    docker run --rm "${DOCKER_BLUEZ[@]}" --entrypoint python "$IMAGE" -c '
import dbus
bus = dbus.SystemBus()
om = dbus.Interface(bus.get_object("org.bluez", "/"), "org.freedesktop.DBus.ObjectManager")
objs = om.GetManagedObjects()
adapters = [p for p, i in objs.items() if "org.bluez.GattManager1" in i and "org.bluez.LEAdvertisingManager1" in i]
assert adapters, "no BlueZ adapter with GATT server and LE advertising"
print("[pair] BlueZ adapter with GATT + LE advertising: ok (%d)" % len(adapters))'
    say "checking SSH to $MAC: state $(mac_state_status)"
    say "check passed: ready to pair once you have the SDK token"
    exit 0
fi

mac_status="$(mac_state_status)" || die "can't reach $MAC over SSH"
if [ "$mac_status" = paired ] && [ "$force" != 1 ]; then
    die "$MAC already has a paired gadget. Pass --force to pair a new one and replace it."
fi

# The token: from a file or a hidden prompt; never an argument.
if [ -n "$token_file" ]; then
    [ -r "$token_file" ] || die "can't read $token_file"
    token="$(tr -d '[:space:]' < "$token_file")"
else
    read -r -s -p "[pair] paste the SDK token from gadgets.muse.ai (hidden): " token; echo
    token="$(printf '%s' "$token" | tr -d '[:space:]')"
fi
[[ "$token" =~ $TOKEN_RE ]] || die "that is not a valid SDK token; copy it again from gadgets.muse.ai"

STATE="$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/muse-pair.XXXXXX")"
chmod 700 "$STATE"
paired=0
cleanup() { [ "$paired" = 1 ] || wipe_dir "$STATE"; }
trap cleanup EXIT
( umask 077; printf '%s\n' "$token" > "$STATE/sdk_token" )
unset token

say "opening Bluetooth setup for $((timeout / 60)) minutes"
say "on your phone: Muse app > Developer mode on > add a device > pick the MuseGadget… name shown below"
tty_flag=(); [ -t 0 ] && tty_flag=(-t)
if ! docker run --rm -i "${tty_flag[@]}" --name muse-gadget-pair "${DOCKER_BLUEZ[@]}" \
        -v "$STATE:/state" "$IMAGE" pair --timeout "$timeout"; then
    die "pairing did not complete; nothing was kept"
fi
[ -s "$STATE/pairing.json" ] || die "pairing did not save credentials; nothing was kept"
paired=1
copy_to_mac "$STATE"
paired=0
say "done. Start the gadget on the Mac with: muse_gadget/mac_gadget.sh start"
