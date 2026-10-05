#!/bin/sh
# Run a Python module of the synced checkout INSIDE "Reachy Edge.app" on a macOS edge host.
#
#   sh ~/assistant-edge/src/scripts/edge_app_run.sh <name> [--stdin] [-e KEY=VALUE]... -- <python args>
#   e.g. ... edge_app_run.sh daemon -e HF_HUB_OFFLINE=1 -- -m assistant_robot_reachy.daemon ...
#
# Why: macOS grants microphone and camera access to the RESPONSIBLE process. Started over
# SSH that would be sshd (never grantable); started as a per-run LaunchAgent in the logged-in
# user's GUI session whose program is the app's launcher, it is "Reachy Edge" (built by
# scripts/edge_host_bootstrap.sh), which keeps the venv Python as its child. So the one-time
# "Reachy Edge would like to access the microphone / camera" grant covers the daemon and the
# reachy edge agent, and Python itself needs no permission.
#
# This script is the job's stand-in for the test runner (it is what runs over SSH):
#   - it loads the LaunchAgent (label com.assistant.reachy-edge.<run>.<name>.<pid>) into
#     gui/<uid>, prints `APP-JOB label=... pid=...` (pid = the job's process group), then the
#     job's output as it is written;
#   - with --stdin, its own stdin is passed to the job (through a FIFO);
#   - SIGTERM/SIGINT/SIGHUP are passed to the job; when the job ends, the agent is unloaded,
#     its run dir removed, and this script exits with the job's exit status.
# A job left behind by a killed runner is unloaded by the sweep (assistant_testing.edge_host).
# `~/` at the start of an -e VALUE means the edge host's home.
set -u

name=${1:?usage: edge_app_run.sh <name> [--stdin] [-e KEY=VALUE]... -- <python args>}
shift
stdin=0
envs=""
while [ $# -gt 0 ]; do
    case $1 in
        --stdin) stdin=1; shift ;;
        -e) envs="$envs$2
"; shift 2 ;;
        --) shift; break ;;
        *) echo "edge_app_run: unknown option $1" >&2; exit 2 ;;
    esac
done

EDGE="$HOME/assistant-edge"
EXE="$EDGE/Reachy Edge.app/Contents/MacOS/reachy-edge"
if [ ! -x "$EXE" ]; then
    echo "edge_app_run: Reachy Edge.app missing: run scripts/edge_host_bootstrap.sh" >&2
    exit 3
fi
uid=$(id -u)
safe() { printf %s "$1" | tr -c 'A-Za-z0-9-' '-'; }
label="com.assistant.reachy-edge.$(safe "${ASSISTANT_TEST_RUN:-manual}").$(safe "$name").$$"
job="gui/$uid/$label"
dir="$EDGE/run/$label"
mkdir -p "$dir"
plist="$dir/job.plist"
log="$dir/out.log"
: > "$log"

plutil -create xml1 "$plist"
plutil -insert Label -string "$label" "$plist"
plutil -insert ProgramArguments -array "$plist"
plutil -insert ProgramArguments -string "$EXE" -append "$plist"
for arg in "$@"; do
    plutil -insert ProgramArguments -string "$arg" -append "$plist"
done
plutil -insert EnvironmentVariables -dictionary "$plist"
plutil -insert EnvironmentVariables.PYTHONUNBUFFERED -string 1 "$plist"
plutil -insert EnvironmentVariables.PATH -string "/usr/bin:/bin:/usr/sbin:/sbin" "$plist"
printf %s "$envs" | while IFS= read -r pair; do
    [ -n "$pair" ] || continue
    key=${pair%%=*}
    value=${pair#*=}
    case $value in "~/"*) value="$HOME/${value#\~/}" ;; esac
    plutil -insert "EnvironmentVariables.$key" -string "$value" "$plist"
done
plutil -insert WorkingDirectory -string "$EDGE/src" "$plist"
plutil -insert StandardOutPath -string "$log" "$plist"
plutil -insert StandardErrorPath -string "$log" "$plist"
plutil -insert RunAtLoad -bool true "$plist"
plutil -insert KeepAlive -bool false "$plist"
plutil -insert ProcessType -string Interactive "$plist"

feeder=""
if [ "$stdin" = 1 ]; then
    mkfifo "$dir/in"
    plutil -insert StandardInPath -string "$dir/in" "$plist"
    exec 3<>"$dir/in"          # held open, so the job never sees EOF before we end
    exec 4<&0                  # a background job's stdin would be /dev/null
    cat <&4 >&3 &
    feeder=$!
fi

# The runner stops a job by signalling this script's whole process group, which also hits
# whatever command the script is running: the job checks below ignore those signals (an
# ignored signal stays ignored across exec), else a signal landing on them would read as "the
# job has ended" and the job would be booted out, its output cut, mid-shutdown.
state() { (trap '' TERM INT HUP; exec launchctl print "$job") 2>/dev/null; }
running() { (trap '' TERM INT HUP; state | grep -q '^	state = running'); }
finish() {
    code=$1
    sleep 0.3
    [ -n "$tailer" ] && kill -KILL "$tailer" 2>/dev/null
    [ -n "$feeder" ] && kill "$feeder" 2>/dev/null
    launchctl bootout "$job" 2>/dev/null
    rm -rf "$dir"
    exit "$code"
}

tailer=""
if ! launchctl bootstrap "gui/$uid" "$plist"; then
    echo "edge_app_run: launchctl bootstrap gui/$uid failed (is this user logged in at the Mac?)" >&2
    finish 4
fi
# The runner stops a job by signalling this script's whole process group: the tail must
# outlive that, to relay the job's shutdown output (it is killed in finish()).
(trap '' TERM INT HUP; exec tail -n +1 -f "$log") &
tailer=$!

pid=""
for _ in $(seq 1 50); do
    pid=$(state | awk '$1 == "pid" && $2 == "=" {print $3; exit}')
    [ -n "$pid" ] && break
    sleep 0.1
done
echo "APP-JOB label=$label pid=${pid:-none}"

stopping=0
stop_job() {
    stopping=1
    launchctl kill "$1" "$job" 2>/dev/null
}
trap 'stop_job SIGTERM' TERM
trap 'stop_job SIGINT' INT
trap 'stop_job SIGHUP' HUP

waited=0
while running; do
    sleep 0.2
    if [ "$stopping" = 1 ]; then
        waited=$((waited + 1))
        [ "$waited" = 100 ] && launchctl kill SIGKILL "$job" 2>/dev/null  # 20 s after TERM
    fi
done
code=$( (trap '' TERM INT HUP; state | awk '$1 == "last" && $2 == "exit" && $3 == "code" {print $5; exit}') )
case $code in
    ''|*[!0-9]*) code=70 ;;
esac
finish "$code"
