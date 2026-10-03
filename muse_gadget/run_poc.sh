#!/usr/bin/env bash
# Talk to Muse through the robot: Pollen's conversation app + MuseHandler on the Mac.
# Run on the PC:
#
#   muse_gadget/run_poc.sh                  real Muse gadget (muse_gadget/mac_gadget.sh start|stop)
#   muse_gadget/run_poc.sh --fake-bridge    echo bridge ("You said ...") instead of Muse
#   options: --duration SECONDS (stop by itself), --lock-timeout SECONDS (default: wait forever),
#            --mic-log SECONDS (log the mic level and VAD score that often),
#            --log-transcripts (debugging: show each turn's text on this terminal; the log files
#            on the PC stay redacted), --tts kokoro|say (reply voice engine, default kokoro),
#            --voice NAME (e.g. af_heart, am_michael), --volume N (robot speaker 0-100, default 25;
#            the daemon plays a short test sound when it's set), -- <extra conversation-app args>
#
# In order: take the hw-run lock on reachy-mac -> sync MuseHandler + install the pinned app into
# ~/assistant-edge/muse-app -> start the bridge (Muse gadget container, or the fake one) -> start
# the daemon, then the app, both inside Reachy Edge.app (edge_app_run.sh). Ctrl-C to stop.
# On exit, error or Ctrl-C: stop the app, robot goto_sleep, motors off, stop the daemon and every
# com.assistant.reachy-edge.<this run>.* job, stop the bridge, release the lock.
# It never deletes another run's lock (it waits) and never stops a daemon it didn't start; it
# sets the speaker volume to --volume (logged before and after). Output from the Mac shows its
# home directory as ~; transcript text (`content=...`) is redacted in the logs in ~/.local/state/muse-poc/.
set -uo pipefail

HOST=reachy-mac
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${MUSE_POC_LOGDIR:-${XDG_STATE_HOME:-$HOME/.local/state}/muse-poc}"   # outside the repo
fake=0; duration=0; lock_timeout=0; mic_log=0; log_transcripts=0; tts=kokoro; voice=; volume=25; app_args=()
while [ $# -gt 0 ]; do
    case $1 in
        --fake-bridge) fake=1; shift ;;
        --duration) duration=$2; shift 2 ;;
        --lock-timeout) lock_timeout=$2; shift 2 ;;
        --mic-log) mic_log=$2; shift 2 ;;
        --log-transcripts) log_transcripts=1; shift ;;
        --tts) tts=$2; shift 2 ;;
        --voice) voice=$2; shift 2 ;;
        --volume) volume=$2; shift 2 ;;
        --) shift; app_args=("$@"); break ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "run_poc: unknown option $1" >&2; exit 2 ;;
    esac
done
case $tts in kokoro|say) ;; *) echo "run_poc: --tts must be kokoro or say" >&2; exit 2 ;; esac
case $volume in ''|*[!0-9]*) echo "run_poc: --volume must be 0-100" >&2; exit 2 ;; esac
[ "$volume" -le 100 ] || { echo "run_poc: --volume must be 0-100" >&2; exit 2; }
case $voice in *[!A-Za-z0-9_]*) echo "run_poc: unsupported voice name: $voice" >&2; exit 2 ;; esac
for a in "${app_args[@]}"; do
    case $a in *[!A-Za-z0-9._=-]*) echo "run_poc: unsupported app argument: $a" >&2; exit 2 ;; esac
done

RUN="muse-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
MID="$(sha256sum /etc/machine-id 2>/dev/null | cut -c1-12)"
[ -n "$MID" ] || MID="$(sha256sum /var/lib/dbus/machine-id | cut -c1-12)"
OWNER="run=$RUN machine=$MID pid=$$ since=$(date +%s)"
mkdir -p "$LOGDIR"
log() { printf '[run_poc %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
scrub() { sed -u -e 's#/Users/[^/ ]*#~#g' -e 's#/home/[^/ ]*#~#g'; }
# The app logs AdditionalOutputs as `role=<r> content=<text>`. MuseHandler only emits those with
# --log-transcripts, but whatever reaches a log file on the PC goes through this first.
redact() { sed -u -e 's/\(role=[A-Za-z_]*\) content=.*/\1 content=<redacted>/'; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" "$@"; }
rsh_sh() { rsh "sh -s --$(printf ' %q' "$@")"; }   # rsh_sh <args...> < script: args keep their spaces

# ------------------------------------------------------------------ hw-run lock (see the spec)
lock_try() {
    rsh_sh "$OWNER" <<'EOF'
L="$HOME/assistant-edge/hw-run.lock"
mkdir -p "$HOME/assistant-edge"
if mkdir "$L" 2>/dev/null; then printf '%s\n' "$1" > "$L/owner"; echo ACQUIRED
else echo HELD; cat "$L/owner" 2>/dev/null; fi
EOF
}
take_lock() {
    local started out held hpid hmach waited_msg=0
    started=$(date +%s)
    while :; do
        if ! out="$(lock_try)"; then log "cannot reach $HOST"; return 1; fi
        if [ "$(printf '%s\n' "$out" | head -1)" = ACQUIRED ]; then log "hw-run lock taken ($RUN)"; return 0; fi
        held="$(printf '%s\n' "$out" | sed -n 2p)"
        hmach="$(printf '%s\n' "$held" | sed -n 's/.*machine=\([0-9a-f]*\).*/\1/p')"
        hpid="$(printf '%s\n' "$held" | sed -n 's/.*pid=\([0-9]*\).*/\1/p')"
        # Stale: its owner ran on this PC and is dead. Remove it only if it is still that owner.
        if [ -n "$held" ] && [ "$hmach" = "$MID" ] && [ -n "$hpid" ] && ! kill -0 "$hpid" 2>/dev/null; then
            log "taking over a stale lock (owner pid $hpid on this PC is gone)"
            rsh_sh "$held" <<'EOF'
L="$HOME/assistant-edge/hw-run.lock"
[ "$(cat "$L/owner" 2>/dev/null)" = "$1" ] && rm -rf "$L"
EOF
            continue
        fi
        if [ "$waited_msg" = 0 ] || [ $(( $(date +%s) - waited_msg )) -ge 60 ]; then
            log "robot busy: hw-run lock held by ${held:-<being written>}; waiting"
            waited_msg=$(date +%s)
        fi
        if [ "$lock_timeout" -gt 0 ] && [ $(( $(date +%s) - started )) -ge "$lock_timeout" ]; then
            log "gave up waiting for the hw-run lock"; return 1
        fi
        sleep 10
    done
}
release_lock() {
    local out
    out="$(rsh_sh "$RUN" <<'EOF'
L="$HOME/assistant-edge/hw-run.lock"
if grep -q "^run=$1 " "$L/owner" 2>/dev/null; then rm -rf "$L"; echo RELEASED; else echo NOT-OURS; fi
EOF
)"
    log "hw-run lock: ${out:-unreachable}"
}

# ------------------------------------------------------------------ jobs inside Reachy Edge.app
pids=()           # local ssh processes we started
daemon_started=0; bridge_started=0; locked=0; lock_tried=0
start_job() {     # start_job <name> <remote args for edge_app_run.sh, already quoted for sh>
    local name=$1 args=$2
    if [ "$log_transcripts" = 1 ]; then   # text on the terminal only, never in the log file
        rsh "ASSISTANT_TEST_RUN=$RUN sh \"\$HOME/assistant-edge/src/scripts/edge_app_run.sh\" $name $args" \
            < /dev/null 2>&1 | scrub | tee >(redact >> "$LOGDIR/$RUN.$name.log") | sed -u "s/^/[$name] /" &
    else
        rsh "ASSISTANT_TEST_RUN=$RUN sh \"\$HOME/assistant-edge/src/scripts/edge_app_run.sh\" $name $args" \
            < /dev/null 2>&1 | scrub | redact | tee -a "$LOGDIR/$RUN.$name.log" | sed -u "s/^/[$name] /" &
    fi
    pids+=($!)
}
api() {           # api <GET|POST> <path>: the daemon's REST API on the Mac's loopback
    rsh "curl -s -m 10 -X $1 http://127.0.0.1:8000/api$2"
}
stop_jobs() {     # stop_jobs <name> <signal>: signal our jobs of that name, wait (20 s), then unload them
    rsh_sh "com.assistant.reachy-edge.$RUN.$1." "$2" <<'EOF'
uid=$(id -u)
labels=$(launchctl list | awk -v p="$1" 'index($3, p) == 1 {print $3}')
for l in $labels; do launchctl kill "$2" "gui/$uid/$l" 2>/dev/null; done
for _ in $(seq 1 100); do
    left=""
    for l in $labels; do
        launchctl print "gui/$uid/$l" 2>/dev/null | grep -q '^	state = running' && left="$left $l"
    done
    [ -z "$left" ] && break
    sleep 0.2
done
for l in $left; do launchctl kill SIGKILL "gui/$uid/$l" 2>/dev/null; done
for l in $labels; do launchctl bootout "gui/$uid/$l" 2>/dev/null; rm -rf "$HOME/assistant-edge/run/$l"; done
[ -n "$labels" ] && echo "stopped:$(printf ' %s' $labels | sed 's/com\.assistant\.reachy-edge\.//g')"
EOF
}

M='"$HOME/assistant-edge/muse-app"'
cleaning=0
cleanup() {
    [ "$cleaning" = 1 ] && return
    cleaning=1
    trap '' INT TERM HUP
    if [ "$locked" != 1 ]; then   # stopped before or while taking the lock: nothing else started
        # release_lock only removes a lock that names this run, so it's safe if we never got it.
        [ "$lock_tried" = 1 ] && release_lock
        return 0
    fi
    log "cleaning up"
    stop_jobs app SIGINT | scrub   # the app shuts down cleanly on SIGINT (KeyboardInterrupt)
    if [ "$daemon_started" = 1 ]; then
        log "robot: goto_sleep, then motors off"
        api POST /move/play/goto_sleep >/dev/null
        for _ in $(seq 1 40); do
            [ "$(api GET /move/running)" = "[]" ] && break
            sleep 0.25
        done
        api POST /motors/set_mode/disabled; echo
        log "motors: $(api GET /motors/status)"
        stop_jobs daemon SIGTERM | scrub
    fi
    if [ "$bridge_started" = 1 ]; then
        if [ "$fake" = 1 ]; then
            rsh "f=$M/run/fake-bridge.$RUN.pid; p=\$(cat \"\$f\" 2>/dev/null); \
[ -n \"\$p\" ] && ps -p \"\$p\" -o command= | grep -q fake_bridge.py && kill \"\$p\"; rm -f \"\$f\"; true"
        else
            "$ROOT/mac_gadget.sh" stop
        fi
    fi
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    left="$(rsh "launchctl list | grep -c 'com.assistant.reachy-edge.$RUN\\.' ; true")"
    [ "${left:-0}" = 0 ] || log "WARNING: $left job(s) of this run still loaded on $HOST"
    [ "$locked" = 1 ] && release_lock
    log "done (logs: $LOGDIR/$RUN.*.log)"
}

# Traps first, so a signal at any point (even right after the lock is taken) releases it.
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup; exit 129' HUP
lock_tried=1
take_lock || exit 1
locked=1

# ------------------------------------------------------------------ 1. sync + install the app
log "syncing MuseHandler and installing the pinned app on $HOST"
(cd "$ROOT/mac" && tar -cf - *.py install.sh app-constraints.txt kokoro-constraints.txt) \
    | rsh "mkdir -p $M/mac $M/run && tar -xf - -C $M/mac" || exit 1
rsh 'bash -s' < "$ROOT/mac/install.sh" 2>&1 | scrub
[ "${PIPESTATUS[0]}" = 0 ] || { log "install failed"; exit 1; }

# ------------------------------------------------------------------ 2. the bridge
if [ "$(rsh 'curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:48080/health')" != 000 ]; then
    log "something already answers on the Mac's 127.0.0.1:48080; not starting a bridge over it"; exit 1
fi
if [ "$fake" = 1 ]; then
    log "starting the fake echo bridge"
    rsh "cd $M/mac && echo \$\$ > ../run/fake-bridge.$RUN.pid && exec ../app/.venv/bin/python fake_bridge.py" \
        < /dev/null 2>&1 | scrub | sed -u 's/^/[bridge] /' &
    pids+=($!)
else
    [ -x "$ROOT/mac_gadget.sh" ] || { log "muse_gadget/mac_gadget.sh missing (task 1); try --fake-bridge"; exit 1; }
    log "starting the Muse gadget (mac_gadget.sh start)"
    "$ROOT/mac_gadget.sh" start || { bridge_started=1; exit 1; }
fi
bridge_started=1
for _ in $(seq 1 60); do
    health="$(rsh 'curl -s -m 3 http://127.0.0.1:48080/health')" && [ -n "$health" ] && break
    sleep 1
done
[ -n "${health:-}" ] || { log "bridge not answering on the Mac's 127.0.0.1:48080"; exit 1; }
log "bridge health: $health"

# ------------------------------------------------------------------ 3. daemon, then the app
if [ "$(rsh 'curl -s -m 3 -o /dev/null -w %{http_code} http://127.0.0.1:8000/api/daemon/status')" != 000 ]; then
    log "a daemon this run didn't start is answering on the Mac; leaving it alone"; exit 1
fi
log "starting the daemon inside Reachy Edge.app"
daemon_started=1
start_job daemon "-e HF_HOME=~/assistant-edge/hf -e HF_HUB_OFFLINE=1 -- $M/mac/exec_python.py \
\"\$HOME/assistant-edge/daemon/bin/python\" $M/mac/run_daemon.py --fastapi-host 127.0.0.1 \
--no-wake-up-on-start --goto-sleep-on-stop --dataset-update-interval 0"
ready=0
for _ in $(seq 1 180); do   # the first start after an install scans GStreamer plugins (slow)
    [ "$(api GET /daemon/status | grep -o '"state":"running"')" = '"state":"running"' ] && { ready=1; break; }
    sleep 1
done
[ "$ready" = 1 ] || { log "daemon did not come up"; exit 1; }
log "daemon running"
vol_now() { api GET /volume/current | sed -n 's/.*"volume":\([0-9]*\).*/\1/p'; }
vol_before="$(vol_now)"
rsh "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '{\"volume\":$volume}' \
http://127.0.0.1:8000/api/volume/set" >/dev/null
log "speaker volume: before ${vol_before:-?}, after $(vol_now) (asked $volume)"

log "starting the conversation app with MuseHandler"
start_job app "-e HF_HOME=~/assistant-edge/muse-app/hf -e HF_HUB_OFFLINE=1 -e MUSE_BRIDGE_URL=http://127.0.0.1:48080 -e MUSE_MIC_LOG=$mic_log -e MUSE_LOG_TRANSCRIPTS=$log_transcripts \
-e MUSE_TTS=$tts -e MUSE_TTS_VOICE=$voice \
-- $M/mac/exec_python.py $M/app/.venv/bin/python $M/mac/run_app.py --no-camera ${app_args[*]:-}"
app_ssh=${pids[-1]}

if [ "$duration" -gt 0 ]; then
    log "running for $duration s (Ctrl-C stops earlier)"
    end=$(( $(date +%s) + duration ))
    while [ "$(date +%s)" -lt "$end" ] && kill -0 "$app_ssh" 2>/dev/null; do sleep 1; done
else
    log "talk to the robot; Ctrl-C stops"
    while kill -0 "$app_ssh" 2>/dev/null; do sleep 1; done
fi
kill -0 "$app_ssh" 2>/dev/null || log "the app exited"
exit 0
