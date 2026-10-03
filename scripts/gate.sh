#!/usr/bin/env bash
# The merge gate for the new assistant stack. Run it from anywhere in the repo:
#
#   scripts/gate.sh
#
# Stages:
#   a  clone   git clone HEAD into a temp dir and `uv sync --locked` from scratch into
#              .venv-assistant; FAILS if the sync changed the legacy root .venv
#   b  lint    ruff (check + format), basedpyright, import-linter
#   c  unit    the few unit tests (pure logic) plus the schema snapshot
#   d  real    real-only check: no mocks, mocker, monkeypatching or Fake/Mock/Stub/Dummy
#              identifiers in e2e/ or anywhere in assistant_testing (prints file:line)
#   e  core    e2e features, tier core
#   f  legacy  the whole legacy suite (local_backend/tests, as the README runs it). Gitignored
#              resources are symlinked into the clone from the main checkout. Tests listed in
#              scripts/legacy_stack_tests.txt are deselected only while their requirement (the
#              running stack, SearXNG, ...) is missing. A failing legacy test is rerun ONCE:
#              a pass is reported as FLAKY, a second failure FAILS. SKIP only when the legacy
#              venv is missing.
#   s  sim     e2e features, tier sim: the same features as on the robot, on Pollen's simulated
#              Reachy Mini (its daemon with --sim, MuJoCo, headless) on this PC, with a virtual
#              PipeWire sound card (assistant_testing.sim; .venv-sim is synced first). One
#              sim at a time; sweeps leftovers before and after (leftovers after FAIL)
#              Features with `tier: [sim, hw, models]` run here too (with the models)
#   g  hw      e2e features, tier hw. The robot is used where it is plugged in: on this PC
#              (/dev/ttyACM*) if attached here, else on the edge host reached by the SSH alias in
#              config [test.edge_host] ssh / ASSISTANT_EDGE_HOST (it needs /dev/cu.usbmodem* or
#              /dev/ttyACM* THERE). Prints which host is used; FAILS if neither has the robot.
#              Before and after the features it sweeps leftovers: processes from ~/assistant-edge
#              on that host, tagged test ssh clients/tunnels on this PC and their sshd forwards
#              there. Leftovers after the features FAIL the stage (docs/robot-on-another-machine.md)
#              Features with `tier: [hw, models]` run here (they need the models too; with
#              GATE_NO_MODELS=1 they are left out)
#   h  models  e2e features, tier models without sim or hw: FAILS unless both GPUs, our LLM server
#              with reachy-gemma4 and the speech server are available
#
# The models (before stage s): the speech server (servers/speech: Parakeet STT + Qwen3-TTS) on
# 127.0.0.1:8772. If none is serving there, the gate syncs its venv in the checkout under test
# and starts it on GPU1 (CUDA_VISIBLE_DEVICES=1; it needs 8 GB free there), and stops it after
# stage h. A server it did not start is used and left alone. Failing to start it fails s, g, h.
# The LLM likewise: our own server (scripts/llm_server.sh on 127.0.0.1:8773, GPU0 only) of the
# configured kind (`[llm] server` of the ci profile, ASSISTANT__LLM__SERVER overrides it): vLLM
# by default (servers/vllm/.venv, synced by the launcher; it needs about 21.7 GB free on GPU0),
# Ollama as the fallback (the system model store read-only). It is started unless one serves
# there (it must then be that kind), loaded once and checked to sit entirely on GPU0
# (assistant_testing.llm_server check), and stopped after stage h. A busy GPU0 fails s, g, h
# with the processes on it named. The system Ollama service is not used or changed (the
# launcher only unloads reachy-gemma4 from it). The old assistant (legacy
# stack: run_app.py, speech-to-speech, run_daemon.py) is stopped first if it runs: it holds
# the GPUs and the robot.
# Spoken turns write their timings to $ROOT/artifacts (ASSISTANT_ARTIFACTS_DIR).
#   i  summary PASS/FAIL and time per stage, the sim and robot feature times. Exit 0 = PASS,
#              1 = FAIL, 3 = INCOMPLETE (a stage opted out)
#
# E2E uses only real devices and the real stack (see e2e/features/README.md).
#
# Environment:
#   GATE_FAST=1       test the working tree in place instead of a fresh clone (uncommitted
#                     changes are then included)
#   GATE_ROBOT=both   the robot features on both robots (the default; the full gate)
#   GATE_ROBOT=sim    the simulated robot only: skip stage g (fast iteration)  } each prints a
#   GATE_ROBOT=hw     the physical robot only: skip stage s                    } loud WARNING
#   GATE_NO_HW=1      the same as GATE_ROBOT=sim                               } and exits 3
#   GATE_NO_MODELS=1  skip stage h and the model features of s and g           } (incomplete)
#   LEGACY_PYTHON     python for the legacy suite (default: third_party/speech-to-speech/.venv)
#   LEGACY_APP_DIR    reachy_mini_conversation_app to link (default: from the main checkout)
#
# Gitignored legacy resources are looked up in the main checkout of this repository, this
# checkout, and (for a clone of a clone) the checkout a local `origin` remote points to and
# its main checkout.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
MAIN_CHECKOUT="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
DAEMON_URL="http://127.0.0.1:8000"
LLM_PORT=8773
LLM_URL="http://127.0.0.1:$LLM_PORT"
LLM_MODEL="reachy-gemma4"
LLM_SERVER=""
declare -A LLM_NEEDS_MIB=([vllm]=21800 [ollama]=21000)
declare -A LLM_START_S=([vllm]=600 [ollama]=60)
LLM_PID=""
LLM_STATUS=""
SPEECH_PORT=8772
SPEECH_URL="http://127.0.0.1:$SPEECH_PORT"
SPEECH_GPU=1
SPEECH_MIN_FREE_MIB=8000
SPEECH_PID=""
SPEECH_STATUS=""
unset VIRTUAL_ENV || true
# The uv workspace lives in .venv-assistant; .venv is the legacy root venv (piper, reachy-mini
# SDK and daemon) and the new stack must never touch it.
export UV_PROJECT_ENVIRONMENT=.venv-assistant
export ASSISTANT_ARTIFACTS_DIR="${ASSISTANT_ARTIFACTS_DIR:-$ROOT/artifacts}"

GATE_ROBOT="${GATE_ROBOT:-both}"
if [[ "${GATE_NO_HW:-}" == "1" ]]; then GATE_ROBOT=sim; fi
case "$GATE_ROBOT" in
    sim | hw | both) ;;
    *) printf 'GATE_ROBOT must be sim, hw or both (got %q)\n' "$GATE_ROBOT" >&2; exit 2 ;;
esac

STATE_DIR="$(mktemp -d -t assistant-gate-state.XXXXXX)"
CLONE_PARENT=""
CLONE_DIR=""
cleanup() {
    if declare -F stop_speech_server >/dev/null; then stop_speech_server; fi
    if declare -F stop_llm_server >/dev/null; then stop_llm_server; fi
    rm -rf "$STATE_DIR"
    if [[ -n "$CLONE_PARENT" ]]; then rm -rf "$CLONE_PARENT"; fi
}
trap cleanup EXIT

if [[ -t 1 ]]; then
    BOLD=$'\e[1m' RED=$'\e[31m' GREEN=$'\e[32m' YELLOW=$'\e[33m' RESET=$'\e[0m'
else
    BOLD="" RED="" GREEN="" YELLOW="" RESET=""
fi

WORK="$ROOT"
STAGES=()
declare -A RESULT NOTE TIME
INCOMPLETE=()

note() { printf '%s\n' "$*" >"$STATE_DIR/note"; }
banner() { printf '\n%s==== %s ====%s\n' "$BOLD" "$*" "$RESET"; }
warn_loud() {
    printf '\n%s%s' "$YELLOW" "$BOLD"
    printf '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
    printf '!!! WARNING: %s\n' "$*"
    printf '!!! THE GATE IS NOT COMPLETE.\n'
    printf '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
    printf '%s' "$RESET"
}

# run_stage <id> <title> <function>: exit 0 = PASS, 77 = SKIP (reason in the note), else FAIL.
run_stage() {
    local id="$1" title="$2" fn="$3" rc started=$SECONDS
    STAGES+=("$id")
    banner "$id) $title"
    rm -f "$STATE_DIR/note"
    set +e
    (set -euo pipefail; if [[ -d "$WORK" ]]; then cd "$WORK"; fi; "$fn")
    rc=$?
    set -e
    TIME[$id]=$((SECONDS - started))
    NOTE[$id]="$(cat "$STATE_DIR/note" 2>/dev/null || true)"
    case "$rc" in
        0) RESULT[$id]="PASS" ;;
        77) RESULT[$id]="SKIP" ;;
        *) RESULT[$id]="FAIL"; NOTE[$id]="${NOTE[$id]:-exit code $rc}" ;;
    esac
    printf '%s-> %s %s (%s)%s\n' "$BOLD" "${RESULT[$id]}" "${NOTE[$id]}" "$(duration "${TIME[$id]}")" "$RESET"
}

duration() { printf '%dm%02ds' $(($1 / 60)) $(($1 % 60)); }

# pytest_features <marker expression>: run one tier; "no tests collected" (exit 5) means 0
# features.
pytest_features() {
    local marker="$1" out rc passed
    out="$STATE_DIR/pytest-${marker// /_}.log"
    set +e
    uv run --locked pytest e2e -m "$marker" -v -p no:cacheprovider 2>&1 | tee "$out"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -eq 5 ]]; then
        note "0 features"
        return 0
    fi
    passed="$(grep -E '^=+ .* in [0-9.]+s' "$out" | tail -1 | sed -E 's/^=+ (.*) in [0-9.]+s.*/\1/' || true)"
    note "${passed:-no pytest summary}"
    return "$rc"
}

# ---------------------------------------------------------------- stages

stage_clone() {
    if [[ "${GATE_FAST:-}" == "1" ]]; then
        note "GATE_FAST=1: working tree in place, no fresh clone"
    else
        if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
            printf '%sNote: uncommitted changes are NOT tested (the gate clones HEAD).%s\n' \
                "$YELLOW" "$RESET"
        fi
        local sha
        sha="$(git -C "$ROOT" rev-parse HEAD)"
        git clone -q --no-checkout "$ROOT" "$CLONE_DIR"
        git -C "$CLONE_DIR" checkout -q --detach "$sha"
        cd "$CLONE_DIR"
        note "fresh clone of ${sha:0:10}"
        provision_legacy
    fi
    local before after
    before="$(legacy_venv_fingerprint)"
    uv sync --locked
    after="$(legacy_venv_fingerprint)"
    if [[ "$before" != "$after" ]]; then
        printf '%suv sync changed the legacy .venv (%s -> %s)%s\n' "$RED" "$before" "$after" "$RESET"
        note "uv sync touched the legacy .venv"
        return 1
    fi
    printf 'legacy .venv unchanged by uv sync (%s)\n' "$before"
}

# legacy_venv_fingerprint: hash of the legacy root venv's installed packages and mtimes.
legacy_venv_fingerprint() {
    if [[ ! -d .venv ]]; then
        printf 'absent\n'
        return 0
    fi
    { stat -L -c '%n %Y' .venv .venv/bin .venv/pyvenv.cfg .venv/lib/python*/site-packages
      ls -1 .venv/lib/python*/site-packages; } 2>/dev/null | sha256sum | cut -c1-16
}

stage_lint() {
    uv run --locked ruff check
    uv run --locked ruff format --check
    uv run --locked basedpyright
    uv run --locked lint-imports
    note "ruff, basedpyright and import-linter clean"
}

stage_unit() {
    local out="$STATE_DIR/unit.log"
    local rc
    set +e
    uv run --locked pytest packages tests -q -p no:cacheprovider 2>&1 | tee "$out"
    rc=${PIPESTATUS[0]}
    set -e
    note "$(grep -E '(passed|failed|error)' "$out" | tail -1)"
    return "$rc"
}

stage_real_only() {
    local out="$STATE_DIR/real-only.log" rc
    set +e
    uv run --locked python -m assistant_testing.real_only . 2>&1 | tee "$out"
    rc=${PIPESTATUS[0]}
    set -e
    note "$(tail -1 "$out")"
    return "$rc"
}

stage_core() { pytest_features core; }

# Gitignored legacy resources a fresh clone lacks. They are symlinked into the clone from the
# main checkout (found through `git rev-parse --git-common-dir`), where the legacy stack lives,
# else from this checkout; nothing in the source checkouts is changed. LEGACY_APP_DIR overrides reachy_mini_conversation_app.
LEGACY_LINKS=(.venv reachy_mini_conversation_app third_party voices local_backend/models)
LEGACY_FIXTURES="local_backend/tests/fixtures"
LEGACY_LIST="scripts/legacy_stack_tests.txt"

legacy_source() {
    local rel="$1" candidate
    if [[ "$rel" == reachy_mini_conversation_app && -n "${LEGACY_APP_DIR:-}" ]]; then
        printf '%s\n' "$LEGACY_APP_DIR"
        return 0
    fi
    for candidate in "$MAIN_CHECKOUT/$rel" "$ROOT/$rel"; do
        if [[ -e "$candidate" ]]; then printf '%s\n' "$candidate"; return 0; fi
    done
    # A clone of a clone: the main checkout of the checkout a local `origin` points to (where
    # the legacy stack lives), else that checkout itself (a worktree may hold a stray copy).
    local origin common
    origin="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
    origin="${origin#file://}"
    if [[ "$origin" == /* && -d "$origin" ]]; then
        common="$(git -C "$origin" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
        for candidate in "${common:+$(dirname "$common")/$rel}" "$origin/$rel"; do
            if [[ -n "$candidate" && -e "$candidate" ]]; then printf '%s\n' "$candidate"; return 0; fi
        done
    fi
    return 1
}

provision_legacy() {
    local rel src
    for rel in "${LEGACY_LINKS[@]}"; do
        if [[ -e "$rel" ]]; then continue; fi
        if [[ "$WORK" == "$ROOT" ]]; then
            printf 'missing here (GATE_FAST does not provision): %s\n' "$rel"
            continue
        fi
        if src="$(legacy_source "$rel")"; then
            ln -s "$src" "$rel"
            printf 'linked %s\n' "$rel"
        else
            printf 'not found in any checkout: %s\n' "$rel"
        fi
    done
    # Personal fixtures (a camera frame, voice clips) are gitignored; the clone gets a copy.
    if [[ ! -f "$LEGACY_FIXTURES/camera_frame.jpg" && "$WORK" != "$ROOT" ]] \
        && src="$(legacy_source "$LEGACY_FIXTURES/camera_frame.jpg")"; then
        cp -r --update=none "$(dirname "$src")/." "$LEGACY_FIXTURES/"
        printf 'copied %s\n' "$LEGACY_FIXTURES"
    fi
}

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# need_met <needs>: is the requirement of a listed legacy test available right now?
need_met() {
    case "$1" in
        stack) curl -s -o /dev/null -m 3 "$DAEMON_URL/" && port_open 8765 ;;
        speech) port_open 8765 ;;
        searxng) port_open 8888 ;;
        online) curl -s -o /dev/null -m 5 https://www.wikipedia.org/ ;;
        *) printf 'unknown need %q in %s\n' "$1" "$LEGACY_LIST" >&2; return 2 ;;
    esac
}

stage_legacy() {
    local py="${LEGACY_PYTHON:-third_party/speech-to-speech/.venv/bin/python}"
    provision_legacy
    if [[ ! -x "$py" ]]; then
        note "legacy venv not found ($py); set LEGACY_PYTHON"
        return 77
    fi
    py="$(cd "$(dirname "$py")" && pwd)/$(basename "$py")"
    local node needs reason args=() summary="" rc
    declare -A met count
    local need missing
    while read -r node needs reason; do
        [[ -z "$node" || "$node" == \#* ]] && continue
        missing=""
        for need in ${needs//+/ }; do
            if [[ -z "${met[$need]:-}" ]]; then
                if need_met "$need"; then met[$need]=yes; else met[$need]=no; fi
            fi
            if [[ "${met[$need]}" == no ]]; then missing+="${missing:++}$need"; fi
        done
        if [[ -n "$missing" ]]; then
            args+=(--deselect "$node")
            count[$missing]=$(( ${count[$missing]:-0} + 1 ))
            printf 'deselect (%s missing): %s  -- %s\n' "$missing" "$node" "$reason"
        fi
    done <"$LEGACY_LIST"
    for need in "${!count[@]}"; do summary+="${summary:+, }$need ${count[$need]}"; done
    local out="$STATE_DIR/legacy.log"
    set +e
    (cd local_backend && "$py" -m pytest -v -p no:cacheprovider -rfEs "${args[@]}") 2>&1 | tee "$out"
    rc=${PIPESTATUS[0]}
    set -e
    local result flaky="" failed=()
    result="$(grep -E '^=+ .* in [0-9.]+s' "$out" | tail -1 | sed -E 's/^=+ (.*) in [0-9.]+s.*/\1/')"
    # Legacy tests only: rerun each failing test ONCE. A pass on the rerun is reported as FLAKY;
    # a second failure fails the stage. (New-code unit tests and features are never rerun.)
    if [[ $rc -eq 1 ]]; then
        mapfile -t failed < <(grep -E '^(FAILED|ERROR) ' "$out" | awk '{print $2}' | sort -u)
    fi
    if [[ ${#failed[@]} -gt 0 ]]; then
        printf '\n%sRerunning %d failing legacy test(s) once:%s\n' "$YELLOW" "${#failed[@]}" "$RESET"
        printf '  %s\n' "${failed[@]}"
        local rerun="$STATE_DIR/legacy-rerun.log" node_id
        set +e
        (cd local_backend && "$py" -m pytest -v -p no:cacheprovider -rfEs "${failed[@]}") 2>&1 | tee "$rerun"
        rc=${PIPESTATUS[0]}
        set -e
        if [[ $rc -eq 0 ]]; then
            for node_id in "${failed[@]}"; do flaky+="${flaky:+, }${node_id##*::}"; done
        else
            result+="; still failing on rerun: $(grep -E '^(FAILED|ERROR) ' "$rerun" | awk '{print $2}' | sed 's/.*:://' | sort -u | paste -sd, -)"
        fi
    fi
    note "${result}${summary:+ (needs missing: $summary)}${flaky:+; FLAKY: $flaky}"
    return "$rc"
}

# robot_present: a robot on this PC, else one on the configured edge host (over SSH). Prints
# which host is used (assistant_testing.edge_host check).
robot_present() {
    uv run --locked python -m assistant_testing.edge_host check | tee "$STATE_DIR/robot-host"
    return "${PIPESTATUS[0]}"
}

models_present() {
    local gpus ok=0
    gpus="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
    printf 'GPUs: %s (need 2)\n' "${gpus:-0}"
    if [[ "${gpus:-0}" -lt 2 ]]; then ok=1; fi
    if curl -sf -m 3 "$LLM_URL/v1/models" | grep -q "\"$LLM_MODEL"; then
        printf 'LLM server: %s serves %s (%s)\n' "$LLM_URL" "$LLM_MODEL" "$LLM_STATUS"
    else
        printf 'LLM server: %s not answering or without %s (%s)\n' "$LLM_URL" "$LLM_MODEL" "$LLM_STATUS"
        ok=1
    fi
    if curl -sf -o /dev/null -m 3 "$SPEECH_URL/health"; then
        printf 'speech server: %s answering (%s)\n' "$SPEECH_URL" "$SPEECH_STATUS"
    else
        printf 'speech server: %s not answering (%s)\n' "$SPEECH_URL" "$SPEECH_STATUS"
        ok=1
    fi
    return "$ok"
}

# stop_legacy_assistant: the old assistant (legacy stack on main) holds the GPUs and the robot.
# Stopped cleanly if it runs: the app with SIGINT (its shutdown), then its speech server and
# daemon with SIGTERM; anchored patterns, so nothing else matches.
stop_legacy_assistant() {
    local spec sig pattern pids i
    for spec in "INT:^[^ ]*python[0-9.]* [^ ]*local_backend/run_app[.]py" \
                "TERM:^[^ ]*python[0-9.]* [^ ]*speech-to-speech serve" \
                "TERM:^[^ ]*python[0-9.]* [^ ]*local_backend/run_daemon[.]py"; do
        sig="${spec%%:*}"
        pattern="${spec#*:}"
        pids="$(pgrep -f "$pattern" || true)"
        [[ -n "$pids" ]] || continue
        printf 'old assistant: stopping %s (pid %s) with SIG%s\n' "$pattern" "$(echo $pids)" "$sig"
        kill -"$sig" $pids 2>/dev/null || true
        for i in $(seq 1 20); do pgrep -f "$pattern" >/dev/null || break; sleep 1; done
        pkill -KILL -f "$pattern" 2>/dev/null || true
    done
}

# start_llm_server: use our LLM server on $LLM_PORT, or start scripts/llm_server.sh (GPU0) of
# the configured kind from the checkout under test; either way reachy-gemma4 is loaded once and
# must sit entirely on GPU0. Runs in the main shell (sets LLM_SERVER/PID/STATUS).
start_llm_server() {
    local log="$STATE_DIR/llm.log" waited=0 free check="$STATE_DIR/llm-check"
    if ! LLM_SERVER="$(cd "$WORK" && uv run --locked -q python -m assistant_testing.llm_server server)"; then
        LLM_STATUS="the configured [llm] server could not be read"
        return 1
    fi
    printf 'LLM server: [llm] server = %s\n' "$LLM_SERVER"
    if curl -sf -o /dev/null -m 3 "$LLM_URL/v1/models"; then
        LLM_STATUS="already running, not started by the gate"
    else
        # The legacy stage may have left reachy-gemma4 loaded in the system Ollama on GPU0:
        # unload it through its API first (as scripts/llm_server.sh does) and give the memory
        # a moment to come back before measuring.
        if curl -sf -m 3 http://127.0.0.1:11434/api/ps 2>/dev/null | grep -q "\"name\":\"$LLM_MODEL"; then
            curl -sf -m 60 http://127.0.0.1:11434/api/generate \
                -d "{\"model\":\"$LLM_MODEL\",\"keep_alive\":0}" >/dev/null || true
            printf 'LLM server: unloaded %s from the system Ollama\n' "$LLM_MODEL"
            local i
            for i in $(seq 1 30); do
                curl -sf -m 3 http://127.0.0.1:11434/api/ps 2>/dev/null \
                    | grep -q "\"name\":\"$LLM_MODEL" || break
                sleep 1
            done
            sleep 2
        fi
        free="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
        printf 'GPU0: %s MiB free (%s needs %s)\n' "${free:-unknown}" "$LLM_SERVER" "${LLM_NEEDS_MIB[$LLM_SERVER]}"
        if [[ -z "$free" || "$free" -lt "${LLM_NEEDS_MIB[$LLM_SERVER]}" ]]; then
            LLM_STATUS="GPU0 missing or busy (${free:-no} MiB free, $LLM_SERVER needs ${LLM_NEEDS_MIB[$LLM_SERVER]}): $(nvidia-smi --id=0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | paste -sd';' -)"
            if [[ "$LLM_SERVER" == vllm ]]; then
                LLM_STATUS="$LLM_STATUS (the fallback is [llm] server = \"ollama\")"
            fi
            return 1
        fi
        # Its own process group (setsid), so stopping it also stops vLLM's engine core.
        (cd "$WORK" && exec setsid scripts/llm_server.sh --server "$LLM_SERVER" --port "$LLM_PORT") >"$log" 2>&1 &
        LLM_PID=$!
        while ! curl -sf -o /dev/null -m 2 "$LLM_URL/v1/models"; do
            if ! kill -0 "$LLM_PID" 2>/dev/null || [[ $waited -ge ${LLM_START_S[$LLM_SERVER]} ]]; then
                grep -E 'Error|error|llm server:' "$log" | tail -20
                stop_llm_server
                LLM_STATUS="the $LLM_SERVER server did not start in ${LLM_START_S[$LLM_SERVER]} s (see above)"
                return 1
            fi
            sleep 1
            waited=$((waited + 1))
        done
        grep '^llm server:' "$log" || true
        LLM_STATUS="$LLM_SERVER started by the gate in ${waited} s (pid $LLM_PID)"
    fi
    if ! (cd "$WORK" && uv run --locked -q python -m assistant_testing.llm_server check \
            --url "$LLM_URL" --model "$LLM_MODEL" --server "$LLM_SERVER") >"$check" 2>&1; then
        LLM_STATUS="$LLM_STATUS; $(tail -1 "$check")"
        return 1
    fi
    printf 'LLM server: %s; %s\n' "$LLM_STATUS" "$(tail -1 "$check")"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader || true
}

stop_llm_server() {
    if [[ -z "$LLM_PID" ]]; then return 0; fi
    if kill -0 "$LLM_PID" 2>/dev/null; then
        kill -TERM -- "-$LLM_PID" 2>/dev/null || kill -TERM "$LLM_PID" 2>/dev/null || true
        local i
        for i in $(seq 1 30); do kill -0 "$LLM_PID" 2>/dev/null || break; sleep 1; done
        printf 'LLM server (pid %s) stopped\n' "$LLM_PID"
    fi
    kill -KILL -- "-$LLM_PID" 2>/dev/null || true   # anything left in its group
    LLM_PID=""
}

# start_speech_server: use the speech server on $SPEECH_PORT, or sync its venv in the checkout
# under test and start it on GPU$SPEECH_GPU. Runs in the main shell (sets SPEECH_PID/STATUS).
start_speech_server() {
    if curl -sf -o /dev/null -m 3 "$SPEECH_URL/health"; then
        SPEECH_STATUS="already running, not started by the gate"
        printf 'speech server: %s %s\n' "$SPEECH_URL" "$SPEECH_STATUS"
        return 0
    fi
    local free venv="$WORK/servers/speech/.venv" log="$STATE_DIR/speech.log" waited=0
    free="$(nvidia-smi --id="$SPEECH_GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
    printf 'GPU%s: %s MiB free (need %s)\n' "$SPEECH_GPU" "${free:-unknown}" "$SPEECH_MIN_FREE_MIB"
    if [[ -z "$free" || "$free" -lt "$SPEECH_MIN_FREE_MIB" ]]; then
        SPEECH_STATUS="GPU$SPEECH_GPU missing or busy (${free:-no} MiB free)"
        return 1
    fi
    if ! (cd "$WORK" && UV_PROJECT_ENVIRONMENT="$venv" uv sync --locked --project servers/speech); then
        SPEECH_STATUS="uv sync of servers/speech failed"
        return 1
    fi
    (cd "$WORK" && CUDA_VISIBLE_DEVICES="$SPEECH_GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID exec "$venv/bin/python" -m assistant_speech \
        --port "$SPEECH_PORT") >"$log" 2>&1 &
    SPEECH_PID=$!
    while ! grep -q '^READY ' "$log" 2>/dev/null; do
        if ! kill -0 "$SPEECH_PID" 2>/dev/null || [[ $waited -ge 300 ]]; then
            tail -20 "$log"
            stop_speech_server
            SPEECH_STATUS="the speech server did not start (see above)"
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
    grep '^READY ' "$log"
    SPEECH_STATUS="started by the gate (pid $SPEECH_PID)"
}

stop_speech_server() {
    if [[ -z "$SPEECH_PID" ]]; then return 0; fi
    if kill -0 "$SPEECH_PID" 2>/dev/null; then
        kill -TERM "$SPEECH_PID" 2>/dev/null || true
        local i
        for i in $(seq 1 30); do kill -0 "$SPEECH_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL "$SPEECH_PID" 2>/dev/null || true
        printf 'speech server (pid %s) stopped\n' "$SPEECH_PID"
    fi
    SPEECH_PID=""
}

stage_hw() {
    if [[ "${GATE_NO_MODELS:-}" != "1" ]] && ! models_present; then
        note "the [hw, models] features need the models: LLM $LLM_STATUS; speech $SPEECH_STATUS (GATE_NO_MODELS=1 to skip)"
        return 1
    fi
    if ! robot_present; then
        note "$(tail -1 "$STATE_DIR/robot-host" 2>/dev/null) (GATE_ROBOT=sim to skip)"
        return 1
    fi
    local where rc=0 swept
    where="$(sed -E 's/^robot: //' "$STATE_DIR/robot-host" | tail -1)"
    # Leftovers of an EARLIER run (e.g. a runner killed with -9) are cleaned first and reported,
    # so they cannot be mistaken for this run's.
    # One hw run at a time: take the exclusive lock on the robot's machine BEFORE the sweep, so
    # a second run fails fast instead of sweeping away the first run's processes.
    export ASSISTANT_TEST_RUN="${ASSISTANT_TEST_RUN:-gate-$(date +%s)-$$}"
    if ! uv run --locked python -m assistant_testing.edge_host lock --pid "$$" \
            | tee "$STATE_DIR/lock"; then
        note "$(tail -1 "$STATE_DIR/lock")"
        return 1
    fi
    trap 'uv run --locked python -m assistant_testing.edge_host unlock || true' EXIT
    printf 'pre-run sweep (an earlier run'"'"'s leftovers):\n'
    uv run --locked python -m assistant_testing.edge_host sweep
    # `tier: [hw, models]` features run here, under the robot lock and sweeps.
    local marker="hw"
    if [[ "${GATE_NO_MODELS:-}" == "1" ]]; then marker="hw and not models"; fi
    pytest_features "$marker" || rc=$?
    # Teardown must have stopped everything on the robot's machine; anything still running from
    # ~/assistant-edge is stopped now and fails the stage.
    uv run --locked python -m assistant_testing.edge_host sweep | tee "$STATE_DIR/sweep"
    swept="$(tail -1 "$STATE_DIR/sweep")"
    note "$(cat "$STATE_DIR/note" 2>/dev/null); robot $where"
    if [[ "$swept" != "sweep: 0 "* ]]; then
        note "$(cat "$STATE_DIR/note"); LEFTOVERS: $swept"
        return 1
    fi
    return "$rc"
}

# The simulated robot (assistant_testing.sim): Pollen's daemon with --sim on this PC, and its
# virtual sound card. Leftovers of an earlier run are swept first and reported; leftovers after
# the features FAIL the stage.
stage_sim() {
    if [[ "${GATE_NO_MODELS:-}" != "1" ]] && ! models_present; then
        note "the [sim, hw, models] features need the models: LLM $LLM_STATUS; speech $SPEECH_STATUS (GATE_NO_MODELS=1 to skip)"
        return 1
    fi
    if ! uv run --locked python -m assistant_testing.sim prepare | tee "$STATE_DIR/sim-ready"; then
        note "$(tail -1 "$STATE_DIR/sim-ready")"
        return 1
    fi
    printf 'pre-run sweep (an earlier run'"'"'s leftovers):\n'
    uv run --locked python -m assistant_testing.sim sweep || true
    local marker="sim" rc=0 swept
    if [[ "${GATE_NO_MODELS:-}" == "1" ]]; then marker="sim and not models"; fi
    pytest_features "$marker" || rc=$?
    uv run --locked python -m assistant_testing.sim sweep | tee "$STATE_DIR/sim-sweep" || true
    swept="$(tail -1 "$STATE_DIR/sim-sweep")"
    if [[ "$swept" != "sim sweep: 0 "* ]]; then
        note "$(cat "$STATE_DIR/note" 2>/dev/null); LEFTOVERS: $swept"
        return 1
    fi
    return "$rc"
}

stage_models() {
    if ! models_present; then
        note "needs both GPUs, $LLM_MODEL at $LLM_URL ($LLM_STATUS) and the speech server ($SPEECH_STATUS) (GATE_NO_MODELS=1 to skip)"
        return 1
    fi
    pytest_features "models and not hw and not sim"
}

stage_skipped_sim() { note "GATE_ROBOT=$GATE_ROBOT"; return 77; }
stage_skipped_hw() { note "GATE_ROBOT=$GATE_ROBOT${GATE_NO_HW:+ (GATE_NO_HW=1)}"; return 77; }
stage_skipped_models() { note "GATE_NO_MODELS=1"; return 77; }

# ---------------------------------------------------------------- run

if [[ "${GATE_FAST:-}" != "1" ]]; then
    CLONE_PARENT="$(mktemp -d -t assistant-gate.XXXXXX)"
    CLONE_DIR="$CLONE_PARENT/robots"
    WORK="$CLONE_DIR"
fi

run_stage a "fresh clone + uv sync" stage_clone
if [[ "${RESULT[a]}" != "PASS" ]]; then
    for id in b c d e f s g h; do STAGES+=("$id"); RESULT[$id]="NOT RUN"; NOTE[$id]="stage a failed"; done
else
    run_stage b "lint: ruff, basedpyright, import-linter" stage_lint
    run_stage c "unit tests" stage_unit
    run_stage d "real-only check" stage_real_only
    run_stage e "e2e features, tier core" stage_core
    run_stage f "legacy tests (local_backend)" stage_legacy
    banner "the old assistant, the LLM server and the speech server"
    stop_legacy_assistant
    if [[ "${GATE_NO_MODELS:-}" != "1" ]]; then
        start_llm_server || printf '%sLLM server: %s%s\n' "$RED" "$LLM_STATUS" "$RESET"
        start_speech_server || printf '%sspeech server: %s%s\n' "$RED" "$SPEECH_STATUS" "$RESET"
    fi
    if [[ "$GATE_ROBOT" == hw ]]; then
        warn_loud "GATE_ROBOT=hw: the simulated robot (sim) features were NOT run."
        INCOMPLETE+=("sim")
        run_stage s "e2e features, tier sim" stage_skipped_sim
    else
        run_stage s "e2e features, tier sim" stage_sim
    fi
    if [[ "$GATE_ROBOT" == sim ]]; then
        warn_loud "GATE_ROBOT=sim: the physical robot (hw) features were NOT run."
        INCOMPLETE+=("hw")
        run_stage g "e2e features, tier hw" stage_skipped_hw
    else
        run_stage g "e2e features, tier hw" stage_hw
    fi
    if [[ "${GATE_NO_MODELS:-}" == "1" ]]; then
        warn_loud "GATE_NO_MODELS=1: the GPU/model-server (models) features were NOT run."
        INCOMPLETE+=("models")
        run_stage h "e2e features, tier models" stage_skipped_models
    else
        run_stage h "e2e features, tier models" stage_models
    fi
    stop_speech_server
    stop_llm_server
fi

banner "i) summary"
declare -A TITLE=(
    [a]="clone + uv sync" [b]="lint" [c]="unit" [d]="real-only check" [e]="features: core"
    [f]="legacy tests" [s]="features: sim" [g]="features: hw" [h]="features: models"
)
failed=0
for id in "${STAGES[@]}"; do
    case "${RESULT[$id]}" in
        PASS) colour="$GREEN" ;;
        SKIP) colour="$YELLOW" ;;
        *) colour="$RED"; failed=1 ;;
    esac
    printf '  %s) %-18s %s%-7s%s %7s  %s\n' "$id" "${TITLE[$id]}" "$colour" "${RESULT[$id]}" "$RESET" \
        "$(duration "${TIME[$id]:-0}")" "${NOTE[$id]}"
done
printf '\n  robot features: sim %s, hw %s; the whole gate %s\n' \
    "$( [[ -n "${TIME[s]:-}" && "${RESULT[s]}" != SKIP ]] && duration "${TIME[s]}" || echo "not run")" \
    "$( [[ -n "${TIME[g]:-}" && "${RESULT[g]}" != SKIP ]] && duration "${TIME[g]}" || echo "not run")" \
    "$(duration "$SECONDS")"
if [[ ${#INCOMPLETE[@]} -gt 0 ]]; then
    warn_loud "opted out of: ${INCOMPLETE[*]}"
fi
if [[ $failed -ne 0 ]]; then
    printf '\n%s%sGATE: FAIL%s\n' "$RED" "$BOLD" "$RESET"
    exit 1
fi
if [[ ${#INCOMPLETE[@]} -gt 0 ]]; then
    printf '\n%s%sGATE: INCOMPLETE (%s not run; exit 3)%s\n' "$YELLOW" "$BOLD" "${INCOMPLETE[*]}" "$RESET"
    exit 3
else
    printf '\n%s%sGATE: PASS%s\n' "$GREEN" "$BOLD" "$RESET"
fi
