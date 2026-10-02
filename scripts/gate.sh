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
#   g  hw      e2e features, tier hw. The robot is used where it is plugged in: on this PC
#              (/dev/ttyACM*) if attached here, else on the edge host reached by the SSH alias in
#              config [test.edge_host] ssh / ASSISTANT_EDGE_HOST (it needs /dev/cu.usbmodem* or
#              /dev/ttyACM* THERE). Prints which host is used; FAILS if neither has the robot.
#              Before and after the features it sweeps leftovers: processes from ~/assistant-edge
#              on that host, tagged test ssh clients/tunnels on this PC and their sshd forwards
#              there. Leftovers after the features FAIL the stage (docs/robot-on-another-machine.md)
#   h  models  e2e features, tier models: FAILS unless both GPUs and Ollama are available
#   i  summary PASS/FAIL per stage. Exit 0 = PASS, 1 = FAIL, 3 = INCOMPLETE (a stage opted out)
#
# E2E uses only real devices and the real stack (see e2e/features/README.md).
#
# Environment:
#   GATE_FAST=1       test the working tree in place instead of a fresh clone (uncommitted
#                     changes are then included)
#   GATE_NO_HW=1      skip stage g   } prints a loud WARNING and exits 3 (incomplete)
#   GATE_NO_MODELS=1  skip stage h   }
#   LEGACY_PYTHON     python for the legacy suite (default: third_party/speech-to-speech/.venv)
#   LEGACY_APP_DIR    reachy_mini_conversation_app to link (default: from the main checkout)
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
MAIN_CHECKOUT="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
DAEMON_URL="http://127.0.0.1:8000"
OLLAMA_URL="http://127.0.0.1:11434/v1/models"
unset VIRTUAL_ENV || true
# The uv workspace lives in .venv-assistant; .venv is the legacy root venv (piper, reachy-mini
# SDK and daemon) and the new stack must never touch it.
export UV_PROJECT_ENVIRONMENT=.venv-assistant

STATE_DIR="$(mktemp -d -t assistant-gate-state.XXXXXX)"
CLONE_PARENT=""
CLONE_DIR=""
cleanup() {
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
declare -A RESULT NOTE
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
    local id="$1" title="$2" fn="$3" rc
    STAGES+=("$id")
    banner "$id) $title"
    rm -f "$STATE_DIR/note"
    set +e
    (set -euo pipefail; if [[ -d "$WORK" ]]; then cd "$WORK"; fi; "$fn")
    rc=$?
    set -e
    NOTE[$id]="$(cat "$STATE_DIR/note" 2>/dev/null || true)"
    case "$rc" in
        0) RESULT[$id]="PASS" ;;
        77) RESULT[$id]="SKIP" ;;
        *) RESULT[$id]="FAIL"; NOTE[$id]="${NOTE[$id]:-exit code $rc}" ;;
    esac
    printf '%s-> %s %s%s\n' "$BOLD" "${RESULT[$id]}" "${NOTE[$id]}" "$RESET"
}

# pytest_features <marker>: run one tier; "no tests collected" (exit 5) means 0 features.
pytest_features() {
    local marker="$1" out rc passed
    out="$STATE_DIR/pytest-$marker.log"
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
    if curl -sf -o /dev/null -m 3 "$OLLAMA_URL"; then
        printf 'Ollama: %s answering\n' "$OLLAMA_URL"
    else
        printf 'Ollama: %s not answering\n' "$OLLAMA_URL"
        ok=1
    fi
    return "$ok"
}

stage_hw() {
    if ! robot_present; then
        note "$(tail -1 "$STATE_DIR/robot-host" 2>/dev/null) (GATE_NO_HW=1 to skip)"
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
    pytest_features hw || rc=$?
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

stage_models() {
    if ! models_present; then
        note "needs both GPUs and Ollama at $OLLAMA_URL (GATE_NO_MODELS=1 to skip)"
        return 1
    fi
    pytest_features models
}

stage_skipped_hw() { note "GATE_NO_HW=1"; return 77; }
stage_skipped_models() { note "GATE_NO_MODELS=1"; return 77; }

# ---------------------------------------------------------------- run

if [[ "${GATE_FAST:-}" != "1" ]]; then
    CLONE_PARENT="$(mktemp -d -t assistant-gate.XXXXXX)"
    CLONE_DIR="$CLONE_PARENT/robots"
    WORK="$CLONE_DIR"
fi

run_stage a "fresh clone + uv sync" stage_clone
if [[ "${RESULT[a]}" != "PASS" ]]; then
    for id in b c d e f g h; do STAGES+=("$id"); RESULT[$id]="NOT RUN"; NOTE[$id]="stage a failed"; done
else
    run_stage b "lint: ruff, basedpyright, import-linter" stage_lint
    run_stage c "unit tests" stage_unit
    run_stage d "real-only check" stage_real_only
    run_stage e "e2e features, tier core" stage_core
    run_stage f "legacy tests (local_backend)" stage_legacy
    if [[ "${GATE_NO_HW:-}" == "1" ]]; then
        warn_loud "GATE_NO_HW=1: the robot (hw) features were NOT run."
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
fi

banner "i) summary"
declare -A TITLE=(
    [a]="clone + uv sync" [b]="lint" [c]="unit" [d]="real-only check" [e]="features: core"
    [f]="legacy tests" [g]="features: hw" [h]="features: models"
)
failed=0
for id in "${STAGES[@]}"; do
    case "${RESULT[$id]}" in
        PASS) colour="$GREEN" ;;
        SKIP) colour="$YELLOW" ;;
        *) colour="$RED"; failed=1 ;;
    esac
    printf '  %s) %-18s %s%-7s%s %s\n' "$id" "${TITLE[$id]}" "$colour" "${RESULT[$id]}" "$RESET" "${NOTE[$id]}"
done
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
