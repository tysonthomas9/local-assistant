#!/usr/bin/env bash
# The merge gate for the new assistant stack. Run it from anywhere in the repo:
#
#   scripts/gate.sh
#
# Stages:
#   a  clone   git clone HEAD into a temp dir and `uv sync --locked` from scratch
#   b  lint    ruff (check + format), basedpyright, import-linter
#   c  unit    the few unit tests (pure logic) plus the schema snapshot
#   d  core    e2e features, tier core
#   e  legacy  the legacy suite as the README "Tests" section runs it (skipped with a reason
#              when its venv, fixtures or running stack are unavailable; legacy files are
#              never modified)
#   f  hw      e2e features, tier hw: FAILS if the robot is missing (/dev/ttyACM0 or the
#              daemon at 127.0.0.1:8000)
#   g  models  e2e features, tier models: FAILS if the GPUs or an LLM server are missing
#   h  summary PASS/FAIL per stage; exits non-zero on any failure
#
# Environment:
#   GATE_FAST=1       test the working tree in place instead of a fresh clone (uncommitted
#                     changes are then included)
#   GATE_NO_HW=1      skip stage f   } prints a loud WARNING: the gate is not complete
#   GATE_NO_MODELS=1  skip stage g   }
#   LEGACY_PYTHON     python for the legacy suite (default: third_party/speech-to-speech/.venv
#                     in this checkout or in the main checkout of this repository)
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
MAIN_CHECKOUT="$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")"
DAEMON_URL="http://127.0.0.1:8000"
LLM_URLS=("http://127.0.0.1:8773/v1/models" "http://127.0.0.1:11434/v1/models")
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true

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
    passed="$(grep -Eo '[0-9]+ passed' "$out" | tail -1 || true)"
    note "${passed:-no scenarios passed}"
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
    fi
    uv sync --locked
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
    uv run --locked pytest packages tests -q -p no:cacheprovider 2>&1 | tee "$out"
    note "$(grep -Eo '[0-9]+ passed' "$out" | tail -1)"
}

stage_core() { pytest_features core; }

find_legacy_python() {
    local candidate
    for candidate in \
        "${LEGACY_PYTHON:-}" \
        "$ROOT/third_party/speech-to-speech/.venv/bin/python" \
        "$MAIN_CHECKOUT/third_party/speech-to-speech/.venv/bin/python"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

stage_legacy() {
    local py fixtures="local_backend/tests/fixtures" source_dir
    if ! py="$(find_legacy_python)"; then
        note "legacy venv (third_party/speech-to-speech/.venv) not found; set LEGACY_PYTHON"
        return 77
    fi
    # Personal fixtures (a camera frame, voice clips) are gitignored. A fresh clone gets a
    # copy from a checkout that has them; nothing in the source checkouts is changed.
    if [[ ! -f "$fixtures/camera_frame.jpg" ]]; then
        source_dir=""
        for candidate in "$ROOT/$fixtures" "$MAIN_CHECKOUT/$fixtures"; do
            if [[ -f "$candidate/camera_frame.jpg" ]]; then source_dir="$candidate"; break; fi
        done
        if [[ -z "$source_dir" || "$WORK" == "$ROOT" ]]; then
            note "legacy fixtures (gitignored camera_frame.jpg, stt/) not available here"
            return 77
        fi
        cp -r --update=none "$source_dir/." "$fixtures/"
    fi
    # The legacy suite talks to the running legacy stack and fails, not skips, without it.
    if ! curl -s -o /dev/null -m 3 "$DAEMON_URL/" || ! (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then
        note "legacy stack not running (needs the daemon on :8000 and the speech server on :8765; start_daemon.sh, start_local_backend.sh, start_conversation.sh)"
        return 77
    fi
    cd local_backend
    local out="$STATE_DIR/legacy.log"
    "$py" -m pytest -v -p no:cacheprovider 2>&1 | tee "$out"
    note "$(grep -Eo '[0-9]+ passed.*' "$out" | tail -1)"
}

robot_present() {
    if [[ -e /dev/ttyACM0 ]]; then
        printf 'robot: /dev/ttyACM0 present\n'
        return 0
    fi
    if curl -s -o /dev/null -m 3 "$DAEMON_URL/"; then
        printf 'robot: daemon answering at %s\n' "$DAEMON_URL"
        return 0
    fi
    return 1
}

models_present() {
    local gpus url
    gpus="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
    if [[ "${gpus:-0}" -lt 1 ]]; then
        printf 'no GPUs found (nvidia-smi -L)\n'
        return 1
    fi
    printf 'GPUs: %s\n' "$gpus"
    for url in "${LLM_URLS[@]}"; do
        if curl -sf -o /dev/null -m 3 "$url"; then
            printf 'LLM server: %s\n' "$url"
            return 0
        fi
    done
    printf 'no LLM server answering (%s)\n' "${LLM_URLS[*]}"
    return 1
}

stage_hw() {
    if ! robot_present; then
        note "robot not found: no /dev/ttyACM0 and no daemon at $DAEMON_URL (GATE_NO_HW=1 to skip)"
        return 1
    fi
    pytest_features hw
}

stage_models() {
    if ! models_present; then
        note "GPUs or model servers missing (GATE_NO_MODELS=1 to skip)"
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
    for id in b c d e f g; do STAGES+=("$id"); RESULT[$id]="NOT RUN"; NOTE[$id]="stage a failed"; done
else
    run_stage b "lint: ruff, basedpyright, import-linter" stage_lint
    run_stage c "unit tests" stage_unit
    run_stage d "e2e features, tier core" stage_core
    run_stage e "legacy tests (local_backend)" stage_legacy
    if [[ "${GATE_NO_HW:-}" == "1" ]]; then
        warn_loud "GATE_NO_HW=1: the robot (hw) features were NOT run."
        INCOMPLETE+=("hw")
        run_stage f "e2e features, tier hw" stage_skipped_hw
    else
        run_stage f "e2e features, tier hw" stage_hw
    fi
    if [[ "${GATE_NO_MODELS:-}" == "1" ]]; then
        warn_loud "GATE_NO_MODELS=1: the GPU/model-server (models) features were NOT run."
        INCOMPLETE+=("models")
        run_stage g "e2e features, tier models" stage_skipped_models
    else
        run_stage g "e2e features, tier models" stage_models
    fi
fi

banner "h) summary"
declare -A TITLE=(
    [a]="clone + uv sync" [b]="lint" [c]="unit" [d]="features: core"
    [e]="legacy tests" [f]="features: hw" [g]="features: models"
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
    printf '\n%s%sGATE: PASS (INCOMPLETE: %s not run)%s\n' "$YELLOW" "$BOLD" "${INCOMPLETE[*]}" "$RESET"
else
    printf '\n%s%sGATE: PASS%s\n' "$GREEN" "$BOLD" "$RESET"
fi
