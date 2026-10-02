# New assistant stack. Run `uv sync` first; `uv run just <recipe>` works without a global just.
# The uv workspace venv is .venv-assistant; .venv stays the legacy root venv (docs/dev-setup.md).

set export

UV_PROJECT_ENVIRONMENT := ".venv-assistant"

default:
    @just --list

# ruff, basedpyright and the import-linter contracts
lint:
    uv run ruff check
    uv run ruff format --check
    uv run basedpyright
    uv run lint-imports

# apply ruff fixes and formatting
fmt:
    uv run ruff check --fix
    uv run ruff format

# unit tests + core features; pass pytest args, e.g. `just test e2e -m core`
test *args:
    uv run pytest {{args}}

# list the e2e feature files, scenarios and steps
features:
    uv run pytest e2e --list-features -q

# the full merge gate (fresh clone, lint, unit, features core/hw/models, legacy suite)
gate:
    scripts/gate.sh

# write the EdgeLink JSON Schema snapshot for a NEW protocol version (never overwrites)
schema-snapshot:
    UPDATE_SCHEMA_SNAPSHOT=1 uv run pytest tests/contract/test_edgelink_schema.py

# our LLM server: ollama serve on 127.0.0.1:8773, GPU0 only (docs/dev-setup.md)
llm:
    scripts/llm_server.sh

# brain + edge dev runner (task S6)
dev:
    @echo "just dev: not implemented yet (skeleton task S6 adds process-compose)"
