# New assistant stack. Run `uv sync` first; `uv run just <recipe>` works without a global just.

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

# default test layers (unit, contract, component); pass pytest args, e.g. `just test -m contract`
test *args:
    uv run pytest {{args}}

# write the EdgeLink JSON Schema snapshot for a NEW protocol version (never overwrites)
schema-snapshot:
    UPDATE_SCHEMA_SNAPSHOT=1 uv run pytest tests/contract/test_edgelink_schema.py

# brain + edge dev runner (task S6)
dev:
    @echo "just dev: not implemented yet (skeleton task S6 adds process-compose)"
