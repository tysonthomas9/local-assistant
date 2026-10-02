# Dev setup (new assistant stack)

## Two Python environments, side by side

| Directory | What it is | Managed by |
|---|---|---|
| `.venv-assistant/` | the uv workspace of the new stack (`packages/*`) | `uv sync` |
| `.venv/` | the legacy root venv: piper, the reachy-mini SDK and daemon | by hand, see `REACHY_MINI_SETUP.md` |

The new stack must never touch the legacy `.venv`. uv puts the workspace venv wherever
`UV_PROJECT_ENVIRONMENT` points, so set it to `.venv-assistant` before running uv:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-assistant
uv sync
uv run just lint
uv run just test
```

- `justfile` exports `UV_PROJECT_ENVIRONMENT=.venv-assistant` for all its recipes (`set export`).
- `scripts/gate.sh` exports it too. Stage a checks that `uv sync` left the legacy `.venv`
  unchanged by comparing a fingerprint of it (package list and mtimes) taken before and after.
- basedpyright reads `.venv-assistant` (`venvPath`/`venv` in `pyproject.toml`).
- Both directories are gitignored.

A plain `uv sync` without the variable would create or replace `.venv`, which breaks the
legacy stack. To avoid that, set the variable once per shell, or with direnv add an
(uncommitted) `.envrc` in the repo root:

```bash
# .envrc
export UV_PROJECT_ENVIRONMENT=.venv-assistant
```

and run `direnv allow`.

## The gate

`scripts/gate.sh` clones HEAD into a temp dir and runs every stage there (see the header of
the script). Gitignored legacy resources (`.venv`, `reachy_mini_conversation_app`,
`third_party`, `voices`, `local_backend/models`) are symlinked into the clone from the main
checkout, so the legacy suite runs against the real legacy environment. Exit codes: 0 pass,
1 fail, 3 incomplete (`GATE_NO_HW=1` or `GATE_NO_MODELS=1`).

The hw stage uses the robot wherever it is plugged in. If it is attached to another machine
(e.g. a Mac), see [Running robot tests with the robot on another machine](robot-on-another-machine.md).
