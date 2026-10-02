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

Before the robot (hw) and models stages the gate checks the models: Ollama with
`reachy-gemma4`, and the speech server (`servers/speech`, see its README) on 127.0.0.1:8772.
If none is serving there it syncs the speech server's venv and starts one on GPU1, and stops
it again at the end; a server it did not start is left alone. No GPU, a busy GPU1, no
Ollama or an Ollama not pinned to GPU0 (below) fail those stages. Timings of the spoken turns
are kept in `artifacts/` of the checkout the gate was run from. See "The models tier" in
`e2e/features/README.md` to run the models features by hand.

### Ollama on GPU0 (one-time setting)

The LLM runs on GPU0 and the speech server on GPU1. Left alone, the system Ollama spreads
`reachy-gemma4` (about 22 GB at its 32k context) over both GPUs, and then the speech server
(about 7 GB) no longer fits on GPU1, or the reverse, depending on which loads first. Pin the
service to GPU0 once. Ollama also drives the GPUs through Vulkan, which ignores
`CUDA_VISIBLE_DEVICES`, so Vulkan is turned off:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="CUDA_VISIBLE_DEVICES=0" "CUDA_DEVICE_ORDER=PCI_BUS_ID" "OLLAMA_VULKAN=0"\n' \
  | sudo tee /etc/systemd/system/ollama.service.d/gpu0.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

The gate and the models features check this setting (`systemctl show ollama`) and fail,
naming it, while it is missing. An `ollama serve` that a feature starts itself gets the same
environment.

The hw stage uses the robot wherever it is plugged in. If it is attached to another machine
(e.g. a Mac), see [Running robot tests with the robot on another machine](robot-on-another-machine.md).
