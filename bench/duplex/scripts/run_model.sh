#!/usr/bin/env bash
# Run on the bench host: scripts/run_model.sh <model> [scenarios|all] [take] [adapter-kwargs-json]
# Refuses to start if any other compute process is on the GPU (never kill other jobs).
set -euo pipefail
source "$HOME/assistant-dxpoc/env.sh"
cd "$DX/src/duplex"
M=$1; SC=${2:-all}; TAKE=${3:-t1}; KW=${4:-"{}"}
busy=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "$busy" ]; then echo "GPU busy, not starting:"; echo "$busy"; exit 3; fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
uv sync -q --locked --project "models/$M"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH="$PWD" uv run --no-sync --project "models/$M" python -m dxbench.runner \
  --adapter "models/$M/adapter.py" --clips "$DX/clips" --out "$DX/runs/$M" \
  --scenarios "$SC" --take "$TAKE" --kw "$KW"
