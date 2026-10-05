#!/usr/bin/env bash
# Copy the harness to the bench host (ssh alias, default "desktop2") under ~/assistant-dxpoc/src/duplex.
set -euo pipefail
HOST=${DX_HOST:-desktop2}
cd "$(dirname "$0")/.."
rsync -a --delete --exclude '.venv' --exclude '__pycache__' ./ "$HOST:assistant-dxpoc/src/duplex/"
