#!/usr/bin/env bash
# Download everything the daemon and app fetch from Hugging Face, so they can run with
# HF_HUB_OFFLINE=1. Run once while online (and again after upgrading reachy-mini).
#   - pollen-robotics/face_detection_yunet_2026may   head tracking (daemon, lazily on first use)
#   - pollen-robotics/reachy-mini-emotions-library   play_emotion (app) + daemon startup preload
#   - pollen-robotics/reachy-mini-dances-library     daemon startup preload
#   - sherpa-onnx English keyword spotter (18 MB)     "Hey Reachy" wake phrase (--wake), local_backend/models/kws/
# The speech-to-speech models are cached by its first run (see LOCAL_CONVERSATION.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_HUB_OFFLINE=0 "$ROOT/.venv/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
from reachy_mini.motion.recorded_move import DEFAULT_DATASETS, preload_dataset
from reachy_mini.vision import face_detector as fd

print("face model:", hf_hub_download(fd._MODEL_REPO, fd._MODEL_FILE, revision=fd._MODEL_REVISION))
for ds in DEFAULT_DATASETS:
    print("dataset:", ds, "->", preload_dataset(ds))
PY

KWS="$ROOT/local_backend/models/kws"
KWS_NAME=sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
if [ ! -f "$KWS/$KWS_NAME/tokens.txt" ]; then
    mkdir -p "$KWS"
    curl -sSLf "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$KWS_NAME.tar.bz2" | tar xj -C "$KWS"
fi
echo "keyword spotter: $KWS/$KWS_NAME"
