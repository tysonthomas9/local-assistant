# Speech server

A minimal loopback HTTP server for the brain's speech: Parakeet TDT 0.6B v3 speech-to-text
(`nano-parakeet`, PyTorch) and Qwen3-TTS 1.7B CustomVoice text-to-speech (`faster-qwen3-tts`
through qwentts.cpp, BF16 GGUF), both on one GPU (GPU1 on the brain PC, about 7 GB).

It has its own locked venv (`servers/speech/.venv`, about 6 GB with CUDA PyTorch), separate
from the uv workspace (`.venv-assistant`) and the legacy root `.venv`:

```bash
uv sync --locked --project servers/speech
CUDA_VISIBLE_DEVICES=1 servers/speech/.venv/bin/python -m assistant_speech --port 8772
```

It prints `READY url=http://127.0.0.1:8772 gpu=... gpu_memory_mib=... load_s=...` once both
models are loaded and warmed up (about 10 s) and `STOPPED` when it exits (SIGTERM or Ctrl-C).
It binds to loopback only. Options: `--port` (0 picks a free one), `--device`, `--tts-quant`
(BF16, Q8_0, Q4_K_M, F32), `--language`, `--online`.

## Weights

The server runs offline (`HF_HUB_OFFLINE=1`) from the Hugging Face cache
(`~/.cache/huggingface/hub`):

| Model | Cache | Size |
|---|---|---|
| STT `nvidia/parakeet-tdt-0.6b-v3` | `models--nvidia--parakeet-tdt-0.6b-v3` | 2.4 GB |
| TTS `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` as GGUF | `models--Serveurperso--Qwen3-TTS-GGUF` | 4.0 GB |

Start it once with `--online` to download missing weights.

## API

| Request | Body | Answer |
|---|---|---|
| `GET /health` | | `{ok, stt_model, tts_model, voices, rate, device, gpu, gpu_memory_mib, ...}` |
| `GET /requests` | | The last 200 requests: kind (`stt`/`tts`), voice, time to first audio, audio length, outcome |
| `POST /v1/audio/transcriptions` | a WAV (`audio/wav`) or raw s16le mono (`audio/pcm`, `?rate=16000`) | `{id, text, audio_ms, ms}` |
| `POST /v1/audio/speech` | JSON `{input, voice, response_format: "pcm"}` | streamed s16le mono PCM at 24 kHz (`X-Sample-Rate`, `X-Request-Id`); closing the response stops the synthesis |

Voices: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian (the assistants'
`voice.speaker`, lower case: Jarvis is ryan, Marvin is eric). One request runs on the GPU at a
time; others wait.

## Golden voice clips

`tests/fixtures/audio/*.wav` (synthetic speech, 16 kHz mono 16-bit, with silence before and
after) are the e2e features' voice input. They were made with this server's TTS from
`tests/fixtures/audio/golden.toml`:

```bash
servers/speech/.venv/bin/python -m assistant_speech.golden --url http://127.0.0.1:8772
```
