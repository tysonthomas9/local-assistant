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
1 fail, 3 incomplete (`GATE_ROBOT=sim`, `GATE_ROBOT=hw`, `GATE_NO_HW=1` or `GATE_NO_MODELS=1`).

The robot features run on both robots by default (`GATE_ROBOT=both`): stage s on Pollen's
simulated Reachy Mini on this PC, then stage g on the real one. `GATE_ROBOT=sim` is for
iterating without the robot; the final gate runs both. The summary shows each stage's time,
and the sim and hw feature times. See `e2e/features/README.md`, "The sim tier".

Before the robot (sim, hw) and models stages the gate stops the old assistant if it runs (the
legacy stack holds the GPUs and the robot) and brings up the models, each on its own GPU:

- the LLM: our own LLM server (below; vLLM by default) on 127.0.0.1:8773, on GPU0;
- the speech server (`servers/speech`, see its README) on 127.0.0.1:8772, on GPU1: the gate
  syncs its venv and starts it.

A server already serving on its port is used and left alone; the gate stops only what it
started, after the models stage. No GPU, a busy GPU, a model not entirely on its GPU or a
server that does not start fail those stages. Timings of the spoken turns are kept in
`artifacts/` of the checkout the gate was run from. See "The models tier" in
`e2e/features/README.md` to run the models features by hand.

### The LLM server

The stack runs its own LLM server on GPU0, `scripts/llm_server.sh`, on 127.0.0.1:8773
(`[llm] base_url` defaults to it). `[llm] server` picks it: `"vllm"` (the default) or
`"ollama"` (the fallback); both serve the OpenAI API with the model `reachy-gemma4`, so
switching is that one value (or `ASSISTANT__LLM__SERVER=ollama`, or `--server ollama`):

```bash
scripts/llm_server.sh                    # the configured server until Ctrl-C; or `uv run just llm`
scripts/llm_server.sh --server ollama    # the fallback
```

Both run on GPU0 only (`CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`), so GPU1
always stays free for the speech server, whichever loads first. Before serving, the script frees
GPU0 of other copies of the model through their APIs: it unloads reachy-gemma4 from an Ollama
(the system service on 11434 and the stack's on 8773; `keep_alive` 0) and puts a vLLM on 8773
to sleep (level 1: about 0.8 GB stays on GPU0; the test steps wake it once GPU0 has room). The
system Ollama service itself is not used or changed.

**vLLM** (default) serves `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` from the local Hugging Face
cache (offline, no downloads) out of its own pinned env, `servers/vllm` (vLLM 0.29.0, uv
project; the script runs `uv sync --locked` into `servers/vllm/.venv`). Flags:

```
--served-model-name reachy-gemma4 gemma4-26b --max-model-len 32768
--gpu-memory-utilization 0.90 --max-num-seqs 8 --enable-prefix-caching
--scheduling-policy priority --enable-auto-tool-choice --tool-call-parser gemma4
--reasoning-parser gemma4 --language-model-only --enable-sleep-mode
```

That takes about 21.9 GB of GPU0 (15.6 GB weights, 3.9 GB KV cache = 80k tokens), so the gate
needs about 21.8 GB free on GPU0. The first start takes about 1.5 minutes (torch.compile, then
cached; later starts about 50 s). `ASSISTANT_VLLM_MAX_MODEL_LEN`,
`ASSISTANT_VLLM_GPU_MEMORY_UTILIZATION` and `ASSISTANT_VLLM_MAX_NUM_SEQS` override the sizes.
Priority scheduling is what the brain's `send_priority` uses (sent to vLLM only): a voice turn
takes the next free slot ahead of waiting background requests. FlashInfer's sampler is off
(`VLLM_USE_FLASHINFER_SAMPLER=0`: its JIT build needs a newer nvcc than the PC's).

**Ollama** (fallback) is `ollama serve` with the system Ollama's models
(`/usr/share/ollama/.ollama/models`, read-only, never pruned; no downloads),
`OLLAMA_NUM_PARALLEL=2` (about 20 GB of GPU0), `OLLAMA_VULKAN=0` (its Vulkan backend ignores
`CUDA_VISIBLE_DEVICES`) and keeps the model loaded 30 minutes after the last request.

### One GPU (`[gpu] layout = "one"`)

The default layout is two cards (LLM on GPU0, speech on GPU1). `[gpu] layout = "one"` (or
`ASSISTANT__GPU__LAYOUT=one`; for the gate `GATE_GPU=one`) puts both on `[gpu] gpu_index`. The
speech server must start first; `scripts/llm_server.sh` then sizes vLLM to what is left:
a bf16 KV cache of (free − `margin_mib` − 16.9 GB), `--max-model-len` `[gpu] max_model_len`
(65536) and `--max-num-seqs` `[gpu] max_num_seqs` (8). It refuses to start if that leaves under
1 GB of KV cache. On a 24 GB RTX 3090 this needs the small speech server,
`--tts 0.6b --tts-quant Q8_0` (about 4.1 GB; the gate passes `ASSISTANT_SPEECH_ARGS` to it):
vLLM then takes about 19.2 GB with 64k of context, and a 60k-token prefill while TTS and STT
work still leaves 0.7 GB free. The 1.7B BF16 TTS (6.8 GB) leaves too little for a useful
context. fp8 KV is not possible on an RTX 3090: Gemma 4's 512-wide heads need SM89+ (Triton)
or SM100+ (FlashInfer) for it.

`GATE_GPU=one` runs the gate this way. It leaves out the features that start a second LLM
or speech server next to the stack's (servers_compared, llm_down, vllm_down,
speech_server_down, speech_worker_crash): one card has no room for a second copy, and putting the stack's vLLM to sleep moves its 16 GB of weights into
CPU memory. `GATE_GPU=two` runs them.

The cost is speed (measured with the sim robot on a single-RTX 3090 machine): TTS and LLM share the card, so the TTS first audio takes about 95 ms
instead of 20 ms. Typed input to first audio is about 215 ms (83 ms on two cards); voice is
about 245 ms (252 ms on two cards).
These are times to the start of playback. The 0.6B TTS begins most replies with 0.1-1 s of
silence (the 1.7B: 80 ms), so its first audible sound comes about 0.4 s later. Trimming that
silence in the speech server (it is generated about 8x faster than real time) is still to do.

If vLLM cannot start (e.g. GPU0 is busy), the gate's status names the processes on GPU0 and
points to `[llm] server = "ollama"`.

The hw stage uses the robot wherever it is plugged in. If it is attached to another machine
(e.g. a Mac), see [Running robot tests with the robot on another machine](robot-on-another-machine.md).
