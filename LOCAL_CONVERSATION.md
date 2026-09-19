# Fully local Reachy Mini conversation

Drafted 2026-09-18. Status: **working: fully local, verified by syscall trace.** Approved 2026-09-18, including removing web search and weather so everything is fully local. Builds on [CONVERSATION_APP.md](CONVERSATION_APP.md), which runs the app against Pollen's hosted Hugging Face backend.

## Current setup (as of 2026-09-18, after the Fable audit)

The conversation app, the daemon and the robot are Pollen's code, unchanged. Only the cloud backend is replaced, using the app's own `local` mode pointed at a self-hosted [speech-to-speech](https://github.com/huggingface/speech-to-speech) server. Everything listens on 127.0.0.1 only.

```
Robot mic ─USB─▶ conversation app ──ws://127.0.0.1:8765/v1/realtime──▶ speech-to-speech (GPU with most free memory)
                 (run_app.py; UI on                                        Silero VAD → Parakeet TDT 0.6B v3 STT
                  127.0.0.1:7860)                                                 │
                                                                                  ▼
                                                                Ollama 127.0.0.1:11434/v1: reachy-gemma4
                                                                (gemma4:26b, 32k ctx, thinking off; Ollama
                                                                 picks its GPU, kept loaded by 10 m pings)
                                                                                  │
                                                                                  ▼
Robot speaker ◀─USB─ conversation app ◀────── audio + tool calls ───────── Qwen3-TTS 1.7B (voice "Aiden")
        daemon (run_daemon.py, 127.0.0.1:8000, signalling 127.0.0.1:8443) drives motors/camera over USB
```

**Run it:**

```bash
local_backend/cache_models.sh            # once, while online (face model + motion datasets)
./start_daemon.sh                         # terminal 1: offline, no dataset updates, signalling on localhost
local_backend/start_local_backend.sh      # terminal 2: loads + keeps Ollama model, starts speech server
./start_conversation.sh --local --ui      # terminal 3: the app; UI at http://127.0.0.1:7860
cd local_backend && ../third_party/speech-to-speech/.venv/bin/python -m pytest -v   # 28 checks
```

## Network: what the hosted setup used, and what the local one does

| Path | Now |
| --- | --- |
| Session allocation via `pollen-robotics-reachy-mini-realtime-url.hf.space/session` | Skipped automatically in `local` mode |
| Speech-to-text, LLM and TTS on the hosted backend | `speech-to-speech serve` on this PC, with the LLM on Ollama |
| Camera images analysed by the hosted model | The app sends `input_image`; speech-to-speech's `chat-completions` backend forwards it as `image_url` to gemma4 (vision) |
| Web search, weather, time tools (Pollen HF Spaces over MCP) | Removed from the `local_reachy` profile; local `get_time` added |
| `play_emotion` clips (HF dataset) | Cached; app runs with `HF_HUB_OFFLINE=1` |
| Speech model weights (Parakeet, Qwen3-TTS, Silero, Smart Turn) | Cached by the first run; speech server runs with `HF_HUB_OFFLINE=1` |
| **onnxruntime 1.30 telemetry** in the speech server (`mobile.events.data.microsoft.com`) | Found by `net_watch` + `strace -k`; off via `ORT_DISABLE_TELEMETRY=1` |
| **NLTK index download** from `raw.githubusercontent.com` on every speech-server start | Found by the Python audit hook; fixed by the `~/nltk_data/tokenizers/…` symlink |
| **Daemon: YuNet face model** downloaded on first `head_tracking` | Found by the Fable audit; cached by `cache_models.sh`, daemon runs with `HF_HUB_OFFLINE=1` |
| **Daemon: dataset preload/update** (emotions + dances) at start and every 24 h | Found by the audit; cached, and `--dataset-update-interval 0` |
| Daemon: TURN credentials from HF | Only when an HF token is configured (none here) |
| App web UI and daemon WebRTC signalling listened on 0.0.0.0 (LAN could change settings / request camera+mic) | Bound to 127.0.0.1 by `run_app.py` / `run_daemon.py` |

---

# Original plan (as first drafted; see the Log below for what changed)

## Goal

Keep the conversation app, the daemon and the robot exactly as they are, and replace the cloud pieces with software on this PC, so nothing leaves the machine.

The app already supports this officially. Its `local` connection mode points at a self-hosted [speech-to-speech](https://github.com/huggingface/speech-to-speech) server, the same open-source pipeline the hosted backend is built on.

```
Robot mic ─USB─▶ conversation app ──ws://127.0.0.1:8765/v1/realtime──▶ speech-to-speech (GPU 1)
                                                                         Silero VAD → Parakeet STT
                                                                                │
                                                                                ▼
                                                              Ollama (127.0.0.1:11434/v1, GPUs 0+1)
                                                              qwen3.6:35b-a3b: text, tools, images
                                                                                │
                                                                                ▼
Robot speaker ◀─USB─ conversation app ◀────── audio + tool calls ─────── Qwen3-TTS (voice "Aiden")
```

## What goes over the network today, and what replaces it

| Today (hosted) | Local replacement |
| --- | --- |
| Session allocation via `pollen-robotics-reachy-mini-realtime-url.hf.space/session` | Skipped automatically in `local` mode |
| Speech-to-text, LLM and TTS on the hosted backend | `speech-to-speech serve` on this PC, with the LLM on Ollama |
| Camera images analysed by the hosted model | Same path: the app sends `input_image`, and speech-to-speech's `chat-completions` backend forwards it as `image_url` to a vision-capable Ollama model |
| Web search, weather and time tools (Pollen's Hugging Face Spaces, called over MCP) | Remove from the profile. Add a small local `get_time` tool. Search and weather need the internet by nature. |
| `play_emotion` clips downloaded from the HF dataset `pollen-robotics/reachy-mini-emotions-library` | Download once, then run with `HF_HUB_OFFLINE=1` |
| Model weights (Parakeet, Qwen3-TTS, Silero) | Download once into `~/.cache/huggingface`, then offline |

## Hardware and model choice

This machine has 2× RTX 3090 (24 GB each), a Ryzen 9 5900X and 62 GB of RAM.

| Ollama model already installed | Vision | Tools | Notes |
| --- | --- | --- | --- |
| `qwen3.6:35b-a3b` (Q4_K_M, 23.9 GB) | yes | yes | **Recommended.** Mixture of experts with ~3B active parameters per token, so it should be the fastest of the three. Ollama will split it across both GPUs. |
| `gemma4:26b` (Q4_K_M, 18.6 GB) | yes | yes | Fallback; fits on one GPU. |
| `qwen3.6:27b` (Q4_K_M, 17.4 GB) | yes | yes | Dense 27B, probably slower per turn. |
| `glm-4.7-flash`, `nemotron-cascade-2`, `lfm2`, `qwen3:14b` | no | yes | No vision, so the camera tool wouldn't work. |

The speech-to-speech models go on GPU 1 (`CUDA_VISIBLE_DEVICES=1`): Parakeet TDT 0.6B v3 for speech-to-text, and Qwen3-TTS 1.7B CustomVoice with the GGML backend for text-to-speech. The upstream README budgets ~8 GB of VRAM for the speech side, so the total fits in 48 GB.

## Phases

1. **Install speech-to-speech** in its own venv under `third_party/`.
   - The Qwen3-TTS GGML wheel targets CUDA 12.8 and glibc 2.39. Ubuntu 24.04 has glibc 2.39, and driver 580 supports CUDA 13, so it should work.
   - Check: `speech-to-speech serve` starts and loads all models on GPU 1.
2. **Connect the LLM to Ollama.** Use `--llm_backend chat-completions --responses_api_base_url http://127.0.0.1:11434/v1 --model_name qwen3.6:35b-a3b`, and keep the model loaded (`keep_alive: -1`) so the first reply isn't slow.
   - Check: a scripted Realtime client sends text, a tool definition and an image, and gets back speech, a tool call and a correct description.
3. **Point the app at it** with `.env`: `HF_REALTIME_CONNECTION_MODE=local` and `HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime`.
   - Check: live conversation with the robot, including dance, move_head and the camera.
4. **Remove the remaining internet dependencies:** drop the three Space tools, add a local `get_time`, cache the emotions dataset, and set `HF_HUB_OFFLINE=1`.
5. **Prove it's local.** During a conversation, `ss -tnp` should show only `127.0.0.1` connections for the app, speech-to-speech and Ollama. The final test is to turn off Wi-Fi (`wlp5s0`) and hold a conversation.
6. **Package it.** Add `start_local_backend.sh`, a local option in `start_conversation.sh`, and write up the results and measured latency here.

## Baseline to beat (hosted backend, 2026-09-18)

These times come from the app log of your first conversation. Each is measured from when the transcript of your words was logged to when the reply text was logged. That's slightly earlier than when the audio starts playing.

| You said | Reply / action | Time |
| --- | --- | --- |
| "Tell me a joke." | joke text | 2.4 s |
| "Tell me a joke, G." | joke text | 2.0 s |
| "Can you do a dance for me?" | `dance` tool call (`groovy_sway_and_roll`) | 0.9 s |
| "What do you see?" | `camera` tool call | 0.75 s |
| (camera image attached) | "I see a cozy living room: sofas, a glass coffee table, a laptop…" | 4.4 s after the image |

## Risks and fallbacks

| Risk | Why it matters | Fallback |
| --- | --- | --- |
| Thinking mode | All three vision models are "thinking" models. The `chat-completions` backend has no reasoning flag, so the model may reason before every reply and add seconds. | Pass Ollama's `reasoning_effort`/`think` through if speech-to-speech allows extra request fields. Otherwise use a no-think chat template, or serve the GGUF with llama.cpp `--reasoning-budget 0`. |
| Tool-call reliability | A local ~30B model may call the 16 tools less reliably than the hosted model. | Trim the tool list in the profile, or try `gemma4:26b`. |
| Voice names | The app offers the hosted backend's voice list. | Keep `Aiden`, which is a Qwen3-TTS CustomVoice speaker, and check the others. |
| Latency | Unknown until measured. | Measure from the end of speech to the first audio. Smaller or faster models, or a single GPU per stage, are the levers. |
| Qwen3-TTS wheel on this CUDA setup | The GGML wheel is built against CUDA 12.8. | Use `--qwen3_tts_backend torch`, or `--tts kokoro` (Kokoro-82M, small and fast, but different voices). |

## Not changing

Robot, daemon, `~/.asoundrc`, WebRTC plugin, `start_daemon.sh`. The hosted mode keeps working: switch back by setting `HF_REALTIME_CONNECTION_MODE=deployed` or choosing "built-in server" in the web UI settings.

## Sources

- [speech-to-speech README](https://github.com/huggingface/speech-to-speech) (commit `16d7f98`, 2026-09-06)
- [reachy_mini_conversation_app README](https://github.com/pollen-robotics/reachy_mini_conversation_app), "Hugging Face Connection Modes"
- Source checked: `chat_completions_language_model.py` (converts `input_image` to `image_url`), and the app's `huggingface_realtime.py` (sends the camera frame as `input_image`) and `tools/play_emotion.py` (HF dataset download)

## Log

### Phase 1: speech-to-speech installed (done)

```bash
cd third_party
git clone --depth 1 https://github.com/huggingface/speech-to-speech   # commit 16d7f98
cd speech-to-speech && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
```

This installed PyTorch 2.14.0 with CUDA 13.0 (both 3090s visible), `nano-parakeet` 0.2.1, `faster-qwen3-tts` 0.4.0 and `qwentts-cpp-python` 0.3.1. `libportaudio2` and `libsndfile1` were already installed, and glibc is 2.39.

### Phase 2 (part 1): Ollama tuned for voice

**Turning thinking off.** I tested `qwen3.6:35b-a3b` through Ollama's OpenAI-compatible API with a one-sentence joke prompt:

| Request option | First token | Total | Result |
| --- | --- | --- | --- |
| (default) | none | 1.45 s | Spent 764 characters thinking and hit `max_tokens` with no answer |
| `chat_template_kwargs.enable_thinking=false` | none | 1.37 s | Ignored by Ollama; still thinks |
| `reasoning_effort="none"` | 0.18 s | 0.29 s | Answered immediately |

So the server runs with `--responses_api_reasoning_effort none`. That sends `reasoning_effort` and takes precedence over the `--responses_api_disable_thinking` default, which sends `enable_thinking=false` and only works on vLLM.

**Shrinking the context.** Ollama loaded the model with its full 262,144-token context: 37.5 GB across both GPUs, leaving ~4.5 GB and ~6.8 GB free. I created a variant with a 32k context ([`local_backend/Modelfile.reachy-qwen3.6`](local_backend/Modelfile.reachy-qwen3.6)):

```bash
ollama create reachy-qwen3.6 -f local_backend/Modelfile.reachy-qwen3.6   # undo: ollama rm reachy-qwen3.6
```

It reuses the same weights on disk. Loaded, it takes **24.2 GB** of GPU memory, with ~11 GB left free on each GPU. Loading takes 13 s; once warm, the first token arrives in **0.07–0.14 s**.

Because loading takes 13 s, the model has to stay loaded. By default Ollama unloads it after 5 minutes idle, and every `/v1` request resets that timer. This will be handled in the launcher (phase 6).

### Housekeeping: stopping processes correctly

I made the same mistake twice. `pgrep -f "speech-to-speech serve"` (like `pgrep -f reachy-mini-daemon` earlier) also matches the `bash -c` wrapper that started the process. `kill` then hit the wrapper and left the real server running. That old server kept GPU 1's memory, and the next server crashed loading Qwen3-TTS (`qt_init: pipeline_tts_load failed`).

I also left the hosted-mode conversation app from the first session (pid 48995) running. It was still listening through the robot's microphone and replying through Pollen's cloud backend, which could have mixed with the local tests. It's now stopped.

To stop these reliably, find the Python process itself, for example `pgrep -f "python.*speech-to-speech serve"`, or check listeners with `ss -ltnp`. The launchers in phase 6 will record PIDs.

### Phase 2 (part 2): speech server verified (done)

```bash
cd third_party/speech-to-speech
CUDA_VISIBLE_DEVICES=1 .venv/bin/speech-to-speech serve \
  --device cuda --stt parakeet-tdt \
  --llm_backend chat-completions --model_name reachy-qwen3.6 \
  --responses_api_base_url http://127.0.0.1:11434/v1 --responses_api_api_key ollama \
  --responses_api_reasoning_effort none --responses_api_stream \
  --tts qwen3 --qwen3_tts_backend ggml --qwen3_tts_speaker Aiden \
  --enable_live_transcription --port 8765
```

- **First start** downloads Parakeet, Qwen3-TTS (GGUF from `Serveurperso/Qwen3-TTS-GGUF`) and Silero VAD. Silero comes from a GitHub zipball via `torch.hub`, which matters for phase 5.
- **Qwen3-TTS GGML** works on this CUDA 13.0 / driver 580 setup; the wheel risk didn't materialize. At warm-up it produced first audio in 0.25 s and ran 2.8× faster than real time. The speech side uses ~6.9 GB on GPU 1.
- **Scripted Realtime test** (`session.update` with a `dance` tool, then three turns on one connection):

| Turn | First audio | Result |
| --- | --- | --- |
| "Tell me a joke." | 0.58 s | Correct |
| "Can you do a dance for me?" | 0.31 s | **No tool call:** it said "[dances joyfully]" |
| Robot camera frame + "What do you see?" | 4.38 s | Correct: white couch, a person on a laptop, glass coffee table |

- **Tool calls.** With `OPENAI_LOG=debug`, the request to Ollama contains the `tools` array and `reasoning_effort: none`, so tools are forwarded correctly. Called directly, the model called `dance` 3/3 times. Through the server, in fresh sessions, it called `dance` 6/6 times. The miss above came after a joke turn in the same conversation, so earlier context and temperature 1.0 can push it into acting the move out in words. The profile below adds a rule against that.
- **Image turns** take ~4.4 s to first audio. The app sends the full 1920×1080 JPEG (~730 KB); a smaller image would speed that up.

### Phase 3 + 4: app on the local backend, internet tools removed

**Local profile, no upstream edits.** [`local_backend/profiles/local_reachy/profile.md`](local_backend/profiles/local_reachy/profile.md) is a copy of the default profile with:
- the three Space tools (search, weather, time) replaced by a local `get_time` ([`local_backend/tools/get_time.py`](local_backend/tools/get_time.py), which reads this PC's clock);
- a rule to call movement and emotion tools in the same turn instead of describing them;
- a rule to say briefly that it's offline when asked for web or weather;
- a speech rule: no emojis, markdown or stage directions.

**Emotions cached.** `RecordedMoves("pollen-robotics/reachy-mini-emotions-library")` downloaded 85 clips (8.7 MB), and they load with `HF_HUB_OFFLINE=1`.

**Launch environment:**

```bash
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=$HOME/codebase/robots/local_backend/profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=$HOME/codebase/robots/local_backend/tools
REACHY_MINI_CUSTOM_PROFILE=local_reachy
HF_HUB_OFFLINE=1
```

**Startup log:** `connection mode: local`, `Using direct Hugging Face realtime endpoint ws://127.0.0.1:8765/v1/realtime`, 17 tools with no `Registered remote tool` lines, `Loaded external tool: get_time`, `profile='local_reachy' voice='Aiden'`. It greeted: "Hello! I'm ready to help. What shall we explore today?"

### First live test with qwen3.6: tools mostly ignored → switched to gemma4:26b

The first live conversation (12:29–12:37) exposed three problems:

1. **Tools weren't called.** Of about ten requests, only `camera` and `get_time` fired. For "Show me a dance", the model wrote "Sure, here's my best dance move for you: Performs 'dizzy_spin'" as text. It also promised to raise the volume without calling `volume_control`, and used emojis despite the speech rule.
2. **Speech-to-text misheard** the robot's name and short phrases: "Lee Chi", "Linksy", "Are you tea?", "Can you cheat?", "See all nice ones". Probably "Reachy", which Parakeet doesn't know.
3. **"Your voice is too low."**

**Diagnosis of (1).** The request sent to Ollama was correct: all 17 tools, `tool_choice: auto`, the system prompt with the new rules, and the camera call properly recorded in history. The model chose not to call the tool. I replayed that exact request (`OPENAI_LOG=debug` dump) 6–8 times per variant:

| Model / settings (thinking off unless noted) | `dance` called | Emoji in reply | First output (warm) |
| --- | --- | --- | --- |
| `qwen3.6:35b-a3b`, as shipped (T=1.0, presence_penalty 1.5) | 0/8 | 5/8 | ~0.2 s |
| `qwen3.6:35b-a3b`, T=0.7, top_p=0.8 | 1/8 | 4/8 | ~0.2 s |
| `qwen3.6:35b-a3b`, T=0.3 | 0/8 | 6/8 | ~0.2 s |
| `qwen3.6:35b-a3b`, `reasoning_effort=low` | 0/6 | – | nothing within 300 tokens (all thinking) |
| `qwen3.6:27b` dense | 6/6 | – | 7.5 s (262k context, partly offloaded) |
| **`gemma4:26b`** | **8/8 and 6/6** | 1/8 | **0.19 s** (0.60 s total) |

So the model is now **`gemma4:26b`**, served as `reachy-gemma4` with a 32k context ([`local_backend/Modelfile.reachy-gemma4`](local_backend/Modelfile.reachy-gemma4)). It loads in 10 s and fits on GPU 0 (~20 GB used), leaving GPU 1 for speech (~7 GB). `ollama ps` reports its size as "1.2 GB", which is wrong; `nvidia-smi` shows the real usage. `reachy-qwen3.6` is no longer used; remove it with `ollama rm reachy-qwen3.6`.

Scripted Realtime test on gemma, one connection:

| Turn | First audio | Result |
| --- | --- | --- |
| Joke | 0.34 s | Correct |
| "Can you do a dance for me?" | 0.35 s | `dance {"move":"happy"}` plus "Sure, here's my best dance!" |
| Camera frame + "What do you see?" | 3.92 s | "a living room with a large grey sofa, a round gold coffee table, and a person sitting in a chair under a blue blanket" |

**On (3):** the voice the server sends isn't quiet: it peaks at -2.3 to -2.7 dBFS, with an RMS of -17 to -19 dBFS. The hardware mixer is at 100%. Still to investigate with a side-by-side listen.

**Another mistake:** a cleanup command using `pgrep -f "python.*reachy-mini-conversation-app"` matched and killed its own shell, because the pattern text is in that shell's command line. Use `pgrep -f "[p]ython.*…"` so the pattern can't match itself.

### Tests and GPU tracking

**Test suite:** [`local_backend/tests/test_local_stack.py`](local_backend/tests/test_local_stack.py), 25 checks. It uses the app's real system prompt and 17 tool schemas ([`fixtures/app_request.json`](local_backend/tests/fixtures/app_request.json), captured from the speech server's debug log) and a real robot camera frame (`fixtures/camera_frame.jpg`, which shows the living room).

```bash
cd local_backend && ../third_party/speech-to-speech/.venv/bin/python -m pytest -v
```

- Covers configuration, services, LLM behaviour (thinking off, right tool per request, plain speech, admits it's offline, no fake reminders, camera description), the Realtime path the app uses, no non-loopback connections, and GPU placement and headroom.
- The `realtime` tests need the speech server's single session slot. With the app connected they skip themselves.
- Every run appends latencies, tool hit rates and GPU use to `local_backend/logs/test_results.jsonl`.

The first run had 3 failures, all bugs in the tests:
- The image was sent as a fresh user message; the app sends it after a `camera` tool call, so the model reasonably called `camera`.
- Latency was judged on the first call of a session only.
- The limit for a session's first turn didn't allow for processing the 4k-character prompt and 17 tools.

After fixing those: **24 passed, 1 skipped** (the app-environment check, because the app wasn't running). Numbers from that run:

| Check | Result |
| --- | --- |
| LLM first token, warm | 0.03 s |
| Tool calls (5 runs each) | dance 4/5; move_head, play_emotion, get_time, camera, volume_control 5/5 (0.2–0.6 s each) |
| Camera frame description, direct | 2.1 s |
| Realtime first audio: first turn / tool turn / image turn | 1.36 s / 0.78 s / 2.33 s |
| Voice peak level | -1.1 dBFS |

`dance` at 4/5 sits exactly at the threshold, so it's the one to watch.

**GPU logger:** [`local_backend/gpu_monitor.py`](local_backend/gpu_monitor.py) samples per-GPU and per-process memory and utilisation every 2 s to `local_backend/logs/gpu.csv`. Use `--summary` for peaks.

```bash
python3 local_backend/gpu_monitor.py --interval 2          # record
python3 local_backend/gpu_monitor.py --summary local_backend/logs/gpu.csv
```

Peaks during the live gemma session and the test run:
- GPU 0: 20.2 / 24 GiB (Ollama `llama-server` 19.8 GiB, daemon 0.2 GiB), 3.8 GiB free.
- GPU 1: 6.7 / 24 GiB (speech-to-speech), 17.3 GiB free.

The first version crashed after 8 minutes: a GPU process briefly had an empty `/proc/<pid>/cmdline`, and the name lookup raised `IndexError`. It now handles that and logs and skips any failed sample instead of exiting.

### Live test with gemma4 (12:46–12:50)

- **Latency:** first audio averaged 858 ms after your transcript (9 turns, max 1.4 s), from the app's own `Turn latency` log. The hosted backend took 2.0–2.4 s to produce reply text.
- **Tools:** "Richi go to sleep" called `go_to_sleep` immediately, and the app shut itself down as designed. The `POST /api/apps/stop-current-app failed: HTTP 400` in the log is harmless: the app was launched by hand, not through the daemon's app manager.
- **Language:** you spoke some Kannada ("ardha beka?", "do you want half?"). Parakeet TDT v3 covers only 25 European languages, so it transcribed the sounds as English ("Arte Beka", "The bacca means…"). It also doesn't know "Reachy" ("Richie", "Richi", "Lee Chi", "Linksy"). Candidate fix: Whisper large-v3 (multilingual, includes Kannada) for speech-to-text. Awaiting your decision.

### Interruption at 13:29 and GPU contention

At 13:28 another process on this machine (a separate Claude session) loaded its own Ollama models (`gemma4:26b`, then `gemma4:26b-64k`, up to 35 GB across both GPUs). At 13:29 the speech server and the daemon received stop signals from outside this session. At 15:55, with your OK, I unloaded that session's model (`keep_alive: 0`) and left the session itself alone. Ollama runs its model servers as the `ollama` system user, so this session can't kill them without `sudo` anyway.

**Consequence:** Ollama chooses the LLM's GPU itself, and after the reload it put `reachy-gemma4` on GPU 1. The speech server, hard-coded to GPU 1, then failed with `qt_init: pipeline_tts_load failed` (out of memory). The launcher now loads the LLM first and puts speech on the GPU with the most free memory; the GPU test checks that they're on different GPUs.

### Phase 6: launchers (done)

```bash
./start_daemon.sh                                   # terminal 1
local_backend/start_local_backend.sh                # terminal 2: Ollama model + keep-alive + speech server
./start_conversation.sh --local --ui                # terminal 3: the app, fully local
```

- **`local_backend/start_local_backend.sh`:**
  - creates the Ollama model from its Modelfile if it's missing;
  - loads it and re-pings it every 2 min (`keep_alive: 10m`), so the 10 s reload never happens mid-conversation;
  - picks the speech GPU;
  - starts speech-to-speech with `HF_HUB_OFFLINE=1 ORT_DISABLE_TELEMETRY=1`, logging to `local_backend/logs/speech.log`;
  - Ctrl+C stops everything it started.
  - Settings: `REACHY_LLM` (default `reachy-gemma4`), `REACHY_STT=parakeet|whisper`, `REACHY_SPEECH_GPU`, `REACHY_NET_AUDIT=1`.
- **`start_conversation.sh --local`:** sets local mode, the `local_reachy` profile and tool directories, `HF_HUB_OFFLINE=1` and `ORT_DISABLE_TELEMETRY=1`. Without `--local`, it uses the hosted backend as before.

### Phase 5: proving it's local — two leaks found and fixed

**Tools:**
- [`local_backend/net_watch.py`](local_backend/net_watch.py) logs every TCP/UDP connection of the daemon, speech server and app to `logs/net_watch.log`, marking non-loopback peers `EXTERNAL`.
- [`local_backend/netaudit/sitecustomize.py`](local_backend/netaudit/sitecustomize.py) is a Python audit hook (`REACHY_NET_AUDIT=1`) that logs every non-local DNS lookup and `connect` with its stack trace.
- `strace -f -k -e trace=connect` on the speech server shows native stack traces.

**Leak 1: NLTK index download from GitHub on every start.** `s2s_pipeline.py:66` runs `nltk.download("averaged_perceptron_tagger_eng")` because it checks for the tagger under `tokenizers/`, while the data lives under `taggers/`. The check always fails, and the download fetches the NLTK index from `raw.githubusercontent.com` (185.199.108–111.133). The audit hook caught it with a full Python stack.
- Fix: a data-only symlink, with no code change: `~/nltk_data/tokenizers/averaged_perceptron_tagger_eng -> ../taggers/averaged_perceptron_tagger_eng`.

**Leak 2: ONNX Runtime telemetry to Microsoft.** `net_watch` saw the speech server connect to 20.42.65.85, 4.150.223.101/.114 and (under strace) 13.89.179.12, all port 443 and all Azure. The Python hook saw nothing, so the connections were native. `strace -k` put every one inside `onnxruntime_pybind11_state.so`, on a background thread. onnxruntime **1.30.0**, used by speech-to-speech for the Smart Turn model, ships Linux telemetry (`PosixTelemetry`, Microsoft 1DS client) that uploads to `https://mobile.events.data.microsoft.com/OneCollector/1.0`. The binary's strings include device and OS fields; I haven't decoded the payload. The app and daemon venvs have onnxruntime 1.27.0, which contains no telemetry code.
- Fix: `ORT_DISABLE_TELEMETRY=1` in both launchers.

**Verification after the fixes.** I ran the whole backend launcher under `strace -f -e trace=connect` through startup, 90 s idle, and the full test suite (text, tool and camera turns through the Realtime server). Every `connect()` went to `127.0.0.1` (10×), `/var/run/nscd/socket` (4×) or `/tmp/nvidia-mps/control` (1×). There were **no external connections**, and the audit hook logged none.

**Test changes:**
- New check `test_known_phone_home_paths_are_disabled`.
- The external-connections test now counts unattributed sockets owned by this user as stack sockets.
- The app-config test falls back to the app's startup log.

These are needed because processes launched via `sg` run with a different group ID, and Linux (ptrace access rules) then hides their socket owner and `/proc/<pid>/environ` from other processes of the same user. Logging in again after the group change removes the need for `sg`.

**Why not the Wi-Fi-off test:** you're connected over SSH through that Wi-Fi (from another machine on the LAN), so turning it off would cut your session.

**Another self-kill:** the `[p]ython…` pgrep trick failed when the same command line contained `python3` elsewhere (a heredoc), so the pattern matched its own shell. All scripts now use anchored patterns: `^[^ ]*python[0-9.]* [^ ]*speech-to-speech serve`.

### Speech-to-text comparison (inconclusive on accuracy)

[`local_backend/record_stt_clips.py`](local_backend/record_stt_clips.py) has the robot speak a prompt, beep and record 6 s: 3 fixed English phrases, 1 free English, 2 Kannada. [`local_backend/stt_benchmark.py`](local_backend/stt_benchmark.py) runs each model on GPU 1, 3 repeats per clip; results go to `logs/stt_benchmark.jsonl`.

| Model | Median per clip | GPU memory |
| --- | --- | --- |
| Parakeet TDT 0.6B v3 (current) | 24 ms | 1.8 GB |
| Whisper large-v3 | 354 ms (one 4.6 s repetition loop) | 3.7 GB |
| Whisper large-v3-turbo | 331 ms | 2.1 GB |

**Accuracy couldn't be judged.** Every model transcribed the same unrelated speech ("…reaching out to him, don't. Now that he's on the FBI's radar"), so the clips mostly captured other audio in the room, not the prompted phrases. The `hotwords="Reachy"` hint made no measurable difference. On the "Kannada" clips, Whisper guessed Malayalam, Sinhala, Telugu and English.

**Decision for now:** keep Parakeet (about 0.3 s faster per turn). `REACHY_STT=whisper` switches to Whisper large-v3-turbo with automatic language detection. Re-record in a quiet room to decide properly.

### Fable audit (16:14–16:40) and fixes (16:47–16:55)

A read-only audit by a Fable subagent reviewed every file above. The robot also lost power at about 16:34 (`/dev/ttyACM0` re-created). The daemon then sat in `Motor communication error!` until it was restarted at 16:45; it doesn't reconnect by itself.

| # | Finding | Fix |
| --- | --- | --- |
| 1 (high) | The daemon was outside the locality work. `head_tracking` would `hf_hub_download` the YuNet face model (not cached), and the daemon preloads the emotions **and dances** datasets at start and re-checks them every 24 h. Under `sg` these would show only as unattributed sockets. | [`local_backend/cache_models.sh`](local_backend/cache_models.sh) caches YuNet (pinned revision) and both datasets; verified `FaceDetector()` and the dataset preload work with `HF_HUB_OFFLINE=1`. `start_daemon.sh` now exports `HF_HUB_OFFLINE=1` and passes `--dataset-update-interval 0`; the startup log no longer shows "Dataset updater started". New test `test_daemon_models_cached_and_launcher_offline`. |
| 2 | The "unattributed socket = stack" rule had false alarms: 16:09:06 → 160.79.104.10 and 16:19:31 → 34.149.66.165 were Claude Code (`claude` processes). So the earlier "no external connections" line in `net_watch.log` needed that caveat; the strace run remains the real evidence. | `net_watch` labels these `unattributed` and says they are leads, not proof. `test_no_external_connections` takes 3 snapshots 0.3 s apart and only counts sockets present in all three; its docstring states it's a spot check. |
| 3 | The readiness check could match a stale "Uvicorn running" line in the appended `speech.log`. | Only lines written by this run are checked (line offset recorded at launch); a warning is printed after 180 s. |
| 4 | The app UI (7860) and the daemon's WebRTC signalling server (8443) listened on 0.0.0.0. From the LAN you could change the robot's settings, or request the camera and microphone stream. | New wrappers, no upstream edits: [`run_app.py`](local_backend/run_app.py) rewrites the hard-coded `uvicorn.Config(host="0.0.0.0")`, and [`run_daemon.py`](local_backend/run_daemon.py) sets `webrtcsink`'s `signalling-server-host`. Both default to 127.0.0.1; override with `REACHY_UI_HOST` / `REACHY_SIGNALLING_HOST`. Verified: the UI answers on 127.0.0.1 and refuses on the PC's LAN address. New test `test_ui_and_signalling_not_exposed_to_lan`. |
| 5 | The nvidia-smi GPU index (PCI order) was passed to CUDA (fastest-first order). | `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the backend launcher and the benchmark instructions. |
| 6 | Test limits sat at their edges: voice peak -9.0 dBFS against a -10 limit, dance 4/5 against 4. The offline check accepted any "can't"; `<think>` leakage wasn't checked; the log fallback didn't tie the log to the running app. | Voice peak > -14 dBFS; tool calls 8 runs with at least 6 correct; a weather answer must contain no weather report; `<think>` must not appear in the answer; the log must be newer than the app process. |
| 7 | The audit-hook docs said "every" connection; it only sees Python sockets. | Reworded in the launcher and `sitecustomize.py`. |
| 8 | `sg -c` runs dash, which can't parse `printf %q`'s `$'…'` quoting. | Both launchers refuse arguments with control or non-ASCII characters while `sg` is needed. |
| 9 | Minor: an orphaned `sleep 120`; a silent failure when the model can't load; `gpu_monitor --summary` crashing on `[N/A]`. | `pkill -P` in cleanup; explicit error message; `[N/A]` tolerated. |
| 10 | The udev rule uses `MODE="0666"` (from the official guide), so the motor and audio USB nodes are writable by every local user. | **Not applied: needs sudo.** Suggested: change to `MODE="0660"` in `/etc/udev/rules.d/99-reachy-mini.rules`, then `sudo udevadm control --reload-rules && sudo udevadm trigger`. |
| 11 | Stale docs. | The current setup is now at the top of this file; REACHY_MINI_SETUP.md and CONVERSATION_APP.md updated. |

**Also found while fixing:** the dances dataset (`pollen-robotics/reachy-mini-dances-library`) wasn't cached either; the audit only named the emotions dataset.

**After the fixes, with the daemon on the new launcher:** 28 checks. With the app stopped: 26 passed, 1 skipped, and 1 failure in the new weather check, which had flagged "connection to the clouds" in a correct offline answer. I removed "cloud" from the weather words and the check then passed 5 of 5 times. With the app running, the 11 checks that need it all passed. Listening ports: 8000, 8443, 8765 and 7860, all on 127.0.0.1.

### "Not picking up sounds, responding slow" (16:53–17:02)

**Measured,** from `speech.log` after the 16:53 restart:
- The voice detector **discarded 9 speech segments and accepted 7**. The discarded ones were 0.3–0.95 s long, but only 64–352 ms of each scored as speech (threshold `thresh=0.6`), below `min_speech_ms=384`. Those utterances never reached speech-to-text.
- One transcribed "user" turn was a TV weather broadcast ("It's 71 degrees Fahrenheit with clear skies…").
- **Latency is steady:** 0.86 s from the end of speech to the first audio out (4 of 4 turns). That's Smart Turn 48 ms, Parakeet 25 ms, **LLM 0.68–0.78 s**, TTS first audio 30 ms, plus about 0.17 s of audio output buffering in the app.
- **Ollama streams fine** with the app's prompt and 17 tools: first token 0.15 s and full reply 0.30 s when warm, 0.5–1.0 s when the prompt must be processed again. The rest of the LLM time was `stream_batch_sentences=3`, which waits for 3 sentences (or the whole short reply) before starting TTS.

**VAD threshold on this robot's mic** (Silero over the recorded clips; ms of frames above the threshold):

| Clip | 0.6 | 0.5 | 0.45 | 0.4 | 0.3 |
| --- | --- | --- | --- | --- | --- |
| speech.wav ("Hello? Can you hear me?") | 544 | 704 | 768 | 896 | 992 |
| en_free.wav | 608 | 832 | 832 | 832 | 864 |
| kn_free_1.wav | 864 | 928 | 960 | 960 | 992 |

Going from 0.6 to 0.45 adds 0–41% detected speech per clip; going lower adds little. Assuming the same gain, about 6 of the 9 discarded segments would have passed a 256 ms minimum.

**Change:** new launcher settings, passed to speech-to-speech:

| Setting | Was | Now |
| --- | --- | --- |
| `REACHY_VAD_THRESH` (`--thresh`) | 0.6 | 0.45 |
| `REACHY_VAD_MIN_SPEECH_MS` (`--min_speech_ms`) | 384 | 256 |
| `REACHY_STREAM_SENTENCES` (`--stream_batch_sentences`) | 3 | 1 |

**Trade-off:** it's now easier for TV and background speech to trigger a turn. Raise `REACHY_VAD_THRESH` again if that happens.

**Another mistake:** I edited `start_local_backend.sh` while the old instance was still running. Bash reads scripts as it goes, so when its speech server exited it read the changed file and exited with code 127 ("command not found") mid-cleanup. No keep-alive loop was left behind. Don't edit launcher scripts while they're running.
