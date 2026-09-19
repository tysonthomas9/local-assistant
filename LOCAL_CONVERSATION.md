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
# or, with online tools (weather, web search, tech news):
local_backend/start_searxng.sh            # local SearXNG search engine in Docker, 127.0.0.1:8888
./start_conversation.sh --web --ui        # --web implies --local; profile local_reachy_web
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
| Reminders and timers (both profiles) | Local only: the scheduler thread calls the app's own `conversation.say` on 127.0.0.1:7860 |
| **Opt-in online radio** (`--web` only) | `play_radio` → station lookup on radio-browser.info, then the station's audio stream |
| **Opt-in online tools** (`--web` only) | `get_weather` → Open-Meteo (place name + coordinates); `web_search` → local SearXNG, which queries public engines; `tech_news` → Hacker News, Ars Technica and The Verge RSS. Speech, the LLM, the camera and the conversation itself stay on this PC. The default `--local` profile has none of these. |

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


### Online tools: weather, web search, tech news (17:05–17:20)

You asked for these after the fully local work, so they're **opt-in**. The offline `local_reachy` profile is unchanged. A new `local_reachy_web` profile adds the three tools and is selected by `./start_conversation.sh --web` (which implies `--local`). The design choices were yours: a separate profile, SearXNG for search, and tech news RSS.

| Tool | File | Uses | Notes |
| --- | --- | --- | --- |
| `get_weather` | [`tools/get_weather.py`](local_backend/tools/get_weather.py) | Open-Meteo geocoding + forecast (free, no key) | Fahrenheit by default (`REACHY_WEATHER_UNITS=celsius` to switch); `REACHY_HOME_LOCATION` is used when no place is named; "City, State" picks the matching region |
| `web_search` | [`tools/web_search.py`](local_backend/tools/web_search.py) | Local SearXNG on 127.0.0.1:8888 ([`start_searxng.sh`](local_backend/start_searxng.sh), Docker `searxng/searxng:latest`) | Top 5 results (title, snippet, source); a clear error if SearXNG isn't running |
| `tech_news` | [`tools/tech_news.py`](local_backend/tools/tech_news.py) | RSS/Atom feeds in [`tech_news_feeds.json`](local_backend/tech_news_feeds.json) | Hacker News, Ars Technica, The Verge, fetched in parallel; optional `source` and `count`; parsed with the standard library |

**SearXNG setup:**
- The container is published on `127.0.0.1:8888` only, with JSON output enabled and the rate limiter off (it's private).
- A random secret key is generated on first start into `local_backend/searxng/settings.yml`, which is git-ignored. The template is [`searxng.template.yml`](local_backend/searxng.template.yml).
- No `--restart` policy, so it doesn't start at boot.
- The container chowns its mounted config dir to its own uid 977. The first version kept the committed template inside that dir, which left it unwritable for you. I fixed the ownership once with a root `chown` inside the image, and moved the template outside the mounted dir.
- First query: 28 results. DuckDuckGo answered with a CAPTCHA and Wikidata was suspended; the other engines covered it.

**`start_conversation.sh` argument handling** was rewritten as a `case` loop, so `--web` works with or without `--local`. Dry-run: `--ui` → hosted; `--local --ui` → `local_reachy`; `--web --ui` and `--local --web --no-camera` → `local_reachy_web`, with the flags removed before the app sees them. With `--web`, it warns if SearXNG isn't answering.

**Tested:**
- **Tools called directly:** Paris 59°F overcast (high 77°F); "San Jose, California" → San Jose, California, US; no place / nonsense place → clear errors; the search returned Pollen's product page; news returned headlines from all three feeds, and The Verge alone when filtered.
- **New tests** (8): the profile diff; the live tool check (`-m online`); and tool choice on 6 prompts × 8 runs with the web profile's prompt and all 20 tools. **All 8/8**: get_weather (Tokyo weather; rain in Seattle), web_search (last F1 winner; Raspberry Pi 5 price), tech_news (tech news today; Hacker News).
- **The rest of the suite** (services not running, since the daemon and speech server were stopped at 17:05): 28 passed.
- **Not yet tried live with the robot.**


### Scheduler (reminders, timers) and radio (2026-09-19)

**Scheduler:** `set_reminder`, `list_reminders`, `cancel_reminder`. They're local, so they're in **both** profiles.
- **Why not a tool that just sleeps until the due time:** the app sends a tool's result to the model only when the tool finishes, and the speech server refuses new responses while a result is pending ("Cannot create a response while function call outputs are pending", seen in phase 2). A sleeping reminder would freeze the conversation.
- **How it works instead:** the tools return at once. A background thread in [`reachy_scheduler.py`](local_backend/reachy_scheduler.py) fires each reminder through the app's own JSON-RPC method `conversation.say` (WebSocket `ws://127.0.0.1:7860/rpc`, the same one its web UI uses; it exists only with `--ui`). The injected text is `(Reminder due now) <message>`. A profile rule tells the model to announce it, starting with "Reminder:".
- **Time formats:** "in N minutes", or a clock time ("17:30", "5:30 pm", "9am", "noon"). A time already past today means tomorrow.
- **Storage:** reminders are kept in `local_backend/state/reminders.json` (git-ignored), so they survive an app restart. After a restart, one up to an hour late is still announced ("it was due at 8:17 AM"); older ones are dropped and logged. Delivery is retried 3× at 5 s intervals if the app is busy or restarting.
- **Missing UI warning:** if `/rpc` isn't reachable when a reminder is set (app started without `--ui`), the tool says so.

**Radio:** `play_radio`, `stop_radio`, in `--web` only (they need the internet).
- **How it plays:** [`reachy_radio.py`](local_backend/reachy_radio.py) looks up stations on radio-browser.info (free, no key; by name first, then by genre tag; MP3/AAC/OGG only; most popular first). It plays them with a GStreamer `playbin` inside the app, into the shared `reachymini_audio_sink` dmix, so the robot's voice mixes over the music.
- **Volume and fallback:** default volume 30% (`REACHY_RADIO_VOLUME`). If a stream won't start within 8 s, it tries the next station, up to 3. A watcher thread cleans up if the stream drops.
- **No spoken reply to stop:** `stop_radio` has `needs_response = False`, so "be quiet" doesn't get a spoken reply.
- **Untested risk:** the microphone may pick up the music and start turns. The audio board's echo cancellation should remove its own output, but this hasn't been tested live yet.

**Loader detail:** the app finds external tools by file name (`<tool_name>.py`) and re-runs those files when the profile changes. So each tool is its own small file, and the shared state (scheduler thread, radio player) lives in normally imported helper modules that are loaded once.

**Tests** (39 passed, services not running):
- **Direct tool runs:**
  - Parsing: 7 good times, and 6 bad inputs rejected with clear messages.
  - A reminder fired after 3 s into a fake `/rpc` server, which first sent a notification that was correctly ignored.
  - Cancel by words; a 5-minute-late reminder is announced, a 2-hour-late one dropped.
  - Radio: "jazz" → 101 Smooth Jazz, playing in 1.4 s (to a `fakesink`); stop, and a double stop; a nonsense name gives an error.
- **The test suite now builds prompts from the profile files.** The captured request gives the app's wrapper around the profile text (`system_prefix`/`system_suffix` in the fixture), and tool schemas missing from the capture are loaded from `local_backend/tools`. Tests no longer go stale when a profile changes.
- **New tests:** scheduler end to end; "announces a due reminder" (no tool call, mentions the reminder); radio play and stop (`-m online`); and tool choice.
- **Tool choice, 8 runs each:** set_reminder 8/8 ×3 (in 10 minutes; 5-minute timer; at 6 pm), list_reminders 8/8, cancel 8/8 (listing first also counts), play_radio 8/8 ×2 (jazz; BBC Radio 1), stop_radio 8/8, and a timer in the web profile 8/8. The old checks: dance 7/8, all others 8/8.
- The old `test_llm_does_not_promise_reminders` was retired, since reminders now exist.
- **Not yet tried live with the robot.**


### Sound effects: alarm, timer, chime… (2026-09-19)

**New tools** (both profiles, fully local): `play_sound` (a name, plus `repeat`, where 0 means until stopped, max 60 s; and `volume`, default 70) and `stop_sound` (`needs_response = False`).

**`set_reminder` got a `sound` argument:** `chime` (default), `timer`, `alarm`, any sound name, or `none`. When a reminder is due, [`reachy_scheduler.py`](local_backend/reachy_scheduler.py) rings it first, then the robot speaks. An `alarm` repeats until `stop_sound` or `REACHY_ALARM_SECONDS` (default 30; it's checked after each ~2 s cycle, so it can run up to one cycle over). Where GStreamer bindings are missing, as in the test venv, the scheduler just speaks.

**Library** ([`reachy_sounds.py`](local_backend/reachy_sounds.py)):
- **Synthesised with numpy on first use** into `local_backend/sounds/generated/` (git-ignored, deterministic, license-free), 16 kHz like the audio board, peaks at -3 dBFS:

| Sound | Length | What |
| --- | --- | --- |
| alarm | 1.94 s per cycle | 4 fast double-beeps (988 Hz, softened square) + pause |
| timer | 2.85 s | three bright bell dings + a longer one (1568 Hz) |
| chime | 2.10 s | rising C6-E6-G6 bells |
| bell | 2.00 s | one struck bell (880 Hz, inharmonic partials) |
| beep / success / error | 0.30 / 0.37 / 0.57 s | short UI sounds |

- **Plus the SDK's bundled robot sounds** (wake_up, go_sleep, dance, confused, impatient, count), and **your own files**: any .wav/.mp3/.ogg/.flac in `local_backend/sounds/custom/`, by file name, overriding built-ins with the same name.
- **Names match loosely:** "alarm clock" → alarm, "kitchen timer" → timer. An unknown name returns an error listing what's available.
- **Playback:** a GStreamer playbin into the shared `reachymini_audio_sink`, so sounds mix with the robot's voice and the radio. One sound at a time; a new one replaces the old.

**Testing gotcha:** a plain `fakesink` isn't clocked, so it consumed the 2.85 s timer in 0.01 s. The first reminder test therefore looked as if the reminder spoke without waiting for its ring. With `fakesink sync=true` (silent, but real time, like the speaker) the timings are right: timer due at 1 s → spoken at 3.9 s; alarm (4 s limit) due at 6 s → spoken at 11.8 s; `none` due at 13 s → spoken at 13.0 s.

**Tests** (48 passed, services not running):
- **Library and timing:** the library is present; the alarm rings and stops; an unknown name errors; the timer ring lasts ~2.85 s; the alarm respects its limit; `none` returns at once.
- **Tool choice:** "wake me up at 7 am" → set_reminder, "ring the alarm" / "play a bell sound" → play_sound, "stop the alarm!" → stop_sound, **8/8 each**.
- **Sound argument:** timers → `timer`, alarms and wake-ups → `alarm`, **32/32**.

**Also:** during the run, one `web_search` for "Reachy Mini robot" returned 0 results; a minute later the same query returned 27. SearXNG's upstream engines sometimes all time out at once (DuckDuckGo keeps answering with a CAPTCHA, and Wikidata times out). `web_search` now retries once when there are no results.

**Not yet heard on the robot:** all sound tests used silent sinks.

### web_search pagination (2026-09-19)

**Before:** `web_search` fetched only SearXNG's first page and passed its top 5 results to the model. There was no page parameter, so "show me more" re-ran the same search.

**Now:** an optional `page` argument (1, 2, 3…, up to 10), 5 results per page; each response carries `page` and `has_more`.
- **Why not map pages directly:** SearXNG's own pages are uneven and overlap. For "reachy mini robot", pageno 1/2/3 had 27/39/36 results. So the tool fetches SearXNG pages in order, only as far as needed (at most 5 per query), drops repeated URLs (ignoring scheme, `www.` and a trailing slash), and keeps the merged list in a per-query cache for 5 minutes. The query key ignores case and extra spaces.
- **What that gives:** page 2 continues exactly where page 1 stopped, and follow-up pages usually need no new search.
- **Failures:** a later SearXNG page that fails returns what's already collected and is retried on the next call. The first page still retries once when all engines return nothing.
- **Direct answers** (SearXNG `answers`) are only included on page 1.

**Tests** (51 passed, services not running):
- **Fake SearXNG** with pages of 7/4 (one a duplicate in another spelling)/6/0 results. Tool pages come out as 0–4, 5–9, 10–14, [15] with `has_more=False`, then "No more results." Page 3 and a repeat of page 1 made no new SearXNG requests. No duplicates across pages.
- **Real SearXNG:** page 1 has 5 results and `has_more`; page 2 shares nothing with it (`-m online`).
- **Model follow-up:** after a search turn, "Can you show me more results?" → `web_search` with the same query and `page: 2`, **8/8**.

### Batch 1 of FEATURE_PLAN.md: bridge, calculator, lists, personas (2026-09-19)

**Bridge** ([`reachy_bridge.py`](local_backend/reachy_bridge.py), installed by `run_app.py`):
- **What it wraps:** `LocalStream.__init__` (keeps the stream), `_dispatch_activity` (publishes activity reasons to subscribers) and `clear_audio_queue` (publishes `interrupted`).
- **Helpers:** `wait_for(reasons, timeout)` from background threads, `run_in_app_loop`, and `mic_muted` / `set_mic_muted`.
- **Why the activity hook survives persona switches:** the handler gets `self._dispatch_activity` as its observer on every rebuild.
- **Unit-tested** against a real `LocalStream` with stub handler and robot.

**Calculator:** `calculate` (both profiles).
- An `ast` whitelist, no `eval`. It accepts "15% of 80", "x", "^", "plus/minus/times/divided by", "squared", and trig in degrees.
- Numbers, exponents and results are capped. Found by the tests: a 400-digit literal wasn't capped and crashed while being formatted; it's now rejected.
- Code injection, attribute access, unknown names, division by zero and 500 nested parentheses all return errors.

**Unit conversion:** `convert_units` (both profiles), a hand-written table: length, mass, volume including US cooking measures, temperature, area, speed, data, time; with aliases and plurals. Examples: 72 °F → 22.22 °C; 3.5 cups → 828.1 ml; kg → celsius is an error.

**Currency:** `convert_currency` (web only).
- **Endpoint:** `https://api.frankfurter.dev/v1/latest` (ECB rates). `api.frankfurter.app`, the address in the plan, now answers 301.
- **Behaviour:** cached for a day in `state/`; understands currency names ("euros", "yen", "rupees"). 50 EUR → 57.30 USD (rate of 2026-09-18).

**Lists:** one `lists` tool (both profiles).
- **Actions:** add (one or several items; case-insensitive duplicates reported), remove (by words), read, clear, list_lists.
- **Names normalised:** "Shopping List" / "groceries" → shopping; "to do" → todo.
- **Stored** in `state/lists.json`.

**Personas:** [`make_personas.py`](local_backend/make_personas.py) generates `local_<name>` and `local_<name>_web` for the 12 visible upstream personas (24 files, committed; a test fails if they're stale). `default` is replaced by our base profiles, and hidden `tedai` is skipped.
- **Each persona profile contains:** the persona's own text under a "## PERSONA" heading, our base profile's tool list, and its "TOOL & MOVEMENT … SPEECH RULES" sections. All 26 profiles load through the app's own parser, and every listed tool exists.
- **Voices:** the GGUF header contains all 9 CustomVoice speakers (Aiden, Ryan, Dylan, Eric, Ono_Anna, Serena, Sohee, Uncle_Fu, Vivian). An unsupported voice would only log a warning and keep the current one (`qwen3_tts_handler.py` `_apply_session_voice_override`). The assignments are in `VOICES` in the generator.
- **`switch_persona`** (all profiles): loose matching (a synonym table + difflib), and it keeps the offline/web mode. It returns at once; a background thread waits for `assistant_transcript_done` (8 s maximum), then calls `stream.apply_personality` on the app loop (the same call the UI's `personalities.apply` ends in). The session restarts, so conversation history is lost. "a pirate" returns "no match" with the list of options.

**Tool counts:** 24 offline, 30 web, plus the app's 2 `task_*` tools.

**Tests: 73 passed** (services not running). **Every tool-choice check was 8/8**, dance included (7/8 before):
- new cases: 17×23, 15% of 240, cups → ml, °F → °C, shopping add, to-do read, detective, butler, 50 € → $;
- the `lists` arguments: action, list name and item, 4 prompts;
- a persona keeping its tools: in `local_victorian_butler` — time, dance, "go back to being yourself", 12×12.

**Not yet tried live:** the persona switch (the session restart and greeting, and whether the voices sound right).

### Batch 2: privacy mute (2026-09-19)

`listening` tool (both base profiles and all personas): `action=stop|resume|status`, `minutes` (default 60, 0 = until resumed), `hard`.
- **Mechanism:** [`reachy_listening.py`](local_backend/reachy_listening.py) sets `LocalStream._mic_muted` through the bridge. The app's `record_loop` then drops every microphone frame before sending (`console.py:881`), so no audio leaves the process. The robot can still speak, so confirmations, reminders and the radio work while muted.
- **Un-muting:** the timer, the web UI's mic toggle, or, once built, the wake word (`hard=true` will make the wake word ignored).
- **Muted pose:** antennas at [-2.2, +2.2] rad and head pitched down 12°.
  - **Values** taken from Pollen's recorded emotions (downcast1, yes_sad1 and sad2 have antennas ≈ ±2.2–2.7 and head pitch ≈ 20°; positive pitch = head down, checked with `create_head_pose`).
  - **Why a `BreathingMove` subclass:** only an idle `BreathingMove` gives way to a queued move (`moves.py:457`); any other endless move would block every dance behind it. A watcher re-queues the pose every 3 s when the robot is back to plain breathing, and clears it on unmute.
- **Tests (79 passed):**
  - Mute/pose/timer/resume against a real `LocalStream` with a fake movement manager.
  - Tool choice for "stop listening", "don't listen for the next hour", "mute yourself for 10 minutes", "are you listening?": all 8/8.
  - "Stop listening for 10 minutes" → `action=stop, minutes=10`, 8/8.
  - Every other tool-choice check still 8/8.
- **Needs the robot:** whether the pose looks right, and confirming in `speech.log` that no turns arrive while muted.

### Batch 3: storyteller / book reader (2026-09-19)

**The key experiment:** an out-of-band response (`response.create` with `conversation: "none"`, the passage as `input`, and instructions to read it "exactly as written, word for word") on the real speech server:
- **Accuracy:** the 148-word opening of *Alice* came back with **word error rate 0.000**.
- **Speed:** first audio in 0.61 s; ~30 s of audio generated in 8.5 s.
- **Kept out of the conversation:** `conversation_id` was null, and a following in-band question ("did I just ask you to read a story?") got "No".

So the reader uses the robot's own voice, word for word, and the Piper fallback isn't needed. Out-of-band responses also skip the speech server's pending-tool-result check, so reading never blocks the conversation.

**[`reachy_reader.py`](local_backend/reachy_reader.py) + `read_book` tool** (both base profiles and all personas):
- **Actions:** `start` (title, optional chapter; resumes from the bookmark unless `from_start`), `continue`, `stop`, `chapter`, `list`, `status`, `download` (web profile only).
- **Parsing:**
  - Strips the Gutenberg licence header and footer and `[Illustration]` markers.
  - Detects headings (CHAPTER/BOOK/PART/STAVE/LETTER + Roman or Arabic numbers, converted: "CHAPTER IV." → "Chapter 4."). Heading-like lines inside a table of contents are skipped, because they aren't standalone paragraphs.
  - ~150-word passages split at sentence boundaries. The first splitter used a variable-width look-behind, which Python's `re` rejects; it now splits on a captured punctuation group.
  - *Alice*: 12 chapters, 218 passages (median 129 words), ends at "THE END".
- **Title search:** the share of the query's key words ("the wonderland book" → Alice), with a fuzzy fallback. "moby dick" → no match, with an offer to download it.
- **Bookmarks** are in `state/books.json`, and reading resumes where it stopped. After an interruption, the interrupted passage is re-read.
- **Pacing:**
  - The bridge now also wraps the SDK's `MediaManager.push_audio_sample` to keep a playback clock (`audio_seconds_left()`; reset on barge-in).
  - Each passage is sent after the previous one has been generated **and** less than 6 s of its audio is left. Generation runs ~3.5× faster than speech, so without this, audio would pile up and keep playing after an interruption.
  - The reader waits for the tool's spoken confirmation to finish before starting, and any `interrupted` / `user_speech_started` pauses it.
- **Downloads:**
  - Gutendex, the third-party API in the plan, timed out, so the reader uses **Gutenberg's own OPDS search feed** and `/ebooks/<id>.txt.utf-8`.
  - It skips hits with no plain-text edition (e.g. illustration collections) and saves into `local_backend/books/`. That folder is git-ignored apart from its README; *Alice* was downloaded there for the tests.
- **Bedtime stories:** a profile rule says to make them up, a few sentences at a time, without `read_book`.

**Tests (92 passed + the realtime group):**
- **Parsing** (offline).
- **Gutenberg download** of *The Time Machine* (`-m online`).
- **Tool choice** for "read me Alice", "keep reading", "stop reading", "what books do you have?": 8/8 each. "Tell me a bedtime story" doesn't call `read_book`: 8/8.
- **Reader integration against the real speech server,** with a stand-in app that feeds the playback clock at 2× real time. The speed-up has to stay below generation speed; a first try at 10× never buffered anything, so pacing went untested. Results:
  - 2 passages read with a normalised word match of **1.0**; the raw token comparison with punctuation gave 0.96;
  - at most ~21 s of audio buffered, under one passage;
  - a barge-in paused the reader, bookmarked the passage that was playing, and `reading` went false.
- **Slot handoff:** the reader test first skipped because the previous test's session hadn't released the server's single slot yet (the server sends an error event, then closes). It now retries for up to 15 s.

**Not yet tried live:** does the robot hear its own reading? It depends on the audio board's echo cancellation, and the SDK adds none of its own with our `~/.asoundrc`. If it does, the reader will keep pausing itself; the wake word (next) would fix that.

### Batch 4: wake word (2026-09-19)

**openWakeWord:**
- **Version:** `uv pip install openwakeword` in the app venv resolved **0.4.0**, not the 0.6 the plan mentioned. It runs on onnxruntime (1.27 is already there) and ships its models **inside the package** (alexa, hey_jarvis, hey_marvin, hey_mycroft, plus timer/weather), so nothing is downloaded at runtime. It isn't in the app's `uv.lock`, so re-run `uv pip install openwakeword` after re-syncing that venv.
- **Tested on Piper-synthesised clips:**

| Clip | hey_jarvis | hey_mycroft |
| --- | --- | --- |
| "Hey Jarvis, what time is it?" | 0.999 | 0.000 |
| "Hey Mycroft, play some music." | 0.000 | 0.989 |
| "Hey Reachy, tell me a joke…" | 0.007 | 0.002 |
| "What's the weather like today?" | 0.000 | 0.000 |

- **Cost:** 1.4–1.8 ms of CPU per 80 ms frame.

**Gate** ([`reachy_wake.py`](local_backend/reachy_wake.py)): enabled by `./start_conversation.sh --wake` (default `hey_jarvis`; `REACHY_WAKE_WORD` selects another bundled word or a custom `.onnx`). `run_app.py` then replaces `LocalStream.record_loop` with the same loop plus the gate:
- **Detection:** every mic frame goes to the detector, including while muted, but not when hard-muted.
- **Closed:** frames only go into a 1.5 s ring buffer and never leave the process. While muted, not even that.
- **On the wake word:** un-mute if muted, send the ring buffer (so the words after the wake word survive), open an 8 s window.
- **What keeps the window open:** the user speaking or being transcribed, tool calls, reminders (`say`), and the robot answering. After a reply it stays open 8 s more for follow-ups and barge-in.
- **Book reading doesn't keep it open,** so the robot can't hear its own reading; "Hey Jarvis, stop" still works.
- **Profile rule:** a leading wake word in a transcript isn't part of the request.

**Tests** (94 passed, 1 skipped because the app wasn't running):
- `test_wake_word_gate` generates its clips with Piper, then checks: non-wake speech forwards 0 s; the wake clip forwards and opens; the window closes when quiet; an answering robot keeps it open; muted + wake word un-mutes; hard-muted + wake word forwards nothing; reading doesn't hold the window.
- Tool choice after all six features (web profile 33 tools incl. the app's 2): all 8/8 except dance 7/8.

**Live checks still needed on the robot** (all six features are built and tested offline):
1. **Does the robot hear itself** during book reading and the radio (is there echo cancellation on the audio board)? Watch `speech.log` for VAD turns while it reads.
2. **Wake word:** false triggers with the TV on, the detection rate from across the room, and whether "Hey Jarvis" needs to be louder than normal speech.
3. **Muted pose:** does it look right? And no turns in `speech.log` while muted.
4. **Persona switch:** the session restart, the greeting, and whether the voices suit the characters.
5. **Radio and sounds:** loudness on the robot's speaker.

### Live test and the second Fable review (2026-09-19, 10:07–10:30)

**Live test** (`--web --wake`):
- **What worked:** the wake word (speech before it never reached the speech server); follow-ups inside the window without the wake word; `lists` add/read; `convert_units` (3.5 cups → 828.1 ml); `read_book` start and resume from the bookmark.
- **Speech-to-text errors** (Parakeet): "Hey Jarvis" → "eight jar is"; "and eggs" → "annex"; "eggs" → "X two" (added to the list as-is).
- **Couldn't interrupt the robot while it read.** From 10:09:40 to 10:12:23 there were no wake detections and no VAD events.
  - **Measured:** playing speech through the robot while recording its mic gives -39.9 dBFS during playback, against -35.1 dBFS in the quiet room. The board's echo cancellation removes the robot's own voice (good: it can't hear itself), but its post-processor also gates the user while the robot talks.
  - **Stopgap (still in place):** a 1.5 s pause after each 110-word passage. You rejected it: "how will I time the 2-second pause". A real fix is below.
- **Stopping didn't cut the passage in progress:** only local audio was flushed. `read_book stop` now cancels the passage on the speech server too (`READER.stop_now`).
- **A bare "Hey Jarvis" with a pause after it was answered in full,** and the follow-up window counted from the end of *generation*, while the reply was still playing and the mic suppressed. Both fixed below.

**Fixes for the review's findings** (all 17), plus the live-test ones:

| # | Finding | Fix |
| --- | --- | --- |
| 1 | The calculator removed every comma, so `min(1,2)`, `max`, `round(x, n)` broke and `round(1, -10**9)` returned -999999999 | Only thousands separators are removed; `round` digits are capped at ±20 (`round(1, -10**8)` would compute 10^100000000); function results are checked against the size cap; `min(5)` is rejected |
| 2 | The reader skipped a passage and moved the bookmark on when no transcript arrived | A timeout now pauses reading at that passage |
| 3 | A stop/start race could overwrite the new bookmark, and bookmark writes were unlocked | Only the current generation writes, under the reader lock; `save_mark` has its own lock; `stop()` saves the position |
| 4 | Sentences like "Part of the reason…" and "Chapter Mix" became chapters | A heading needs a real number (digits, number words or ordinals, or a *capitalised*, valid Roman numeral). A paragraph naming several chapters is a table of contents: the committed excerpt's 2-line contents was parsed as a chapter until this. |
| 5 | The `push_audio_sample` wrapper could raise inside the app's play loop | All bookkeeping is inside `try`; samples counted as the longer axis ((n,), (n,ch), (ch,n)) |
| 6 | A persona switch didn't stop the reader | `_apply_later` stops it first |
| 7 | The web UI's mic toggle left stale mute state (a hard flag, timer) | `_sync_with_mic()` reconciles on every status/hard-mute check |
| 9 | A re-mute could knock off the new pose | The pose is only cleared when really un-muted |
| 10 | The muted pose started from hard-coded antenna positions | Uses `last_primary_pose` antennas |
| 11 | Superseded mute timers slept for up to 24 h | Each timer waits on an Event that the next mute/resume sets |
| 12 | `--wake` applied in hosted mode; a missing openwakeword gave a traceback | `--wake` needs `--local`/`--web` (otherwise a note); `run_app.py` gives a clear install message |
| 13 | The wake ring buffer kept pre-window audio | Cleared whenever the window opens. My first version also cleared the pre-roll *on detection*, before it was sent; caught while writing the test and fixed (the pre-roll is taken first). |
| 14 | Lists and currency edge cases | Corrupt `lists.json` is backed up; removal prefers exact/plural matches ("egg" removes "eggs", not "vegan eggs"); "my/the shopping list" → shopping; currency guards bad responses, locks and writes the cache atomically |
| 15 | `sys.path` grew on every profile reload | Guarded in every tool file |
| 16 | Test gaps | Committed a public-domain excerpt fixture ([`tests/fixtures/books/alice_excerpt.txt`](local_backend/tests/fixtures/books/alice_excerpt.txt)), so the parsing and reader tests always run. New tests: two-argument calculator and the rounding cap; lists priority/names/corruption; mute following the UI toggle; the bridge clock with odd frames (it was vacuous in my first draft; rewritten to go through the real wrapper); a reader that never gets a transcript pauses. The reader integration test's bookmark check is now deterministic. |
| 8, 17 | Reminders pause a book (reminder `say` flushes audio); doc drift | Documented here; reading resumes with "keep reading" |
| live | Bare wake word; follow-up window too short | Profile rule: a bare wake word gets "Yes?". The follow-up window is counted from the end of *playback* (+10 s). Near-miss wake scores (0.2–0.5) are logged. |

**Tests:** 95 passed without the speech-server group, and every tool-choice check was 8/8. Then 5/5 realtime tests with the app stopped: reader passages matched 1.0, max buffered 9.2 s.

### Echo-cancellation channel experiment and the third Fable review (2026-09-19, 10:31–10:50)

**Question:** can "Hey Jarvis" be heard while the robot reads, if we listen to the audio board's *unsuppressed* echo-cancelled signal instead of its processed output? The XVF3800 can route the linear AEC residual (category 7) to one USB channel: `audio_control_utils.py AUDIO_MGR_OP_R --values 7 0` (default `8 0` = processed; not saved, so a power cycle resets it too).

**Test:** right channel set to 7, 45 s stereo recording while the robot played a cue and a 40 s passage, and the user said "Hey Jarvis, stop" three times. Scored offline with the same openWakeWord model:

| | Channel 0 (processed; what the app uses) | Channel 1 (AEC residual) |
| --- | --- | --- |
| User's "Hey Jarvis" during reading | **1.00 (detected)** at 23.3 s; near miss 0.42 at 27.8 s | 0.37 / 0.38 near misses only |
| Robot's own "Hey Jarvis" (in the cue) | ignored | 0.36 (echo leaks through) |
| Level during reading | -25.8 dBFS | -34.8 dBFS |

**Result:** the residual channel is worse on both counts, so there is no second detector on it; the board is back to `8 0`. The processed channel *does* let a clear "Hey Jarvis" through during playback (1 of 3 tries); the live app detected it at 10:33:20. The words after it came out as "See that oh.", so "stop" itself was lost. Next idea (not built): on a wake detection while the robot is speaking, stop playback at once instead of relying on the transcribed "stop".

**Wake threshold:** a later "Hey Jarvis" scored 0.48 and was ignored, so the default is now 0.4 (`REACHY_WAKE_THRESHOLD` overrides it).

**Third review (Fable, of e11dbe6):** 14 of the 17 earlier findings fixed, 2 partial (4, 17), 1 documented only (8). New findings and fixes:

| # | Finding | Fix |
| --- | --- | --- |
| 1 | "Hey Jarvis, stop" in the pause was racy: the server hears the user after the 1.5 s pause ends, so the next passage was queued behind the user's turn and read after "stop" | The reader pauses on the in-process `wake_word` event (it arrives before the pause ends), and only sends a passage when no in-band response is active or queued (`_response_done_event`, `_pending_responses`). An interruption in the pause after a passage resumes at the *next* passage. |
| 2 | `mute()` flipped the mic after releasing its lock; a mic frame in between (every frame calls `_sync_with_mic` with `--wake`) looked like a UI un-mute and dropped the hard flag and timer | The mic flag is set, and read by `_sync_with_mic`, under the mute lock. The new test forces a frame into that moment; it fails on the old code. |
| 3 | `stop_now` cancelled whichever response was active, usually the model's own reply that called the tool (cutting its "Okay" and its tool bookkeeping) | Only cancels when a passage is active on the server (sent, `response_created` seen, no transcript/interruption yet) |
| 4 | "Book one was better than…", "Part one of the plan…" became chapters; "Twenty-One" parsed as 20 | A heading's title must start with a capital or punctuation; compound number words (twenty-one … ninety-nine) parse |
| 5 | Follow-up window assumes the playback clock runs ahead of real time | Not changed (needs a live check) |
| 6 | `save_mark` file I/O under the reader lock | Not changed (milliseconds, no deadlock) |
| 7 | The reader timeout test had ~0.7 s of slack | Polls up to 10 s instead; dead `publish` removed |
| 8 | `list_key("my list")` → "list"; "tomatoes" didn't remove "tomato" | "my list" → the default list (notes); `-es` plurals match |
| 17 | Docstrings still said ~150-word passages and "just before the current one runs out" | Updated (reader, bridge) |

**Tests:** 35 offline tests pass (LLM and realtime groups not re-run: no prompt changes, and the live app holds the speech server).

### Head jerks on every app restart (2026-09-19, 10:46–10:55)

**Symptom:** "the motors shake a lot" whenever the app was restarted. It looked like a second connection, but there was only one: the daemon had one `/ws/sdk` client and drops clients on disconnect.

**Measured** (head pose from `/api/state/full` at 60 Hz through a restart): 0.8 s after the app started, the head was frozen for 0.75 s, then nodded down 24° and dropped 27 mm within 130 ms, and sprang back within 250 ms. That was followed by the normal talking wobble of the greeting. The daemon logged 2–4 `IK error: Collision detected or head pose not achievable!` at every app start (16 in total).

**Cause:** the daemon's *reported* head pose was wrong: z = -154 mm at neutral, while the joints (all ±0.627 rad) are exactly neutral and the SDK's own `AnalyticalKinematics().fk()` gives -1 mm for them. The daemon's forward kinematics is iterative, seeded from its previous estimate, and had locked onto a wrong solution. The app starts its idle `BreathingMove` with a 1 s glide from `get_current_head_pose()` (that wrong pose) to neutral. The first 75 % of the glide was unreachable (IK errors, head held still), then the head snapped to the first reachable pose.

**Fix:** restart the daemon (the app only reconnects). Afterwards: z = -1.8 mm at rest, 0 IK errors, no jerk on app start (max 4.5 mm per half second, all of it the greeting's normal wobble). Ruled out on the way: two wobblers (the app-side one moves the head for audio the app plays, the daemon's for the daemon's own audio, so there's no doubling) and the app's startup audio-board writes (rewriting all 7 settings moved the head at most 0.8 mm / 0.6°).

**Check:** `curl -s 127.0.0.1:8000/api/state/present_head_pose`. At rest, `z` should be within about ±10 mm. If it's around -150 mm, restart the daemon.

**Also:** stop the app with Ctrl-C / SIGINT, not a plain `pkill` (SIGTERM). SIGTERM skips the app's `finally:` shutdown (motion loop stop, wobbler off, clean disconnect).
