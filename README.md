# Local voice assistant for Reachy Mini Lite

A fully local voice assistant for the USB-tethered **Reachy Mini Lite** robot. It runs Pollen Robotics' [conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app) and the `reachy-mini` daemon **unmodified**. The app's own `local` mode points them at a self-hosted [speech-to-speech](https://github.com/huggingface/speech-to-speech) server instead of the hosted Hugging Face backend:

- **Silero VAD** for voice activity detection, and **Parakeet TDT 0.6B v3** for speech-to-text
- **Ollama** `reachy-gemma4` (gemma4:26b, 32k context, thinking off) as the LLM, with tools and vision
- **Qwen3-TTS 1.7B** (GGML, voice "Aiden") for text-to-speech

Every service listens on `127.0.0.1` only. Nothing in this repo patches upstream code: our extra behaviour lives in `local_backend/` as profiles, external tools and small wrappers (`run_app.py`, `run_daemon.py`, `reachy_bridge.py`).

## Architecture

```text
   REACHY MINI LITE (the robot)                YOUR PC  (everything talks over 127.0.0.1)
  +-------------------------+       USB       +--------------------------------------------------------+
  |                         |                 |                                                        |
  |  Mic  ------------------+---------------->|  CONVERSATION APP  (Pollen's, unmodified)              |
  |                         |                 |  web UI on :7860                                       |
  |                         |                 |                                                        |
  |                         |                 |   1. Wake gate:  "hey jarvis" or "hey marvin"?         |
  |  Speaker <--------------+-----------------|   2. Router:     send to that assistant                |
  |                         |                 |                  (own memory, own voice, own persona)  |
  |                         |                 |              |                        ^             ^  |
  |                         |                 |              | your voice             | reply audio |  |
  |                         |                 |              v                        |             |  |
  |                         |                 |  SPEECH SERVER  :8765                 |             |  |
  |                         |                 |   3. Silero       -> are you talking? |             |  |
  |                         |                 |   4. Parakeet     -> voice to text    |             |  |
  |                         |                 |              |                        |             |  |
  |                         |                 |              v                        |             |  |
  |                         |                 |  OLLAMA  :11434                       |             |  |
  |                         |                 |   5. gemma4       -> thinks, answers, |             |  |
  |                         |                 |                      may call a tool  |             |  |
  |                         |                 |              |                        |             |  |
  |                         |                 |              v                        |             |  |
  |                         |                 |   6. Qwen3-TTS    -> text to voice ---+             |  |
  |                         |                 |                                                     |  |
  |  Head + antenna motors <+-----------------|  DAEMON  :8000  (moves the head, reads the camera)  |  |
  |  Camera ----------------+---------------->|    camera frames to the app, via a local socket ----+  |
  +-------------------------+                 |  TOOLS (local): reminders, timers, sounds, lists,      |
                                              |    calculator, unit convert, storyteller, mute ...     |
                                              |                                                        |
                                              |  TOOLS (only with --web): weather, radio, news, search |
                                              |    currency, book download                             |
                                              +---------------------------|----------------------------+
                                                                          |
                                                                          v
                                                                      INTERNET
                                                                (off by default)
```

One turn, in order:

1. You say "hey jarvis, what time is it?"
2. The wake gate hears "hey jarvis" and routes to Jarvis.
3. Silero detects when you start and stop talking.
4. Parakeet turns your voice into text.
5. gemma4 reads it, calls `get_time`, and writes a reply.
6. Qwen3-TTS speaks it in Jarvis's voice through the robot's speaker.

Meanwhile the daemon moves the head. Nothing leaves the PC unless started with `--web`.

| Port | Service |
| --- | --- |
| 8000 | `reachy-mini-daemon` API |
| 8443 | Daemon WebRTC signalling (camera + mic stream) |
| 8765 | speech-to-speech Realtime server |
| 7860 | Conversation app web UI (`--ui`) |
| 11434 | Ollama |
| 8888 | SearXNG (only with `--web`) |

## Repo layout

```text
local-assistant/
|
|-- README.md                    <- start here
|
|-- Docs (the detailed write-ups)
|   |-- REACHY_MINI_SETUP.md       connecting the robot over USB
|   |-- CONVERSATION_APP.md        running Pollen's app (cloud version)
|   |-- LOCAL_CONVERSATION.md      making it fully local + the privacy audit
|   |-- FEATURE_PLAN.md            plan for the tools
|   `-- MULTI_ASSISTANT_PLAN.md    plan for Jarvis / Marvin
|
|-- Launchers (what you actually run)
|   |-- start_daemon.sh            terminal 1: the robot daemon
|   |-- start_conversation.sh      terminal 3: the app  (--local --web --wake --assistants)
|   `-- hello.py, say.py           tiny first tests: move / speak
|
`-- local_backend/               <- ALL the code this project adds
    |
    |-- start_local_backend.sh     terminal 2: Ollama + speech server
    |-- start_searxng.sh           optional local search engine
    |-- run_app.py, run_daemon.py  wrappers that lock things to 127.0.0.1
    |-- cache_models.sh            download models once, then run offline
    |
    |-- reachy_*.py                the features' engines
    |     wake, assistants, scheduler, sounds, radio,
    |     reader, lists, listening (mute), bridge
    |
    |-- tools/                     19 tools the LLM can call
    |     get_time, set/list/cancel_reminder, play/stop_sound,
    |     play/stop_radio*, calculate, convert_units, convert_currency*,
    |     lists, listening, read_book, switch_persona,
    |     forget_conversation, get_weather*, web_search*, tech_news*
    |                                                  (* = --web only)
    |
    |-- profiles/                  30 folders = 15 personalities x 2
    |     local_jarvis, local_jarvis_web, local_marvin, ...
    |
    |-- assistants.json            Jarvis <-> "hey jarvis", voice Ryan
    |                              Marvin <-> "hey marvin", voice Eric
    |
    |-- net_watch.py, netaudit/    proves nothing phones home
    |-- gpu_monitor.py, stt_benchmark.py
    `-- tests/                     pytest suite


NOT in the repo (git-ignored, downloaded separately):
    reachy_mini_conversation_app/   Pollen's app
    third_party/speech-to-speech/   the speech server
    third_party/gst-plugins-rs/     video plugin
    .venv/  models/  voices/  logs/  sounds/  books/  state/
```

Pollen's app, daemon and speech server live next to this repo, unmodified. This repo adds `local_backend/` and plugs it in from the outside: environment variables point the app at `profiles/` and `tools/`, and `run_app.py` / `run_daemon.py` wrap the upstream start-up to bind `127.0.0.1`.

## How the third-party code fits

`░░ THIRD PARTY ░░` = downloaded code, not modified; `██ THIS REPO ██` = code in this repo.

```text
 ╔══════════════════════════════════════════════════════════════════════════════════════╗
 ║ PROCESS 3: CONVERSATION APP   (terminal 3: ./start_conversation.sh)   UI :7860       ║
 ║                                                                                      ║
 ║  ░░ THIRD PARTY ░░  reachy_mini_conversation_app/   (Pollen, git clone)              ║
 ║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
 ║  │ main.py  console.py  huggingface_realtime.py  memory.py  moves.py              │  ║
 ║  │ tools/core_tools.py  (the Tool base class)     profiles/  (Pollen's personas)  │  ║
 ║  └───▲───────────────▲───────────────────▲──────────────────────▲─────────────────┘  ║
 ║      │ wraps         │ subclass Tool     │ loaded via env var   │ hooks into         ║
 ║      │ main()        │                   │ EXTERNAL_PROFILES_   │ console + realtime ║
 ║      │               │                   │ DIRECTORY            │ client             ║
 ║  ┌───┴──────────┐ ┌──┴──────────────┐ ┌──┴─────────────────┐ ┌──┴─────────────────┐  ║
 ║  │██ run_app.py │ │██ tools/  (19)  │ │██ profiles/ (30)   │ │██ reachy_wake.py   │  ║
 ║  │ binds UI to  │ │ loaded via env  │ │ generated from     │ │██ reachy_assistants│  ║
 ║  │ 127.0.0.1    │ │ EXTERNAL_TOOLS_ │ │ Pollen's by        │ │██ reachy_bridge.py │  ║
 ║  │              │ │ DIRECTORY       │ │ make_personas.py   │ │ wake word, Jarvis/ │  ║
 ║  └──────────────┘ └─────────────────┘ └────────────────────┘ │ Marvin routing     │  ║
 ║                                                              └────────────────────┘  ║
 ╚═════════════╤════════════════════════════════════════════════════════════╤═══════════╝
               │ realtime websocket  ws://127.0.0.1:8765                    │ robot SDK
               ▼                                                            │ :8000
 ╔════════════════════════════════════════════════════╗                     │
 ║ PROCESS 2: SPEECH SERVER                           ║                     │
 ║ (terminal 2: local_backend/start_local_backend.sh) ║                     │
 ║                                                    ║                     │
 ║  ░░ THIRD PARTY ░░  third_party/speech-to-speech/  ║                     │
 ║  ┌──────────────────────────────────────────────┐  ║                     │
 ║  │ Silero VAD → Parakeet STT → Qwen3-TTS        │  ║                     │
 ║  └─────────▲─────────────────────┬──────────────┘  ║                     │
 ║            │ run with our flags  │ HTTP :11434     ║                     │
 ║  ┌─────────┴────────────────┐    │                 ║                     │
 ║  │██ start_local_backend.sh │    │                 ║                     │
 ║  │ offline, telemetry off,  │    │                 ║                     │
 ║  │ picks GPU, VAD tuning    │    │                 ║                     │
 ║  └──────────────────────────┘    ▼                 ║                     │
 ║  ░░ THIRD PARTY ░░  Ollama (system install)        ║                     │
 ║  ┌──────────────────────────────────────────────┐  ║                     │
 ║  │ gemma4:26b   ◀── ██ Modelfile.reachy-gemma4  │  ║                     │
 ║  │                   (32k context)              │  ║                     │
 ║  └──────────────────────────────────────────────┘  ║                     │
 ╚════════════════════════════════════════════════════╝                     │
                                                                            ▼
 ╔══════════════════════════════════════════════════════════════════════════════════════╗
 ║ PROCESS 1: ROBOT DAEMON   (terminal 1: ./start_daemon.sh)   :8000, WebRTC :8443      ║
 ║                                                                                      ║
 ║  ░░ THIRD PARTY ░░  reachy-mini 1.10.0  (Pollen, pip install into .venv/)            ║
 ║  ┌────────────────────────────────────────────────────────────────────────────────┐  ║
 ║  │ daemon: motors, camera, mic, speaker          media server ──uses──┐           │  ║
 ║  └───▲────────────────────────────────────────────────────────────────┼───────────┘  ║
 ║      │ wraps daemon, binds signalling to 127.0.0.1                    ▼              ║
 ║  ┌───┴─────────────────┐              ░░ THIRD PARTY ░░  webrtcsink plugin           ║
 ║  │██ run_daemon.py     │              built once from third_party/gst-plugins-rs/,   ║
 ║  └─────────────────────┘              found via GST_PLUGIN_PATH=                     ║
 ║                                       ~/.local/gst-plugins-rs                        ║
 ╚═══════════════════════════════════════════╤══════════════════════════════════════════╝
                                             │ USB
                                             ▼
                                   Reachy Mini Lite robot
```

How this repo plugs in (no third-party code is edited):

- **Conversation app**: `run_app.py` wraps Pollen's `main()` after binding the UI to `127.0.0.1`; two env vars (`REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`, `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`) make the app load this repo's `tools/` and `profiles/`; every tool subclasses Pollen's `Tool`, and the 30 profiles are generated from Pollen's personas by `make_personas.py`; the wake word and Jarvis/Marvin routing hook into the app's console and realtime client.
- **speech-to-speech**: never imported; run as a separate server with this repo's flags (offline, telemetry off, GPU choice, VAD tuning); the app reaches it only over the websocket on `:8765`.
- **gst-plugins-rs**: never run; compiled once to produce the `webrtcsink` plugin, which the daemon loads via `GST_PLUGIN_PATH` set by `start_daemon.sh`.
- **reachy-mini daemon**: a pip package, not a clone; `run_daemon.py` wraps it and binds signalling to `127.0.0.1`.
- **Ollama**: system install; this repo adds only `Modelfile.reachy-gemma4` (32k context); `start_local_backend.sh` turns thinking off per request.

### Swapping the LLM server

The speech server reaches the LLM only through an OpenAI-compatible `/v1` Chat Completions API (`--llm_backend chat-completions --responses_api_base_url ...`). So Ollama can be replaced by another OpenAI-compatible server, such as vLLM, llama.cpp's `llama-server` or LM Studio. The app, tools and profiles don't change.

`start_local_backend.sh` is Ollama-specific today:
- `OLLAMA=http://127.0.0.1:11434` is hardcoded (not an env var).
- It runs `ollama create` from `Modelfile.<name>` if the model is missing.
- It loads the model and keeps it warm with `keep_alive` pings to Ollama's `/api/generate`.

To swap: start the other server yourself, point `--responses_api_base_url` at its `/v1`, set `REACHY_LLM` (passed as `--model_name`) to the model name that server serves, and skip the `ollama create` and keep-alive steps.

Notes:
- The model must support tool calling and, for camera questions, image input.
- Thinking is turned off with `--responses_api_reasoning_effort none`. The `--responses_api_disable_thinking` default sends `chat_template_kwargs.enable_thinking=false`, which only works on vLLM.
- vLLM's default port 8000 clashes with the robot daemon, so run it on another port (e.g. 8001).

## Requirements

- **Robot:** Reachy Mini Lite on USB (motor controller on `/dev/ttyACM0`, plus the Reachy Mini camera and audio devices).
- **OS:** Linux (developed on Ubuntu 24.04, glibc 2.39).
- **GPU:** NVIDIA with CUDA. Developed on 2× RTX 3090 (24 GB each): the LLM uses about 20 GB on one GPU and speech about 7 GB on the other. The launcher puts speech on the GPU with the most free memory.
- **Ollama** with `gemma4:26b` pulled. The launcher creates `reachy-gemma4` from [`local_backend/Modelfile.reachy-gemma4`](local_backend/Modelfile.reachy-gemma4).
- **Python 3.12 and [uv](https://github.com/astral-sh/uv).** Three venvs: `.venv` at the repo root (the `reachy-mini` SDK and daemon), one in the conversation app and one in speech-to-speech.
- **GStreamer** development packages and the Rust-built WebRTC plugin (`webrtcsink`), see below.
- **Docker**, only for the optional SearXNG web search.

## Setup

Full details and the reasoning behind each step are in [REACHY_MINI_SETUP.md](REACHY_MINI_SETUP.md), [CONVERSATION_APP.md](CONVERSATION_APP.md) and [LOCAL_CONVERSATION.md](LOCAL_CONVERSATION.md).

### 1. USB permissions and groups (needs sudo)

Install the udev rule from the [official install guide](https://huggingface.co/docs/reachy_mini/SDK/installation) (see REACHY_MINI_SETUP.md; `MODE="0660"` is safer than the guide's `0666`) and join the device groups:

```bash
sudo usermod -aG dialout,audio,video $USER    # then log out and back in
```

Until you log in again, the launchers re-run themselves under `sg`.

### 2. GStreamer packages and the SDK

```bash
sudo apt-get install libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-base1.0-dev \
  libgstreamer1.0-dev libglib2.0-dev libssl-dev libgirepository1.0-dev libcairo2-dev \
  libportaudio2 libnice10 gstreamer1.0-alsa gstreamer1.0-plugins-bad gstreamer1.0-nice python3-gi-cairo
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install reachy-mini
```

### 3. Upstream clones (git-ignored, fetched separately)

These are separate git repos and are not part of this one.

**WebRTC plugin**, `third_party/gst-plugins-rs` at tag `0.14.5` (needs Rust ≥ 1.83; installed under your home, no sudo):

```bash
git clone --branch 0.14.5 https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git third_party/gst-plugins-rs
cd third_party/gst-plugins-rs
rustup toolchain install 1.90.0 --profile minimal
rustup run 1.90.0 cargo install cargo-c --version 0.10.19 --locked
rustup run 1.90.0 cargo cinstall -p gst-plugin-webrtc --prefix=$HOME/.local/gst-plugins-rs --release
```

**Conversation app**, `reachy_mini_conversation_app/` (developed against app 1.0.1, commit `f58523b`):

```bash
git clone https://github.com/pollen-robotics/reachy_mini_conversation_app
cd reachy_mini_conversation_app
uv venv --python 3.12 .venv && uv sync --frozen
uv pip install --python .venv/bin/python "reachy-mini==1.10.0"   # match the daemon's version
uv pip install --python .venv/bin/python openwakeword sherpa-onnx sentencepiece   # wake words (not in uv.lock)
```

**Speech server**, `third_party/speech-to-speech` (developed against commit `16d7f98`):

```bash
git clone https://github.com/huggingface/speech-to-speech third_party/speech-to-speech
cd third_party/speech-to-speech
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
```

### 4. Audio route

The daemon and SDK use two shared ALSA devices on the robot's card, `reachymini_audio_sink` (dmix) and `reachymini_audio_src` (dsnoop), defined in `~/.asoundrc`. See CONVERSATION_APP.md, step 2. Don't use the SDK's `write_asoundrc_to_home()`: it also makes the robot the default speaker for every app on your account.

### 5. Cache models (once, while online)

```bash
local_backend/cache_models.sh
```

This caches the daemon's YuNet face model, the emotions and dances datasets, and the sherpa-onnx keyword spotter, so everything can run with `HF_HUB_OFFLINE=1`. The speech models (Parakeet, Qwen3-TTS, Silero, Smart Turn) are cached by the speech server's first run. Also create the NLTK symlink described in LOCAL_CONVERSATION.md ("Leak 1"), which stops speech-to-speech from downloading an index on every start.

## Run

Three terminals:

```bash
./start_daemon.sh                          # 1: daemon, offline, signalling on localhost
local_backend/start_local_backend.sh       # 2: loads and keeps the Ollama model, starts the speech server
./start_conversation.sh --local --ui       # 3: the app; UI at http://127.0.0.1:7860
```

Options for `start_conversation.sh`:

| Flag | Effect |
| --- | --- |
| `--local` | Fully local backend, profile `local_reachy` |
| `--web` | Implies `--local`; profile `local_reachy_web` adds the online tools. Start SearXNG first: `local_backend/start_searxng.sh` (Docker, 127.0.0.1:8888) |
| `--wake` | Only listen after the wake word ("Hey Marvin" by default; `REACHY_WAKE_WORD` picks another) |
| `--assistants` | Several assistants picked by wake word ([`assistants.json`](local_backend/assistants.json)); implies `--wake` |
| `--ui` | Serve the web UI (needed for reminders, which are delivered through it) |
| `--no-camera`, `--debug` | Passed through to the app |

Without `--local`/`--web`, the app uses Pollen's hosted backend.

Backend settings (environment variables for `start_local_backend.sh`): `REACHY_LLM`, `REACHY_STT=parakeet|whisper`, `REACHY_SPEECH_GPU`, `REACHY_VAD_THRESH`, `REACHY_VAD_MIN_SPEECH_MS`, `REACHY_STREAM_SENTENCES`, `REACHY_NET_AUDIT=1`.

## Features

Tools added in `local_backend/tools/`, on top of the app's own (dance, emotions, move_head, camera, head tracking, memory, volume, …):

| Feature | Tools | Online? |
| --- | --- | --- |
| Time | `get_time` | No |
| Reminders and timers, with sounds; kept across restarts | `set_reminder`, `list_reminders`, `cancel_reminder` | No |
| Sound effects (alarm, timer, chime, bell, …, or your own files) | `play_sound`, `stop_sound` | No |
| Calculator (safe `ast` evaluator, no `eval`) | `calculate` | No |
| Unit conversion | `convert_units` | No |
| Lists (shopping, to-do, notes) | `lists` | No |
| Privacy mute: no mic audio leaves the process; auto-resume timer | `listening` | No |
| Storyteller: reads books word for word in the robot's voice, with bookmarks | `read_book` | No (except `download`) |
| Personas: 12 upstream personalities, each offline and web, with their own voice | `switch_persona` | No |
| Wake words: openWakeWord ("Hey Marvin", "Hey Jarvis") or a sherpa-onnx phrase ("Hey Reachy") | `--wake` | No |
| Multiple assistants: Jarvis (calm butler, voice Ryan) and Marvin (gloomy robot, voice Eric), each with its own history, memory, lists, reminders and bookmarks | `--assistants`, `forget_conversation` | No |
| Weather (Open-Meteo) | `get_weather` | **Yes** |
| Web search via local SearXNG, with pagination | `web_search` | **Yes** |
| Tech news (Hacker News, Ars Technica, The Verge RSS) | `tech_news` | **Yes** |
| Internet radio (radio-browser.info) | `play_radio`, `stop_radio` | **Yes** |
| Currency conversion (ECB rates via frankfurter) | `convert_currency` | **Yes** |
| Book download from Project Gutenberg | `read_book` `download` | **Yes** |

The online tools exist only in the `_web` profiles, selected with `--web`. Even then, speech, the LLM, the camera and the conversation stay on this machine.

## Privacy: how "fully local" was verified

- **All listeners on 127.0.0.1.** Upstream binds the app UI and the daemon's WebRTC signalling to `0.0.0.0`; `run_app.py` and `run_daemon.py` rebind them to localhost (`REACHY_UI_HOST` / `REACHY_SIGNALLING_HOST` to override).
- **[`net_watch.py`](local_backend/net_watch.py)** logs every TCP/UDP connection of the daemon, speech server and app, and flags non-loopback peers.
- **Python audit hook** ([`netaudit/sitecustomize.py`](local_backend/netaudit/sitecustomize.py), `REACHY_NET_AUDIT=1`) logs every non-local DNS lookup and `connect` from Python, with a stack trace.
- **`strace -f -k -e trace=connect`** catches native connections.

Leaks found and fixed:

| Leak | Found by | Fix |
| --- | --- | --- |
| onnxruntime 1.30 telemetry to `mobile.events.data.microsoft.com` (native, in the speech server) | `net_watch` + `strace -k` | `ORT_DISABLE_TELEMETRY=1` in the launchers |
| NLTK index download from GitHub on every speech-server start | Audit hook | Data-only symlink under `~/nltk_data` |
| Daemon: YuNet face model download, and emotions/dances dataset preload and 24 h update | Code audit | `cache_models.sh`, `HF_HUB_OFFLINE=1`, `--dataset-update-interval 0` |
| App UI and WebRTC signalling reachable from the LAN | Code audit | Bound to 127.0.0.1 |

After the fixes, the whole backend ran under `strace` through startup, idle and the full test suite: every `connect()` went to `127.0.0.1` or local sockets, with no external connections.

## Tests

```bash
cd local_backend && ../third_party/speech-to-speech/.venv/bin/python -m pytest -v
```

The suite covers configuration, services, LLM behaviour and tool choice, the Realtime path, locality (no external connections, nothing exposed to the LAN), GPU placement, and each feature. Markers: `realtime` needs the speech server's single session slot (stop the app first); `online` needs the internet and SearXNG. Tests that need a service that isn't running skip themselves.

## Safety and operating notes

- **Keep the daemon off when the robot is idle.** With the app stopped, the daemon keeps holding the last antenna targets and the antenna servos report overload errors continuously. Stop the daemon too when you're done.
- Starting the daemon powers the motors: clear the space around the head first.
- If the robot loses power, the daemon doesn't reconnect by itself: restart it.
- If the head jerks when the app starts, check `curl -s 127.0.0.1:8000/api/state/present_head_pose`. A `z` outside about -70…+50 mm means the daemon's head-pose estimate is stuck on a wrong solution: restart the daemon.
- Stop the app with Ctrl+C (SIGINT), not SIGTERM, so it shuts down cleanly.

## Documentation

- [REACHY_MINI_SETUP.md](REACHY_MINI_SETUP.md): connecting the Reachy Mini Lite (USB, udev, SDK, first script, speaking)
- [CONVERSATION_APP.md](CONVERSATION_APP.md): the conversation app on Linux with the hosted backend (WebRTC plugin, audio route)
- [LOCAL_CONVERSATION.md](LOCAL_CONVERSATION.md): the fully local stack, model choice, locality audit, and a log of every feature
- [FEATURE_PLAN.md](FEATURE_PLAN.md): calculator, lists, privacy mute, wake word, personas, storyteller
- [MULTI_ASSISTANT_PLAN.md](MULTI_ASSISTANT_PLAN.md): several assistants picked by wake word

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

Third-party code is fetched separately and keeps its own licenses: Pollen Robotics' conversation app and daemon, the speech-to-speech server, the Ollama models and the other models (speech recognition, text-to-speech, wake word).

### Third-party code

A few files in this repository contain modified code or text from upstream projects. Each file says so:

- `local_backend/profiles/*/profile.md` (generated by `local_backend/make_personas.py`): persona text, and prompt text adapted from the default profile, from Pollen Robotics' [reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app), Apache License 2.0.
- `hello.py`: adapted from Pollen Robotics' [Reachy Mini quickstart](https://huggingface.co/docs/reachy_mini/SDK/quickstart) (the `reachy_mini` SDK, Apache License 2.0).
