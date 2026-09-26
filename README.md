# Local voice assistant for Reachy Mini Lite

A fully local voice assistant for the USB-tethered **Reachy Mini Lite** robot. It runs Pollen Robotics' [conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app) and the `reachy-mini` daemon **unmodified**. The app's own `local` mode points them at a self-hosted [speech-to-speech](https://github.com/huggingface/speech-to-speech) server instead of the hosted Hugging Face backend:

- **Silero VAD** for voice activity detection, and **Parakeet TDT 0.6B v3** for speech-to-text
- **Ollama** `reachy-gemma4` (gemma4:26b, 32k context, thinking off) as the LLM, with tools and vision
- **Qwen3-TTS 1.7B** (GGML, voice "Aiden") for text-to-speech

Every service listens on `127.0.0.1` only. Nothing in this repo patches upstream code: our extra behaviour lives in `local_backend/` as profiles, external tools and small wrappers (`run_app.py`, `run_daemon.py`, `reachy_bridge.py`).

## Architecture

```
Robot mic ─USB─▶ conversation app ──ws://127.0.0.1:8765/v1/realtime──▶ speech-to-speech (GPU with most free memory)
                 (run_app.py; UI on                                        Silero VAD → Parakeet TDT 0.6B v3 STT
                  127.0.0.1:7860)                                                 │
                                                                                  ▼
                                                                Ollama 127.0.0.1:11434/v1: reachy-gemma4
                                                                (gemma4:26b, 32k ctx, thinking off; kept
                                                                 loaded by the backend launcher)
                                                                                  │
                                                                                  ▼
Robot speaker ◀─USB─ conversation app ◀────── audio + tool calls ───────── Qwen3-TTS 1.7B (voice "Aiden")
        daemon (run_daemon.py, 127.0.0.1:8000, signalling 127.0.0.1:8443) drives motors/camera over USB
```

| Port | Service |
| --- | --- |
| 8000 | `reachy-mini-daemon` API |
| 8443 | Daemon WebRTC signalling (camera + mic stream) |
| 8765 | speech-to-speech Realtime server |
| 7860 | Conversation app web UI (`--ui`) |
| 11434 | Ollama |
| 8888 | SearXNG (only with `--web`) |

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
