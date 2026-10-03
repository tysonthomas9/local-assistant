# Reachy Mini Lite with Muse as the brain

A voice assistant for the **Reachy Mini Lite** robot. The brain is [Muse](https://gadgets.muse.ai). The robot runs Pollen Robotics' [conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app) (pinned at `f58523b`, app 1.0.1) and the `reachy-mini` daemon (1.10.0), both **unmodified**, on a Mac. A small backend, `MuseHandler`, takes the place of the app's Hugging Face realtime backend:

- **Silero VAD** cuts the robot's mic stream into one utterance at a time.
- **Speech-to-text on the Mac** (MLX): Qwen3-ASR 0.6B by default, with parakeet and whisper as fallbacks.
- **Muse** writes the reply. Only the text goes to Muse, through the Muse gadget (Muse's open-source [Linux Device SDK](https://github.com/facebookincubator/muse-gadget-sdk), in a Podman container on the Mac).
- **Text-to-speech on the Mac**: Qwen3-TTS 1.7B CustomVoice (voice "Aiden", playful style), with Kokoro-82M and macOS `say` as fallbacks.
- **Robot tools**: Muse can make the robot show an emotion, dance, look around and track faces, using Pollen's own moves.

Everything Muse-specific lives in [`muse_gadget/`](muse_gadget/). The details are in [muse_gadget/README.md](muse_gadget/README.md).

## Branches / brains

One robot stack (Reachy Mini Lite, Pollen's conversation app, the daemon, local speech) with different "brain" agents, each on its own branch.

- **This branch (`muse-gadget-poc`)**: the brain is Muse, through the gadget in `muse_gadget/`. This README describes it.
- **`main`**: the fully local brain (Ollama gemma4, a local speech-to-speech server, wake words, Jarvis/Marvin, the local tools in `local_backend/`). See `main`'s README.

This branch is long-lived and never merged into `main`. Fixes both brains need go to `main` first and come here when `main` is merged in (see "Branch strategy" in [muse_gadget/README.md](muse_gadget/README.md#branch-strategy)).

## Which machine does what

- **The PC (Linux)** pairs the gadget with Muse **once**, over Bluetooth LE, and drives every run over SSH. It runs nothing of the conversation itself.
- **The Mac** (Apple silicon, SSH alias `reachy-mac`) runs everything at runtime: the robot on USB, the daemon, the conversation app with MuseHandler, speech-to-text, text-to-speech and the Muse gadget container. Every port it opens is on `127.0.0.1`.

## Architecture

```text
 PC (Linux)                     reachy-mac  (runtime; every port on 127.0.0.1)
 ----------                     -------------------------------------------------------------------
 pair_on_pc.sh (once)           REACHY MINI LITE (USB)          DAEMON :8000 (run_daemon.py)
   Docker + host BlueZ            mic, speaker, motors  <---->    motors, camera, audio
   <-- BLE --> Muse phone app                                          ^
   state -> Mac, then deleted                                          | Pollen's moves
                                CONVERSATION APP (Pollen's, unmodified, in Reachy Edge.app)
 run_poc.sh (every run, SSH)    backend = MuseHandler
   lock, install, start,          robot mic
   clean up                         -> 1. Silero VAD   one utterance
                                    -> 2. STT          Qwen3-ASR 0.6B (parakeet, whisper)
                                    -> 3. POST /turn {"text"} ---------------+
                                                                             v
                                                          MUSE GADGET (Podman container)
                                                            /turn bridge :48080
                                                            gadget link <== encrypted ==> MUSE (cloud)
                                                                             |    4. reply text
                                                                             |    5. client.invoke
                                    <- reply text ---------------------------+       reachy.*
                                    -> 6. TTS          Qwen3-TTS 1.7B, Aiden |
                                                       (Kokoro, say)         |
                                    -> robot speaker                         |
                                  ROBOT TOOLS :48081  <---- POST /tool ------+
                                    per-run secret, allowlist     (host.containers.internal)
                                    -> Pollen's tools -> daemon
```

One turn, in order:

1. You say "do a little dance". Silero VAD finds where you start and stop talking.
2. Qwen3-ASR turns your voice into text, on the Mac.
3. MuseHandler sends only that text to the bridge (`POST /turn` on `127.0.0.1:48080`).
4. The gadget passes it to Muse over its encrypted link and reads Muse's reply.
5. If Muse wants to move the robot, it sends a `client.invoke` for a `reachy.*` command. The gadget calls MuseHandler's robot-tools endpoint, which starts the dance at once with Pollen's own `dance` tool and tells Muse it's in progress.
6. Qwen3-TTS speaks the reply on the robot's speaker while the robot dances, and Pollen's wobbler moves the head while it plays. The reply streams in (`POST /turn?stream=1`): the first sentence is spoken as soon as Muse has written it, without waiting for the rest.

What you say ends after 0.5 s of quiet (`--end-silence`). Each turn logs the time from the end of your speech to the robot's first audio.

It's half-duplex: the mic is ignored while a turn is being transcribed, sent or spoken. There's no wake word, so anything said near the robot becomes a turn.

| Port (Mac, 127.0.0.1) | Service |
| --- | --- |
| 8000 | `reachy-mini` daemon API |
| 8443 | Daemon WebRTC signalling |
| 48080 | Muse gadget `/turn` bridge (published by Podman on loopback only) |
| 48081 | MuseHandler's robot-tools endpoint (only while a run is up) |

## Repo layout

```text
README.md                   <- this file
muse_gadget/                <- everything this branch adds
|-- README.md                 the details: bridge, pairing, robot tools, logging
|-- run_poc.sh                PC: one run (lock, install, bridge, daemon, app, clean-up)
|-- pair_on_pc.sh             PC: pair once over BLE, copy the state to the Mac
|-- mac_gadget.sh             PC: start|stop|status|logs the gadget container on the Mac
|-- Containerfile             gadget image: Muse SDK (pinned commit) + gadget/
|-- gadget/                   runs in the container
|     link.py, bridge.py, chat.py     Muse link and the /turn bridge
|     restrict.py                     allowed commands only (no shell, no files)
|     robot.py, client_invoke.py      reachy.* commands and Muse's client.invoke
|-- mac/                      copied to the Mac by run_poc.sh
|     install.sh                      pinned app, venvs and models
|     run_app.py, run_daemon.py       launch the app / daemon, everything on loopback
|     muse_handler.py                 the app backend: VAD -> STT -> /turn -> TTS
|     muse_vad.py, muse_stt.py, muse_tts.py, *_worker.py
|     robot_tools.py                  the robot-tools endpoint
|     fake_bridge.py                  echo bridge for testing without Muse
`-- tests/                    gadget tests (mac/tests/ for MuseHandler)

local_backend/, start_*.sh  main's fully local brain (not used by the Muse brain)
*.md                        write-ups, mostly about main's local stack (see Documentation)
```

## Requirements

- **PC:** Linux with BlueZ and Docker (for pairing), SSH access to the Mac as `reachy-mac`.
- **Mac:** Apple silicon, Podman (`podman-machine-default`), and Reachy Edge.app with `~/assistant-edge` set up by `scripts/edge_host_bootstrap.sh` from the `redesign/architecture` branch (it provides uv and Python 3.12). Reachy Edge needs the microphone grant once.
- **Robot:** Reachy Mini Lite on the Mac's USB.
- **Muse:** an SDK token from gadgets.muse.ai > Account > SDK tokens, and the Muse phone app, signed in, with Developer mode on.

## Quick start

All commands run on the PC, from the repo root.

**1. Pair once.** The phone must be near the PC. See [Pairing](muse_gadget/README.md#pairing-once-on-the-pc) for the BlueZ tweaks you may need.

```bash
muse_gadget/pair_on_pc.sh --check                                 # optional: image, BlueZ and SSH only
install -m 600 /dev/null ~/.config/muse-sdk-token && nano ~/.config/muse-sdk-token
muse_gadget/pair_on_pc.sh --token-file ~/.config/muse-sdk-token   # or no flag: hidden prompt
```

In the Muse app, add a device and pick the name the terminal shows (like `MuseGadgetA1B2C3`). Once paired, the state is copied to `reachy-mac:~/assistant-edge/muse-state` and deleted from the PC.

**2. Cache the face-tracking model once** (on the Mac, with network access; the daemon runs offline). Without it, `reachy.head_tracking` crashes the face tracker. The command is in [muse_gadget/README.md](muse_gadget/README.md#running-the-robot-with-muse).

**3. Run.**

```bash
muse_gadget/run_poc.sh                  # talk to Muse through the robot; Ctrl-C to stop
muse_gadget/run_poc.sh --fake-bridge    # no Muse: the robot answers "You said: ..."
```

The first run installs the app, its venvs and the speech models on the Mac (`mac/install.sh`, into `~/assistant-edge/muse-app`); later runs only check them. Main options:

| Flag | Effect |
| --- | --- |
| `--stt qwen3-asr\|parakeet\|whisper` | Speech-to-text engine (default `qwen3-asr`, which falls back to parakeet if it can't load) |
| `--stt-model ID` | `0.6b` (default) or `1.7b` for Qwen3-ASR, or a model repo id |
| `--tts qwen3\|kokoro\|say` | Reply voice engine (default `qwen3`) |
| `--voice NAME` | Qwen3: `Aiden` (default), `Ryan`; Kokoro: `af_heart` (default), `am_michael`, ...; `say`: a macOS voice |
| `--instruct TEXT` | Qwen3's speaking style (default "playful and cheeky, like a friendly cartoon robot"; `""` for none). Changes how a reply is said, never the words |
| `--volume N` | Robot speaker volume 0-100 (default 100) |
| `--style-hint` | Opt in: put a short "spoken by a desk robot" note before your words. Off by default |
| `--duration S`, `--lock-timeout S` | Stop by itself after S seconds; give up waiting for the robot lock |
| `--end-silence S` | Quiet time that ends what you say, 0.1-5 seconds (default 0.5). Shorter answers sooner but may cut you off mid-pause |
| `--mic-log S`, `--log-transcripts` | Debugging: log mic level and VAD score; show each turn's text on the terminal |
| `-- <app args>` | Passed to the conversation app (for example `--no-camera`, `--debug`) |

`mac_gadget.sh start|stop|status|logs` manages the gadget container by itself; `run_poc.sh` calls it for you.

## Robot tools

Muse can call six commands. Each runs one of Pollen's own app tools, so the robot only ever plays Pollen's moves:

| Command | Pollen tool | What it does |
| --- | --- | --- |
| `reachy.emotion` | `play_emotion` | a recorded emotion (`happy`, `sad`, `surprised`, ...) |
| `reachy.dance` | `dance` | a dance move (random if none is given) |
| `reachy.stop_move` | `stop_dance` | stop the dance or emotion |
| `reachy.look` | `move_head` | left, right, up, down or front |
| `reachy.head_tracking` | `head_tracking` | the face tracker on or off |
| `reachy.status` | `robot_status` | `name`, `software` or `imu` only |

Path: Muse `client.invoke` → gadget (checks the command and its parameters) → `POST /tool` to MuseHandler's endpoint on the Mac's `127.0.0.1:48081` → Pollen's tool. The endpoint accepts only these tools and needs a random secret that `run_poc.sh` makes for each run and deletes at clean-up. It exists only while a run holds the robot; otherwise a command answers "robot is asleep". **Move and talk together:** emotions, dances and looks start at once, and the reply is spoken while the robot moves; Muse is told the move is in progress. There's no camera, volume, sleep, memory or web command. Besides these, the gadget registers only `device.health`: Muse gets no shell and no file access.

## Privacy

- **What goes to Muse:** the text of each turn (what you said, after speech-to-text), and the `reachy.*` command results. Nothing is added to your words: the style note is off unless you pass `--style-hint`.
- **What stays on the Mac:** the audio. VAD, speech-to-text and text-to-speech all run locally. The camera isn't sent anywhere.
- **Nothing on the LAN:** the bridge, the robot-tools endpoint, the daemon API and its WebRTC signalling are on `127.0.0.1`, and the daemon's mDNS announcement is skipped.
- **Logs have no transcripts by default:** MuseHandler and the bridge log lengths, timings and command names, never text. The PC keeps each run's daemon, app and gadget logs in `~/.local/state/muse-poc/`, with any text redacted.
- **Secrets:** the SDK token is never on the command line, in git, in the image or in logs. The pairing state lives only on the Mac (0600). This repo never contains the Mac's address, user name or home path.

See [Token and privacy](muse_gadget/README.md#token-and-privacy) for details, and Muse's terms (gadgets.muse.ai/sdk-terms: personal, non-commercial use, your own account).

## Safety

- **One run at a time.** `run_poc.sh` takes a hw-run lock on the Mac and waits while another run holds it. It refuses to start if a daemon or bridge it didn't start is already running.
- **Clean stop.** On exit, an error or Ctrl-C, it stops the app, puts the robot to sleep (`goto_sleep`), turns the motors off, stops the daemon and the bridge, and releases the lock.
- Starting the daemon powers the motors: clear the space around the head first.
- The gadget never starts the robot: a robot command with no run going answers "robot is asleep".

## Tests

```bash
# gadget (PC, in the image)
docker build -f muse_gadget/Containerfile -t localhost/muse-gadget:latest muse_gadget
docker run --rm --entrypoint sh localhost/muse-gadget:latest -c 'cd /opt/gadget && python -m pytest -q -p no:cacheprovider tests'

# MuseHandler, VAD, STT, TTS and robot tools (PC, with the conversation app's venv)
cd muse_gadget/mac && <app venv>/bin/python -m pytest tests -q
```

The gadget tests use a fake Muse link and a fake Muse VM that speaks the SDK's real encrypted protocol. More in [Build and test](muse_gadget/README.md#build-and-test).

## Documentation

- [muse_gadget/README.md](muse_gadget/README.md): the Muse gadget, the bridge, pairing, running on the Mac, robot tools, privacy and logging
- [REACHY_MINI_SETUP.md](REACHY_MINI_SETUP.md): connecting a Reachy Mini Lite over USB (Linux)
- [CONVERSATION_APP.md](CONVERSATION_APP.md): Pollen's conversation app with the hosted backend (Linux)
- [LOCAL_CONVERSATION.md](LOCAL_CONVERSATION.md), [FEATURE_PLAN.md](FEATURE_PLAN.md), [MULTI_ASSISTANT_PLAN.md](MULTI_ASSISTANT_PLAN.md): `main`'s fully local brain

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

Third-party code is fetched separately and keeps its own licenses: Pollen Robotics' conversation app and daemon, Muse's gadget SDK, and the models. Use of the Muse SDK is also governed by Muse's SDK terms (see [Token and privacy](muse_gadget/README.md#token-and-privacy)).

### Third-party code

A few files in this repository contain modified code or text from upstream projects. Each file says so:

- `muse_gadget/gadget/link.py` (`RobotService._session`): adapted from muse-gadget-sdk `linux/src/musegadget/service.py` (commit 7e88df2). Copyright (c) Meta Platforms, Inc. and affiliates, Apache License 2.0.
- `muse_gadget/tests/test_link.py` (`Pipe`, `FakeVm`): adapted from muse-gadget-sdk `linux/tests/test_link_client.py` (commit 7e88df2). Copyright (c) Meta Platforms, Inc. and affiliates, Apache License 2.0.
- `local_backend/profiles/*/profile.md` (generated by `local_backend/make_personas.py`): persona text, and prompt text adapted from the default profile, from Pollen Robotics' [reachy_mini_conversation_app](https://github.com/pollen-robotics/reachy_mini_conversation_app), Apache License 2.0.
- `hello.py`: adapted from Pollen Robotics' [Reachy Mini quickstart](https://huggingface.co/docs/reachy_mini/SDK/quickstart) (the `reachy_mini` SDK, Apache License 2.0).
