# Muse gadget for Reachy Mini

This lets the robot talk to [Muse](https://gadgets.muse.ai). Muse's open-source
Linux Device SDK ([muse-gadget-sdk](https://github.com/facebookincubator/muse-gadget-sdk),
`linux/`) is packaged as a container, together with a small HTTP bridge. The
robot's conversation app sends the text of what you said to the bridge, and
the bridge returns Muse's reply as text.

```
PC (once)                         reachy-mac (every run)
---------                         ----------------------
pair_on_pc.sh                     mac_gadget.sh start
  Docker + host BlueZ               Podman (podman-machine-default)
  musegadget pair  <-- BLE -->        muse-gadget container
  Muse phone app                        gadget link  <== encrypted ==> Muse
        |                               /turn bridge on 127.0.0.1:48080
        +-- state files, copied once -->  ~/assistant-edge/muse-state
            then deleted from the PC   Pollen app (Reachy Edge.app) --POST /turn-->
```

- **Pairing runs once, on the PC.** The SDK pairs over Bluetooth LE through
  Linux BlueZ. Podman's VM on the Mac can't reach the Mac's Bluetooth.
- **Runtime is on the Mac**, in Podman. The bridge is published on the Mac's
  `127.0.0.1` only, so only programs on the Mac can call it.
- **The upstream SDK is never edited.** It's pinned to commit
  `7e88df2bbb3fa92403024b9d161d798f937d6716` (see `Containerfile`) and
  installed with uv. Everything robot-specific lives in `gadget/`.

## What the wrapper changes

- **Commands Muse can call: only `device.health` and the robot's `reachy.*`
  commands** (see "Robot tools" below). The SDK's `system.run`, `file.read`
  and `file.write` are left out of the registration, and the executor refuses
  them anyway (`gadget/restrict.py`). So Muse gets no shell and no file access.
- **Name**: the gadget shows up in Muse as "Reachy Mini" (`MUSE_DISPLAY_NAME`),
  never as the machine's host name. `device.health` reports that name too, and
  the container's host name is `reachy-mini`.
- **One process**: the gadget link and the HTTP bridge run together
  (`python -m gadget run`).
- **Runs as a regular user** (uid 10001) with all capabilities dropped,
  no-new-privileges and a read-only root filesystem.
- **No state in the image.** The identity, pairing and SDK token live in
  `MUSEGADGET_STATE_DIR` (`/state`), a mounted volume.

## The bridge

| Request | Response |
|---|---|
| `POST /turn`, JSON `{"text": "..."}` | 200 `{"reply": "..."}` |
| not paired yet | 503 `{"error": "not_paired"}` |
| paired, but the link to Muse is down | 503 `{"error": "link_down"}` |
| no reply within 60 s | 504 `{"error": "timeout"}` |
| Muse refused the message | 502 `{"error": "muse_error"}` |
| body that isn't JSON (audio included) | 415 `{"error": "unsupported_media_type"}` |
| JSON without `text` | 400 `{"error": "bad_request"}` |
| `GET /health` | 200 `{"paired": bool, "linked": bool}` |

How a turn works (`gadget/chat.py`). `POST /chat/stream` only acknowledges the
message, so the bridge reads the reply the way the SDK's ESP32 firmware does
(`esp32/components/muse/muse_chat_session.cpp`):

1. It opens a `POST /chat/subscribe` NDJSON event stream on the same session.
2. It posts the text and keeps the ids the ack names.
3. It collects the assistant messages that answer those ids
   (`delta.message_start`, `delta.text_append`, `delta.message_done`, or a
   whole `message.assistant`) until all are done and nothing has arrived for
   0.3 s (a busy `agent.status` keeps the turn open for up to 20 s more), then
   closes the stream and strips markdown so the reply reads well aloud.

`/chat/history` isn't used: Muse answers it with 403 for a gadget's device
token.

Turns run one at a time. Each one goes to a fixed side chat
(`MUSE_SESSION_ID`, default the fixed UUID
`06cab6b7-2197-526c-90eb-7aef229fdea5`; Muse refuses a `session_id` that isn't
a UUID with 400 `invalid_params`). Muse gets the user's words only: nothing is
added to them. A short note asking for brief, spoken-style answers can be put in
front only by opting in (`MUSE_STYLE_HINT_ON=1` in the gadget, `run_poc.sh
--style-hint`). The bridge logs only how long messages and replies are, never
what they say.

Speech-to-text runs on the Mac, so the bridge accepts text only.

## Build and test

```bash
# PC, amd64 (Docker)
docker build -f muse_gadget/Containerfile -t localhost/muse-gadget:latest muse_gadget
docker run --rm --entrypoint sh localhost/muse-gadget:latest -c \
  'cd /opt/muse-sdk/linux && python -m pytest -q -p no:cacheprovider tests;
   cd /opt/gadget && python -m pytest -q -p no:cacheprovider tests'
```

On the Mac (arm64), `mac_gadget.sh start` builds the image whenever the
sources have changed (the image is labelled with a hash of the build context).
To run the same tests there:
`ssh reachy-mac 'PATH=/opt/homebrew/bin:$PATH podman run --rm --entrypoint sh localhost/muse-gadget:latest -c "cd /opt/gadget && python -m pytest -q -p no:cacheprovider tests"'`.

The wrapper's tests (`tests/`) use a fake Muse link and a fake Muse VM that
speaks the SDK's real encrypted protocol. They check four things: `/turn`
returns the reply, the registered commands exclude `system.run` and `file.*`,
the display name isn't the host name, and the bridge binds only to loopback.

## Pairing (once, on the PC)

You need:
- an **SDK token** from gadgets.muse.ai > Account > SDK tokens (`mgst_…`);
- the **Muse app** on your phone, signed in, with **Developer mode** on;
- the phone near this PC.

1. Optional: check that everything except the phone step works. This builds
   the image, checks that the container can reach BlueZ, and checks SSH to the
   Mac:
   ```bash
   muse_gadget/pair_on_pc.sh --check
   ```
2. Save the token to a file that only you can read, outside the repo. Or skip
   this step and paste the token at the hidden prompt in step 3.
   ```bash
   install -m 600 /dev/null ~/.config/muse-sdk-token && nano ~/.config/muse-sdk-token
   ```
3. Pair:
   ```bash
   muse_gadget/pair_on_pc.sh --token-file ~/.config/muse-sdk-token
   # or: muse_gadget/pair_on_pc.sh   (asks for the token, hidden)
   ```
   The terminal shows a name like `MuseGadgetA1B2C3`. In the Muse app, add a
   device and pick that name. Setup stays open for 10 minutes (`--timeout`).
4. When the app says it's paired, the script copies the state to
   `reachy-mac:~/assistant-edge/muse-state` (folder 0700, files 0600). It then
   deletes the state from the PC (shred, then delete). If the copy fails, the
   state stays in a private temporary folder, and the script prints the
   command that retries the copy (`--copy-only DIR`).
5. Start the gadget on the Mac: `muse_gadget/mac_gadget.sh start`. Then check
   that `/health` says `"paired": true, "linked": true`.

If the Mac already has a paired gadget, the script stops. Pass `--force` to
pair a new one and replace it. To remove the old device, delete it in the
Muse app.

**Why pairing needs extra Docker options.** Docker's default AppArmor
profile blocks all D-Bus traffic, so the short-lived pairing container runs
with `--security-opt apparmor=unconfined`. To make up for it, it runs as your
user, not root, with every capability dropped and `no-new-privileges`. It
only gets the host's D-Bus system socket and a private temporary state
folder.

### Optional BlueZ tweaks on the PC (sudo; only if pairing fails)

Upstream's installer makes two BlueZ changes on the host. Ours doesn't touch
the host. Make these changes only if pairing fails as described, and undo
them afterwards.

1. **GATT MTU 256.** Android writes packets up to `MTU - 3` bytes and fails
   above 512. The symptom: pairing gets as far as Wi-Fi, then the app says
   "Couldn't connect". We needed this one in practice: our pairing failed
   exactly that way until the MTU was set to 256.
   ```bash
   sudo cp /etc/bluetooth/main.conf /etc/bluetooth/main.conf.pre-muse
   sudo sed -i 's/^#\?\s*ExchangeMTU\s*=.*/ExchangeMTU = 256/' /etc/bluetooth/main.conf
   grep -n '^ExchangeMTU' /etc/bluetooth/main.conf   # must show it under [GATT]
   sudo systemctl restart bluetooth
   # undo:
   sudo mv /etc/bluetooth/main.conf.pre-muse /etc/bluetooth/main.conf && sudo systemctl restart bluetooth
   ```
2. **Battery plugin off.** BlueZ reads a connecting phone's battery level.
   iPhones answer only over a bonded link, so BlueZ asks the phone to bond, and
   iOS pairing can drop. The symptom: an iPhone disconnects during setup.
   ```bash
   sudo mkdir -p /etc/systemd/system/bluetooth.service.d
   printf '[Service]\nExecStart=\nExecStart=/usr/libexec/bluetooth/bluetoothd --noplugin=battery\n' \
     | sudo tee /etc/systemd/system/bluetooth.service.d/zz-muse-gadget.conf
   sudo systemctl daemon-reload && sudo systemctl restart bluetooth
   # undo:
   sudo rm /etc/systemd/system/bluetooth.service.d/zz-muse-gadget.conf
   sudo systemctl daemon-reload && sudo systemctl restart bluetooth
   ```
   Before you add this, check that `systemctl cat bluetooth` shows the same
   `ExecStart` path.

## Running on the Mac

Run these from the PC:

```bash
muse_gadget/mac_gadget.sh start    # idempotent
muse_gadget/mac_gadget.sh status   # machine, container, /health, who listens on 48080
muse_gadget/mac_gadget.sh stop
```

- `start` starts `podman-machine-default` only if it's stopped, and records
  that it did (in `~/assistant-edge/muse-gadget/started-machine` on the Mac).
  It never changes the machine's settings. It builds the arm64 image on the
  Mac if needed, then runs the `muse-gadget` container with
  `-p 127.0.0.1:48080:48080` and the state folder mounted. If something else
  already listens on 48080 (for example a fake bridge), it stops with an
  error.
- `stop` removes only the `muse-gadget` container. It stops the machine only
  if `start` started it, and leaves it running if other containers have
  started since then.
- The gadget doesn't use the robot (no daemon, motors, mic, speaker or
  camera), so it doesn't take the hw-run lock. Whatever drives the robot takes
  the lock itself.
- To clean up completely, run `stop`, then (with the machine running)
  `podman rmi localhost/muse-gadget:latest` on the Mac.

Inside the container the bridge listens on `0.0.0.0:48080`, because a
published port has to reach the container's own interface. That's allowed
only when `MUSE_BRIDGE_IN_CONTAINER=1`, which is set in the image. Anywhere
else the bridge refuses any address that isn't loopback. On the Mac, the only
listener is Podman's `gvproxy` on `127.0.0.1:48080`.

## Robot tools

Muse can move the robot through six gadget commands (`gadget/robot.py`). Each one runs one of
Pollen's own app tools, so the robot only ever plays Pollen's moves:

| Command | Pollen tool | What it does |
|---|---|---|
| `reachy.emotion` `{emotion}` | `play_emotion` | a recorded emotion (`happy`, `sad`, `surprised`, ...) |
| `reachy.dance` `{move?}` | `dance` | a dance move (random if no move) |
| `reachy.stop_move` | `stop_dance` | stop the dance or emotion, clear queued moves |
| `reachy.look` `{direction}` | `move_head` | left, right, up, down or front |
| `reachy.head_tracking` `{enabled}` | `head_tracking` | the SDK face tracker on or off |
| `reachy.status` `{topic}` | `robot_status` | `name`, `software` or `imu` only |

- **Path**: Muse → gadget (Podman) → `POST http://host.containers.internal:48081/tool` →
  MuseHandler's endpoint (`mac/robot_tools.py`) on the Mac's `127.0.0.1:48081` → the app's
  `BackgroundToolManager`, the same way the Hugging Face backend runs a model's tool call. Podman's
  gvproxy forwards `host.containers.internal` to the Mac's loopback, so the endpoint is never on
  the LAN.
- **Only while a run holds the robot.** The endpoint lives inside the app, which runs only while
  `run_poc.sh` holds the hw-run lock. With no run, a command answers "robot is asleep". A command
  never starts the robot.
- **Per-run secret.** `run_poc.sh` makes a random secret on the Mac (0600 file
  `~/assistant-edge/muse-app/run/robot-tools.<run>.env`), passes it to the gadget as Podman's
  `--env-file` and to the app as a file path, and deletes it at cleanup. The endpoint refuses any
  request without it (401), so nothing else on the Mac can drive the robot.
- **Allowlist on both ends.** The gadget checks the enums; the endpoint accepts only the tools
  above (no volume, camera, sleep, memory or web tools) and only those status topics (no Wi-Fi
  address or account). There's no camera or photo command.
- **Log**: the app log has one line per call, `robot tool <name> -> <result>` (no transcript
  text); the gadget logs `robot command <name> -> ok|error`.

### Robot tools by voice

For turns the gadget sends itself, Muse asks for a command with a `client.invoke` event on the
`/chat/subscribe` stream (`command_id`, `invoke_id`, `params_json`, `timeout_ms`), never with
`link.invoke`. **This is undocumented and may change.** The gadget answers it
(`gadget/client_invoke.py`) with the documented `link.result` on `/link-control`, `id` =
`invoke_id`. It uses no other endpoint and adds no fields.

- Only the commands above and `device.health` run, through the same `RestrictedExecutor`; anything
  else (`system.run`, `file.*`, `device.ota`, unknown names) and `params_json` that isn't a JSON
  object get an error result.
- Each `invoke_id` is answered once, also across `link.invoke`. Events without a usable
  `command_id`/`invoke_id` are ignored. At most 4 commands run at a time, as upstream.
- Events are seen only while a subscription is open, i.e. during a `/turn`.
- The log has `client.invoke command=<name> id=<id>` and `... ok=True|False`, never message text.
- `MUSE_CLIENT_INVOKE=0` in the gadget container's environment turns it off.

## Token and privacy

- The SDK token is read from a file or a hidden prompt, never from a
  command-line argument, and it is never printed. It's stored only as
  `sdk_token` (0600) in the gadget's state folder. The gadget reports it to
  Muse when it refreshes its device token, as the SDK does. It's never in git,
  the image, notes or logs.
- The pairing (`pairing.json`) holds the device's access and refresh tokens
  for your Muse account. It exists only on the Mac (0600). Treat it like a
  password.
- If Meta revokes the token, or you stop using this, delete the device in the
  Muse app and delete `~/assistant-edge/muse-state` on the Mac.
- This repo never contains the Mac's address, user name, host name or home
  path. The scripts reach the Mac only as `reachy-mac`, and `$HOME` is
  expanded on the Mac.
- Terms (gadgets.muse.ai/sdk-terms): personal, non-commercial use, and only
  your own Muse account. The bridge uses only documented SDK paths: text
  `/chat/stream` turns and the `/chat/subscribe` event stream the ESP32
  firmware uses.

## Notes from the first paired run (2026-10-03)

- **Side chat.** The firmware subscribes with an empty body (`{}`) and posts
  to the main chat. With a `session_id` on `/chat/stream` and `{}` on
  `/chat/subscribe`, the stream carried only status events, so every turn
  timed out. The bridge therefore sends the same `session_id` in the
  subscribe body (`{"session_id": "..."}`), and the reply events arrive. With
  `MUSE_SESSION_ID=` (empty) it posts to the main chat and subscribes with
  `{}`, exactly as the firmware does.
- The event fields (`type`, `seq`, `event`, `payload.message_id`,
  `payload.reply_to_message_id`, `payload.text`, `payload.display_text`,
  `display_text_ready`, `activity_code`) come from the ESP32 firmware, not
  from API docs.

## POC run

Pollen's conversation app (pinned at `f58523b`, unmodified) runs on the Mac inside Reachy
Edge.app, with `MuseHandler` (`mac/`) as its backend: the robot's mic, then Silero VAD (one
utterance), then local speech-to-text (parakeet-mlx, with mlx-whisper as the fallback), then
`POST /turn` on the bridge at the Mac's `127.0.0.1:48080`. Muse's reply is spoken with Qwen3-TTS
(MLX, on the Mac; Kokoro-82M and then macOS `say` are the fallbacks) on the robot's speaker, and Pollen's wobbler moves
the head while it plays.
It's half-duplex: the mic is ignored while a turn is being transcribed, sent or spoken. There's no
wake word, so anything said near the robot becomes a turn.

Run it from the PC:

```bash
muse_gadget/run_poc.sh                  # real Muse gadget (mac_gadget.sh start/stop)
muse_gadget/run_poc.sh --fake-bridge    # echo bridge: the robot answers "You said: ..."
# options: --duration SECONDS, --lock-timeout SECONDS, --mic-log SECONDS, --log-transcripts,
#          --tts qwen3|kokoro|say, --voice NAME, --instruct TEXT, --volume N (default 100),
#          -- <app args>
```

In order, it:
1. takes the hw-run lock on `reachy-mac` (it waits while another run holds it and only takes over
   a lock whose owner on this PC has died);
2. copies `mac/` to `~/assistant-edge/muse-app/mac` and runs `mac/install.sh` there. The first
   time, that clones the app, builds its venv from `mac/app-constraints.txt` (the app's `uv.lock`
   pins: the Mac's uv can't read that lock format), and caches the STT model in
   `muse-app/hf`. It also builds the Kokoro venv `muse-app/kokoro/.venv` from
   `mac/kokoro-constraints.txt` and caches the Kokoro model (about 1.4 GB in all) and the
   Qwen3-TTS model (about 2.9 GB); if Qwen3 fails, the run uses Kokoro, and if Kokoro fails, `say`.
   Later runs only validate;
3. starts the bridge: `mac_gadget.sh start`, or `mac/fake_bridge.py` with `--fake-bridge`;
4. starts the daemon (`mac/run_daemon.py`: API, WebRTC signalling and mDNS all kept on loopback),
   then the app (`mac/run_app.py`), both inside Reachy Edge.app via `edge_app_run.sh`.

On exit, an error or Ctrl-C, it stops the app, puts the robot to sleep (`goto_sleep`), turns
the motors off, then stops the daemon, unloads this run's `com.assistant.reachy-edge.*` jobs,
stops the bridge and releases the lock. It refuses to start if a daemon or bridge it didn't start
is already running. Once the daemon is up it sets the robot's speaker volume to `--volume`
(default 100; the daemon plays a short test sound) and logs the value before and after.

One-time setup for head tracking: the daemon runs offline (`HF_HUB_OFFLINE=1`, `HF_HOME` is
`~/assistant-edge/hf`), so Pollen's YuNet face model must already be in that cache or
`reachy.head_tracking` crashes the face tracker. `install.sh` doesn't fetch it. Cache it once on
the Mac, with network access, using the daemon's own Python, so the pinned revision matches:

```bash
cd ~/assistant-edge && HF_HOME=$PWD/hf daemon/bin/python -c "from reachy_mini.vision import face_detector as f; \
from huggingface_hub import hf_hub_download as d; print(d(f._MODEL_REPO, f._MODEL_FILE, revision=f._MODEL_REVISION))"
```

This caches `pollen-robotics/face_detection_yunet_2026may` (about 230 KB).

Details:
- `mac/run_app.py` swaps `MuseHandler` in for `HuggingFaceRealtimeHandler` in the app's
  module before the app starts. It sets a placeholder realtime URL, because the app only
  starts its audio loops once one is configured; nothing connects to it.
- Reachy Edge.app always starts `~/assistant-edge/src/.venv-assistant/bin/python`.
  `mac/exec_python.py` swaps that process for the daemon's or the app's own Python, so the app
  is still responsible for the mic. It first removes that venv's GStreamer environment
  variables.
- **Voice (`--tts`, `--voice`)**: Kokoro-82M (`mlx-community/Kokoro-82M-bf16`, pinned) runs
  as a worker process (`mac/kokoro_worker.py`) in its own venv, because mlx-audio needs torch
  and spaCy versions that don't fit the app's pinned venv. It loads once per app start (about
  2.5 s, offline from `muse-app/hf`) and warms up on a short phrase. A reply is split into
  sentences; each is rendered, trimmed of Kokoro's ~0.3 s of leading silence, resampled from 24 kHz
  to the speaker's rate (16 kHz; the app pushes frames to the speaker without resampling), and
  queued right away, so the robot starts talking after the first sentence. The default voice
  is `af_heart` (Kokoro's best-rated English voice: warm and clear, which suits a friendly
  robot); `--voice am_michael` is a good male voice, and `mac/muse_tts.py` lists the rest. If
  Kokoro can't load, the run uses `say`; if it fails during a reply, `say` speaks the rest of
  that reply. `--tts say` (with `--voice <macOS voice>`) uses `say` only.
- **Qwen3-TTS (`--tts qwen3`, the default)**: Qwen3-TTS 1.7B CustomVoice (8-bit,
  `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`, pinned) runs as `mac/qwen3_worker.py` in
  the Kokoro venv (same mlx-audio), and only one TTS model is loaded at a time. The whole reply is
  rendered in one streaming call; each ~0.5 s chunk is resampled to 16 kHz and queued as it
  arrives. On the M4 (no robot), the first audio comes about 295 ms after the reply text, rendering
  runs about 1.7× faster than real time (RTF ~0.58), and loading takes about 5 s with ~3.5 GiB of
  memory. The speaker is `Aiden` (`--voice Ryan` also works, but Ryan is much slower). `--instruct`
  sets the style, which changes how a reply is said, never the words: the default is "playful and
  cheeky, like a friendly cartoon robot", and `--instruct ""` turns it off. If Qwen3 fails before
  a reply's first audio, `say` speaks that reply.
- Measured on the M4 (warm, no robot): Kokoro's first audio comes about 175 ms after the reply
  text arrives (1, 2 or 4 sentences), and it renders about 10× faster than real time.
  `say` takes about 530–610 ms, because it renders the whole reply before any of it plays.
- Speaking or listening needs nobody at the Mac, once Reachy Edge has its microphone grant.
- **What's logged where (no transcripts by default):** prompts and replies aren't kept.
  MuseHandler logs only lengths and timings, and it doesn't hand the text to the upstream app's
  logger (which would print `role=... content=<text>`). The text goes only to the app's live
  transcript push (`_emit_transcript`). On the Mac, each job's output goes to
  `~/assistant-edge/run/<job>/out.log`, which `run_poc.sh` deletes when the job stops. On the PC,
  each run's daemon and app output is in `~/.local/state/muse-poc/<run>.{daemon,app}.log`
  (`MUSE_POC_LOGDIR`, outside the repo), with home paths shown as `~` and any `content=...`
  redacted. The fake bridge logs only turn lengths.
- **`--log-transcripts`** (debugging only) lets the app log each turn's text. `run_poc.sh` then
  shows it on your terminal, but the log files on the PC stay redacted. The Mac's `out.log`
  holds the text until the job stops (or until you delete it, if the run was killed with
  `kill -9`).

Tests (on the PC, with the app's venv; the Silero test runs when the `silero-vad` wheel is
importable):

```bash
cd muse_gadget/mac && <app venv>/bin/python -m pytest tests -q
```
