# Talking to Reachy Mini: Conversation app on Linux

Started 2026-09-18. Goal: run Pollen's [conversation app](https://github.com/pollen-robotics/reachy_mini_conversation_app) on this Ubuntu PC with the Reachy Mini Lite, so you can talk to the robot and it answers out loud.

## How it works

```
Robot mic ──USB──▶ conversation app (mini.media.get_audio_sample, 16 kHz)
                        │  WebSocket audio stream
                        ▼
              Hugging Face realtime backend (cloud, no API key by default)
              speech-to-text → LLM → text-to-speech, plus tool calls
                        │  audio + tool calls
                        ▼
conversation app ──▶ mini.media.push_audio_sample ──USB──▶ robot speaker
                 └─▶ tools (dance, emotion, move_head, camera…) ──▶ daemon ──▶ motors
The daemon's "wobbler" moves the head in time with the speech it plays.
```

The app reaches the robot's audio through the SDK's media layer. On macOS that works out of the box. On Linux it needs two things we didn't have:

1. **The GStreamer WebRTC plugin (`webrtcsink`).** Without it, the daemon's media server fails to start, so there's no camera feed, no head wobble, and the SDK falls back to a WebRTC backend that can't connect.
2. **An audio route to the Reachy card.** The SDK plays through PipeWire unless `~/.asoundrc` defines `reachymini_audio_sink` and `reachymini_audio_src`. PipeWire can't see the card until you log out and back in after joining the `audio` group.

## Plan

1. Build the WebRTC plugin from `gst-plugins-rs` 0.14.5, installed under the home directory so it needs no `sudo`.
2. Give the SDK a direct ALSA route to the Reachy card.
3. Restart the daemon with the plugin and the `dialout` and `audio` groups.
4. Install the conversation app in its own venv.
5. Run it and talk to it.

## Log

### Step 1: WebRTC plugin (done)

- Cloned `gst-plugins-rs` at tag `0.14.5` into `third_party/gst-plugins-rs`. The first clone from gitlab.freedesktop.org got "Connection reset by peer"; a retry worked.
- The plugin needs Rust 1.83 or newer, and the default toolchain here is 1.82. Instead of changing the default, I installed Rust 1.90.0 alongside it with `rustup toolchain install 1.90.0 --profile minimal` and build with `rustup run 1.90.0 ...`.
- The latest `cargo-c` requires Rust 1.96, so I pinned a compatible version: `rustup run 1.90.0 cargo install cargo-c --version 0.10.19 --locked`.
- The build installs to `~/.local/gst-plugins-rs` instead of `/opt/gst-plugins-rs`, so no `sudo` is needed:

```bash
cd third_party/gst-plugins-rs
rustup run 1.90.0 cargo cinstall -p gst-plugin-webrtc --prefix=$HOME/.local/gst-plugins-rs --release
```

The build took 1 min 42 s on this 24-core machine. To check it, run `GST_PLUGIN_PATH=$HOME/.local/gst-plugins-rs/lib/x86_64-linux-gnu gst-inspect-1.0 webrtcsink`, which should report version `0.14.5-a18c320`. I didn't add `GST_PLUGIN_PATH` to `~/.bashrc`. The launcher scripts set it instead.

### Step 2: audio route (done)

I wrote `~/.asoundrc` with two named ALSA devices on the Reachy card (`hw:Audio,0`): `reachymini_audio_sink` (dmix, for playback) and `reachymini_audio_src` (dsnoop, for capture). With both present, the daemon and the SDK use `alsasrc`/`alsasink` on them instead of PipeWire, and several processes can share the card at once.

This deliberately differs from the SDK's `write_asoundrc_to_home()`: that also redefines `pcm.!default`, which would make the robot the default speaker for every app on the account. To undo it, delete `~/.asoundrc`.

Checked: `has_reachymini_asoundrc()` returns `True`, and `aplay`/`arecord` both work through the two devices.

### Step 3: daemon restarted with media (done)

`start_daemon.sh` sets `GST_PLUGIN_PATH`, re-runs itself under `sg` for any of `dialout`, `audio` or `video` the session is missing, and starts the daemon. With the plugin and `~/.asoundrc` in place, the daemon log shows the media server starting:

```
media_server - INFO - Using ALSA device reachymini_audio_src for capture.
media_server - INFO - Using ALSA device reachymini_audio_sink for playback.
gst_plugin_webrtc_signalling::handlers: registered as [Producer]
```

`/api/media/status` now reports `"available": true`, and the daemon reports `camera_specs_name: "lite"`. From a client, `ReachyMini()` auto-selects `MediaBackend.LOCAL` and uses the two shared ALSA devices. `get_frame()` returned a 1920×1080 camera frame.

The log also shows one warning at startup, `Can't record audio fast enough`, which is common as a pipeline starts.

**A mistake to avoid when stopping the daemon:** `pgrep -f bin/reachy-mini-daemon` also matches the `sg`/`sh` wrapper processes, so my first `kill -INT` hit a wrapper and left the old daemon holding port 8000 and `/dev/ttyACM0`. The new daemon then failed with `Device or resource busy` and `address already in use`. To stop it correctly, signal the Python process, or press Ctrl+C in its terminal.

### Step 4: conversation app installed (done)

```bash
git clone --depth 1 https://github.com/pollen-robotics/reachy_mini_conversation_app
cd reachy_mini_conversation_app
uv venv --python 3.12 .venv && uv sync --frozen
uv pip install --python .venv/bin/python "reachy-mini==1.10.0"   # match the daemon's version
```

This installed app version 1.0.1 (commit `f58523b`, 2026-09-17). Its lock file pins `reachy-mini` 1.10.0rc5, so I upgraded the venv to 1.10.0 to match the daemon and avoid a version-mismatch warning.

**Groups without logging out:** a nested `sg` gives one process all the groups it needs, for example `sg dialout -c "sg audio -c '...'"`. Logging out and back in makes this unnecessary.

### Step 5: running (done)

```bash
./start_daemon.sh                 # terminal 1, keep it open
./start_conversation.sh --ui      # terminal 2; add --no-camera to skip vision
```

Startup log highlights (2026-09-18):

- `POST https://pollen-robotics-reachy-mini-realtime-url.hf.space/session "200 OK"`, then `Allocated realtime session …`. No API key or HF token was needed.
- 19 tools were registered: dance, emotions, camera, move_head, sweep_look, head_tracking, remember/forget, volume_control, robot_status, task status/cancel, plus web search, time and weather through Pollen's Hugging Face Spaces.
- `Head wobbling enabled`, and the voice is `Aiden`.
- The app applied the audio board's processing settings (automatic gain, noise suppression).
- The robot greeted: "Hi, I'm Reachy Mini—what would you like to explore together today?"
- The web UI is at http://127.0.0.1:7860/ and shows live transcripts, personalities, tools and settings. (The app itself binds it to 0.0.0.0, which makes it reachable from the LAN; since the audit fixes, `start_conversation.sh` runs the app through `local_backend/run_app.py`, which binds it to 127.0.0.1. Set `REACHY_UI_HOST=0.0.0.0` to open it again.)

Just talk to the robot. The app streams the microphone continuously and the backend detects when you stop speaking. To end, say "go to sleep" or press Ctrl+C in terminal 2.

## Files and changes made

| Path | What |
| --- | --- |
| `~/.asoundrc` | New file: shared ALSA devices on the Reachy card (delete it to undo) |
| `~/.local/gst-plugins-rs/` | GStreamer WebRTC plugin 0.14.5 |
| `~/.rustup` toolchain `1.90.0`, `~/.cargo/bin/cargo-c*` | Build tools; your default Rust toolchain is unchanged |
| `third_party/gst-plugins-rs/` | Plugin source and build output |
| `reachy_mini_conversation_app/` | The app and its own `.venv` |
| `start_daemon.sh`, `start_conversation.sh` | Launchers: plugin path plus `sg` group handling. Since the audit fixes, the daemon also runs offline with signalling on 127.0.0.1, and the app UI binds to 127.0.0.1; see LOCAL_CONVERSATION.md. |
