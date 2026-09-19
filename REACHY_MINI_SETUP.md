# Connecting to Reachy Mini Lite

As of 2026-09-18.

## How the connection works

The robot is a **Reachy Mini Lite**, the USB-tethered model, so a daemon on this PC drives it and Python code talks to that daemon. There is no robot IP address to configure.

```
Python script (ReachyMini())  --HTTP/WebSocket-->  reachy-mini-daemon (localhost:8000)  --USB serial /dev/ttyACM0-->  motors
Camera + microphone           --USB-->             your script
```

The daemon owns the motor port and serves an API at `http://localhost:8000`. `ReachyMini()` detects the Lite model and connects to localhost on its own. The camera and microphone appear as ordinary USB devices.

## Current state

**Resolved (2026-09-18).** All the setup below is done, and the robot moves, speaks, hears and sees. For the conversation app, see [CONVERSATION_APP.md](CONVERSATION_APP.md) (hosted backend) and [LOCAL_CONVERSATION.md](LOCAL_CONVERSATION.md) (fully local).

| USB device                        | ID        | Shows up as    |
| --------------------------------- | --------- | -------------- |
| Motor controller (QinHeng serial) | 1a86:55d3 | `/dev/ttyACM0` |
| Reachy Mini Camera                | 38fb:1002 | video device   |
| Reachy Mini Audio                 | 38fb:1001 | audio device   |

What originally blocked the connection, and how it was fixed:

| Blocker | Fix |
| --- | --- |
| `/dev/ttyACM0` is `root:dialout`, and `desktop` wasn't in `dialout` | `sudo usermod -aG dialout,audio,video $USER` (step 1). Until you log in again, the launchers use `sg`. |
| No udev rule | Installed (step 1). Consider `MODE="0660"` instead of the guide's `0666`, so other local users can't write to the robot's USB devices. |
| `uv pip install reachy-mini` couldn't build `pygobject` | GStreamer and GI dev packages installed (step 2). |

If the robot loses power, the daemon shows "Motor communication error! Check connections and power supply." and doesn't reconnect by itself. Restart it with `./start_daemon.sh` (stop the old one first).

## Setup steps

Steps 1 and 2 need `sudo`. After step 1, log out and back in before continuing.

### 1. USB permissions

This is the udev rule and group change from the [official install guide](https://huggingface.co/docs/reachy_mini/SDK/installation).

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
| sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout $USER
```

To check it after logging back in, run `id`. The output should list `dialout`.

### 2. GStreamer system packages

These are the headers `pygobject` needs to build, taken from the [GStreamer installation guide](https://huggingface.co/docs/reachy_mini/SDK/gstreamer-installation).

```bash
sudo apt-get update
sudo apt-get install libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-base1.0-dev \
  libgstreamer1.0-dev libglib2.0-dev libssl-dev libgirepository1.0-dev libcairo2-dev \
  libportaudio2 libnice10 gstreamer1.0-alsa gstreamer1.0-plugins-bad gstreamer1.0-nice python3-gi-cairo
```

### 3. Install the SDK

```bash
cd ~/codebase/robots
source .venv/bin/activate
uv pip install reachy-mini
```

### 4. Start the daemon

```bash
reachy-mini-daemon
```

Keep this terminal open while you use the robot. Starting the daemon powers the motors, so clear the space around the head first.

If you haven't logged out since step 1, start the daemon with the `dialout` group applied instead:

```bash
sg dialout -c "reachy-mini-daemon"
```

Without it, `http://localhost:8000/api/daemon/status` reports `"state":"error"` and `"error":"Permission denied"`.

### 5. Verify

Open `http://localhost:8000/docs` in a browser. If the Reachy Mini API page loads, the daemon is running and connected.

## First script

With the daemon running, this script wiggles the antennas and returns them to center. It is adapted from the [quickstart](https://huggingface.co/docs/reachy_mini/SDK/quickstart).

```python
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    print("Connected to Reachy Mini!")
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=0.5)
```

Save it as `hello.py`. Then run it from a second terminal with the venv activated:

```bash
cd ~/codebase/robots && source .venv/bin/activate
python hello.py
```

To move the head, use `create_head_pose` from `reachy_mini.utils`, for example `mini.goto_target(head=create_head_pose(z=10, roll=15, degrees=True, mm=True), duration=1.0)`.

## Speaking

The SDK has no text-to-speech, so `say.py` uses [Piper](https://github.com/OHF-Voice/piper1-gpl) to turn text into a WAV file. It then plays that file on the robot's speaker with `aplay` and wiggles the antennas while it talks.

```bash
uv pip install piper-tts
mkdir -p voices && (cd voices && python -m piper.download_voices en_US-lessac-medium)
python say.py "Hello! I am Reachy Mini."
```

The robot's speaker starts out too quiet to hear: `PCM,0` is at -23 dB and `PCM,1` at -20 dB. Set both to full once:

```bash
sg audio -c "amixer -c Audio sset 'PCM',0 100% unmute && amixer -c Audio sset 'PCM',1 100% unmute"
```

It plays straight to the ALSA card `plughw:CARD=Audio,DEV=0` because PipeWire can't see the robot's speaker until you log out and back in after joining the `audio` group. Until then, run it as `sg audio -c "python say.py '...'"`. Don't use the SDK's `write_asoundrc_to_home()`: it makes the robot the default speaker for every app on your account.

## Notes

- **No-terminal option:** the [Reachy Mini Control](https://hf.co/reachy-mini/#/download) desktop app starts the daemon for you and updates the robot software. The docs warn it may not work on some Linux distributions. The Python SDK is the fallback.
- **WebRTC plugin:** the Lite needs it too. In SDK 1.10.0 the daemon's media server fails to start without `webrtcsink` ("Failed to create webrtcsink element"), so the camera is unavailable. Build it with step 3 of the [GStreamer installation guide](https://huggingface.co/docs/reachy_mini/SDK/gstreamer-installation). Audio still works without it if the script passes `media_backend="local"`.
- **Audio and camera permissions:** the speaker, microphone and camera need the `audio` and `video` groups: `sudo usermod -aG audio,video $USER`. Without them, `aplay -l` reports "no soundcards found" unless you're logged in at the machine's own screen.
- **Simulation:** `uv pip install "reachy-mini[mujoco]"`, then run `reachy-mini-daemon --sim` to test code without the robot.
- **Building apps with an AI agent:** Pollen Robotics maintains an [AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md) guide for coding agents.

## Sources

- [reachy_mini on GitHub](https://github.com/pollen-robotics/reachy_mini)
- [Installation guide](https://huggingface.co/docs/reachy_mini/SDK/installation)
- [Quickstart](https://huggingface.co/docs/reachy_mini/SDK/quickstart)
- [GStreamer installation](https://huggingface.co/docs/reachy_mini/SDK/gstreamer-installation)
- [Reachy Mini Lite: get started](https://huggingface.co/docs/reachy_mini/platforms/reachy_mini_lite/get_started)
