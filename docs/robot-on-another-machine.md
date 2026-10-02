# Running robot tests with the robot on another machine

The robot tests (`tier: hw` features, gate stage g) need the real Reachy Mini. Its USB cable may
be plugged into another machine, for example a Mac, instead of the brain PC. The brain, the
models and the test runner stay on the PC. The device side (the reachy-mini daemon and, from S3,
the edge agent) runs on the machine with the robot, called the **edge host**. The PC drives it
over SSH.

A robot attached to the PC (`/dev/ttyACM*`) is always used first. Otherwise the tests use the
edge host, and the robot must be visible **there** (`/dev/cu.usbmodem*` on macOS,
`/dev/ttyACM*` on Linux). If neither machine has it, the hw stage fails. It is never skipped.

## One-time setup on the PC

1. Key-based SSH from the PC to the edge host must work without a password prompt.
2. Give the edge host an alias in `~/.ssh/config` on the PC. The repo only ever uses the alias;
   never commit an address or a user name. A control master makes the many short SSH calls
   fast:

   ```sshconfig
   Host reachy-mac
       HostName <the edge host's LAN address>
       User <your user on the edge host>
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 10m
       ServerAliveInterval 15
   ```

3. The alias the tests use is set in `config/assistant.toml`:

   ```toml
   [test.edge_host]
   ssh = "reachy-mac"
   ```

   The env var `ASSISTANT_EDGE_HOST=<alias>` overrides it for one run.

Check it:

```bash
UV_PROJECT_ENVIRONMENT=.venv-assistant uv run python -m assistant_testing.edge_host check
# robot: /dev/cu.usbmodem... on edge host 'reachy-mac' (over SSH)
```

## What happens on the edge host

`scripts/edge_host_bootstrap.sh` prepares the edge host. The feature
`e2e/features/robot/edge_host_ready.yaml` runs it on every gate run, and it only validates once
everything is installed. You can also run it by hand:

```bash
ssh reachy-mac 'bash -s' < scripts/edge_host_bootstrap.sh
```

The script needs only `bash`, `git`, `curl` and `perl` there. It reuses `uv` and a uv-managed
Python 3.12, and installs them only if they are missing. Everything else goes into
`~/assistant-edge/`:

| Path | What |
|---|---|
| `daemon/` | venv with `reachy-mini==1.10.0` (the same version as the PC) and its daemon. On macOS the SDK's media stack comes from the `gstreamer-bundle` wheel, so no Homebrew packages are needed |
| `repo.git` | bare repo. The tests `git push` the exact commit under test here over SSH (never GitHub) |
| `src/` | checkout of that commit. `uv sync --locked --package assistant-edge --package assistant-robot-reachy` installs only the edge packages into `src/.venv-assistant` |
| `cache/uv`, `hf/` | uv cache and `HF_HOME`, kept here so nothing else on the host changes |

## How the tests reach it

- **Processes** started on the edge host (`ProcessGroup.start(..., ssh=alias)`) run through
  SSH in their own process group. Their output streams back to the PC, steps can type into
  them, and `kill_process` sends real STOP/CONT/KILL to the remote group. Teardown always stops
  them.
- **No orphans, even after a crash.** Every ssh process the tests start on the PC (remote
  launchers, tunnels, one-off commands) carries the tag `-o SetEnv=ASSISTANT_TEST_RUN=<run id>`
  and, on Linux, a parent-death signal (`PR_SET_PDEATHSIG`), so it exits when the test runner
  dies, even with `kill -9`. EdgeLink's `ssh -R` forwards use edge-host ports 47000-47999 only.
  `python -m assistant_testing.edge_host sweep` finds and stops what is left: processes running
  from `~/assistant-edge` on the edge host, tagged ssh processes on the PC, and sshd listeners
  in that port range on the edge host. The gate sweeps before the hw features (an earlier run's
  leftovers are reported and cleaned) and after them; leftovers after the features fail the
  stage.
- **One hw run at a time.** Before its pre-run sweep the gate takes an exclusive lock on the
  edge host (`~/assistant-edge/hw-run.lock`, `python -m assistant_testing.edge_host lock`) and
  releases it when the stage ends (`unlock`). A second run fails fast with "another hw run is
  in progress" instead of sweeping away the first run's processes. A lock whose owner (a gate
  on the same PC) has died is taken over.
- **No user names in logs.** Output from the edge host (sweep, launchers, bootstrap, sync)
  shows its `$HOME` as `~`.
- **Tunnel ports.** An `ssh -R` port counts as ours only if, after the port is seen listening,
  our ssh is still running and did not log "remote port forwarding failed" (someone else may
  have taken the port first); otherwise another port is tried.
- **The daemon** runs WITH media (camera, WebRTC) and no wake-up or sleep motion of its own
  (the robot features' moves do `wake_up()` -> a small move -> `goto_sleep()` -> motors off;
  see the safety rules in `e2e/features/README.md`), started as
  `python -m assistant_robot_reachy.daemon` from the synced checkout. That launcher keeps every
  socket on loopback: the API on `127.0.0.1:8000`, the WebRTC signalling server (upstream:
  `0.0.0.0:8443`) on `127.0.0.1:8443`, and no mDNS announcement (upstream: UDP 5353 on every
  interface). The hw feature `robot/daemon_connect.yaml` checks it with `lsof`. The PC reads
  the API through an `ssh -L` tunnel.
- **Reachy Edge.app owns the camera and microphone (macOS).** macOS grants camera and
  microphone access to the *responsible* process of whatever opens them. For anything started
  over SSH that is `sshd`, which can never be granted, so the camera fails and the microphone
  records silence. `scripts/edge_host_bootstrap.sh` therefore builds `~/assistant-edge/Reachy
  Edge.app` (bundle id `com.assistant.reachy-edge`, no Dock icon, usage text "Reachy Mini
  robot: voice and camera"). Its executable is a small compiled launcher that starts the synced
  venv's Python with the requested module as a child and stays alive, so the app stays
  responsible for it. The tests run the daemon and the reachy edge agent inside the app through
  `scripts/edge_app_run.sh`: a per-run LaunchAgent (`com.assistant.reachy-edge.<run>.<name>.<pid>`)
  in the logged-in user's GUI session, whose output streams back over SSH and whose stdin is
  fed through a FIFO. The feature `robot/daemon_connect.yaml` checks that both report Reachy
  Edge as their responsible process.
  - **Granting (once).** The first run shows "Reachy Edge would like to access the microphone"
    and then "... the camera" on the Mac's screen; someone at the Mac clicks **Allow** on both.
    The user must be logged in at the Mac (the LaunchAgent runs in that session).
  - **Signed once.** The app is ad-hoc signed and rebuilt only when its launcher source or
    Info.plist changes (the bootstrap keeps a hash of both); a re-sign changes its code hash and
    macOS would ask again. Our Python code lives outside the bundle, so code syncs never touch
    it, and a uv reinstall of Python does not affect the grant.
  - **Python needs no permission.** The shared uv `python3.12` is never the responsible
    process; leave it off (or remove it) under Privacy & Security.
  - **Revoking.** System Settings > Privacy & Security > Microphone (and > Camera): switch off
    or remove "Reachy Edge". To reset it from a terminal: `tccutil reset Microphone
    com.assistant.reachy-edge` and `tccutil reset Camera com.assistant.reachy-edge`.
  - **Leftovers.** The sweep unloads any `com.assistant.reachy-edge.*` LaunchAgent a crashed
    run left behind.
- **Xcode command-line tools** are needed once on the Mac to compile the launcher
  (`xcode-select --install`); the bootstrap fails with that hint if they are missing.
- **EdgeLink** stays on `127.0.0.1` on the PC. A process on the edge host reaches it through an
  `ssh -R` tunnel to `127.0.0.1` there. Plain `ws://` never crosses the LAN. S7 replaces the
  tunnel with TLS and pairing.

## Running

```bash
scripts/gate.sh                                    # stage g prints which host has the robot
UV_PROJECT_ENVIRONMENT=.venv-assistant uv run pytest e2e -m hw -v   # only the robot features
ssh reachy-mac 'pgrep -fl reachy'                  # afterwards: nothing left running
```

The robot must not be in use by anything else on the edge host. If a daemon the test did not
start is already answering on port 8000, the daemon step fails rather than stopping it.

If `edge_body_is aec: hw` fails with every XVF3800 echo-canceller parameter "unreadable",
the audio board's DSP control has got stuck (seen once on firmware 2.1.2; the board keeps
answering "retry"). Reboot the audio board alone, with no daemon running and no motion:

```bash
ssh reachy-mac 'cd ~/assistant-edge/src && .venv-assistant/bin/python -m reachy_mini.media.audio_control_utils REBOOT --values 1'
```
