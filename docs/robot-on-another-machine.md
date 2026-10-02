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
- **The daemon** runs with its API on `127.0.0.1:8000` on the edge host, with no media and no
  wake-up or sleep motion. The PC reads it through an `ssh -L` tunnel. Upstream behaviour: the
  reachy-mini daemon always announces itself over mDNS (UDP 5353, service `reachy_mini`) on the
  edge host's network, advertising port 8000. The port is bound to loopback, so the
  announcement points at something the LAN cannot reach.
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
