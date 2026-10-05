"""Edge-host steps: the real robot on whichever machine it is plugged into.

The host comes from `assistant_testing.edge_host.resolve`: this machine when the robot is
attached here, else the SSH alias from config `[test.edge_host] ssh` / ASSISTANT_EDGE_HOST.
On the physical robot nothing here is simulated: the bootstrap script, `git push`, `uv sync`,
the reachy-mini daemon and the SSH tunnels are all real, and every process is stopped in
teardown.

The daemon runs with media (camera, WebRTC) through `python -m assistant_robot_reachy.daemon`,
which keeps every one of its sockets on loopback. Its HTTP API stays on 127.0.0.1 on the edge
host; the PC reaches it through an `ssh -L` tunnel. EdgeLink stays on 127.0.0.1 on the PC; an
edge-host process reaches it through an `ssh -R` tunnel (S7 replaces the tunnel with TLS and
pairing).

On the simulated robot (a `[sim, hw]` scenario's `[sim]` item, `ctx.sim`) the same steps drive
Pollen's daemon in MuJoCo on this PC (`assistant_testing.sim`): `robot_host_found` plugs in
the sim's sound card, there is no code to sync (the edge agent and the daemon run from this
checkout), and the daemon is `python -m assistant_robot_reachy.sim` from `.venv-sim`.
"""

import asyncio
import json
import random
import re
import shlex
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from assistant_testing import edge_host as eh
from assistant_testing import sim
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.processes import (
    REMOTE_STOP_GRACE_S,
    ManagedProcess,
    RemoteProcess,
    home_scrubber,
    ssh_tag_options,
)

DAEMON = "daemon"
DAEMON_PORT = 8000
DAEMON_MODULE = "assistant_robot_reachy.daemon"
"""The daemon WITH media, every socket on loopback (WebRTC signalling 8443 too, no mDNS)."""
DAEMON_ARGS = (
    "--no-wake-up-on-start",  # no motion: the motors stay as they are (disabled)
    "--no-goto-sleep-on-stop",
    "--dataset-update-interval",
    "0",
    "--fastapi-host",
    "127.0.0.1",
    "--fastapi-port",
    str(DAEMON_PORT),
)
DAEMON_ENV = {"HF_HOME": "~/assistant-edge/hf", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
BOOTSTRAP = Path("scripts/edge_host_bootstrap.sh")
_READY = re.compile(r"^edge-host ready: (.*)$", re.M)


def host_of(ctx: ScenarioContext) -> eh.EdgeHost:
    host = ctx.state.get("edge_host")
    if host is None:
        raise AssertionError("no edge host selected; use robot_host_found first")
    return host


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ssh_tunnel_argv(alias: str, *forward: str) -> list[str]:
    """A dedicated SSH connection (not the shared ControlMaster) that only forwards ports.

    Tagged as a test-run ssh; it gets SIGTERM if the test runner dies (ProcessGroup.start).
    """
    return [
        "ssh", *ssh_tag_options(), "-S", "none", "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=5", "-N", *forward, alias,
    ]  # fmt: skip


async def _scrubber(host: eh.EdgeHost) -> Callable[[str], str]:
    """Rewrites the edge host's $HOME to `~` in output (asked once, off the event loop)."""
    return home_scrubber(await asyncio.to_thread(host.home))


async def _run(
    ctx: ScenarioContext,
    name: str,
    argv: list[str],
    timeout_s: float,
    scrub: Callable[[str], str] | None = None,
) -> str:
    done = await ctx.processes.run(name, argv, timeout_s=timeout_s, scrub=scrub)
    if done.returncode != 0:
        raise AssertionError(f"{name} failed (exit {done.returncode}):\n{done.output}")
    return done.output


def _http_json(url: str, timeout_s: float = 5.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read())


async def _get(url: str, timeout_s: float = 5.0) -> Any:
    return await asyncio.to_thread(_http_json, url, timeout_s)


# ---------------------------------------------------------------- host, bootstrap, sync


@step("robot_host_found")
async def robot_host_found(ctx: ScenarioContext) -> None:
    """Pick the machine with the robot: this one if attached here, else the edge host over SSH.

    Fails (never skips) when neither has the robot's USB serial device. On the simulated robot:
    this machine, once the sim env is ready and the sim's sound card is plugged in.
    """
    if ctx.sim:
        await _sim_robot_found(ctx)
        return
    host = await asyncio.to_thread(eh.resolve, ctx.repo_root)
    device = await asyncio.to_thread(eh.robot_device, host)
    if device is None:
        raise AssertionError(
            f"no robot on {host.label} (looked for {', '.join(eh.ROBOT_GLOBS)}); "
            f"configure [test.edge_host] ssh or {eh.ENV_VAR}, or plug the robot in here"
        )
    ctx.state["edge_host"] = host
    ctx.state["robot_device"] = device
    print(f"robot: {device} on {host.label}")


async def _sim_robot_found(ctx: ScenarioContext) -> None:
    facts = await asyncio.to_thread(sim.prepare, ctx.repo_root)
    robot = sim.SimRobot(ctx.repo_root, ctx.processes, ctx.feature_path.stem)
    ctx.cleanups.append(lambda: print(robot.unplug()))
    await robot.plug_in()
    ctx.state["sim"] = robot
    ctx.state["edge_host"] = eh.EdgeHost(None)
    ctx.state["robot_device"] = "sim"
    print(f"robot: the simulated Reachy Mini (Pollen's daemon, MuJoCo) on this machine; {facts}")


def sim_robot(ctx: ScenarioContext) -> sim.SimRobot:
    robot = ctx.state.get("sim")
    if robot is None:
        raise AssertionError("no simulated robot; use robot_host_found first")
    return robot


@step("edge_host_bootstrapped")
async def edge_host_bootstrapped(ctx: ScenarioContext, reachy_mini: str) -> None:
    """Run scripts/edge_host_bootstrap.sh there (idempotent: installs once, then validates).

    Checks the pinned reachy-mini version and that the script saw the robot.
    """
    if ctx.sim:
        raise AssertionError("the edge host bootstrap needs the physical robot (tier hw only)")
    host = host_of(ctx)
    script = (ctx.repo_root / BOOTSTRAP).read_text()
    argv = host.sh(f"exec /bin/bash -c {shlex.quote(script)} edge_host_bootstrap")
    output = await _run(ctx, "bootstrap", argv, timeout_s=1200, scrub=await _scrubber(host))
    found = _READY.findall(output)
    if not found:
        raise AssertionError(f"bootstrap printed no 'edge-host ready' line:\n{output}")
    facts = dict(item.split("=", 1) for item in found[-1].split())
    ctx.state["edge_facts"] = facts
    print(f"edge host facts: {facts}")
    if facts.get("reachy_mini") != reachy_mini:
        raise AssertionError(f"reachy-mini {facts.get('reachy_mini')} there, want {reachy_mini}")
    if facts.get("robot", "none") == "none":
        raise AssertionError(f"bootstrap found no robot device on {host.label}")
    if facts.get("os") == "Darwin" and facts.get("app") not in ("built", "kept"):
        raise AssertionError(f"bootstrap did not set up Reachy Edge.app: {facts}")
    if not facts.get("emotions", "0").isdigit() or int(facts.get("emotions", "0")) == 0:
        raise AssertionError(
            f"Pollen's emotions dataset is not cached on {host.label} (the robot's moves): {facts}"
        )


@step("code_synced_to_edge_host")
async def code_synced_to_edge_host(ctx: ScenarioContext) -> None:
    """Ship the commit under test (HEAD of this checkout) and `uv sync` the edge packages there.

    `git push` to the edge host's bare repo over SSH (never GitHub), check it out in
    ~/assistant-edge/src, then `uv sync --locked` only the edge/robot packages into
    .venv-assistant and import them from that env. The simulated robot runs this checkout as it
    is: nothing to ship.
    """
    if ctx.sim:
        print("sim: the edge agent and the daemon run from this checkout (nothing to sync)")
        return
    host = host_of(ctx)
    rev = await _run(
        ctx, "git-rev-parse", ["git", "-C", str(ctx.repo_root), "rev-parse", "HEAD"], 30
    )
    sha = rev.strip()
    target = (
        f"{host.ssh}:assistant-edge/repo.git"
        if host.ssh
        else str(Path.home() / "assistant-edge" / "repo.git")
    )
    await _run(
        ctx,
        "git-push",
        ["git", "-C", str(ctx.repo_root), "push", "-q", "-f", target, f"{sha}:{eh.SYNC_REF}"],
        timeout_s=120,
    )
    packages = " ".join(f"--package {p}" for p in eh.EDGE_PACKAGES)
    script = f"""set -e
cd "$HOME/assistant-edge/src"
git fetch -q origin {eh.SYNC_REF}
git checkout -q -f --detach {sha}
git clean -fdxq -e /.venv-assistant
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
export UV_PROJECT_ENVIRONMENT=.venv-assistant UV_CACHE_DIR="$HOME/assistant-edge/cache/uv"
export UV_NO_PROGRESS=1
"$UV" sync -q --locked {packages}
echo "synced $(git rev-parse HEAD)"
.venv-assistant/bin/python -c 'import assistant_edge, assistant_link, assistant_robot_reachy; \
print("imports ok from", assistant_edge.__file__)'
"""
    output = await _run(ctx, "sync", host.sh(script), timeout_s=600, scrub=await _scrubber(host))
    if f"synced {sha}" not in output or "imports ok from" not in output:
        raise AssertionError(f"edge host did not end up on {sha}:\n{output}")
    if "/assistant-edge/src/" not in output:
        raise AssertionError(f"edge packages not imported from the synced checkout:\n{output}")
    ctx.state["edge_sha"] = sha


# ---------------------------------------------------------------- the reachy-mini daemon


async def _daemon_answers(host: eh.EdgeHost) -> bool:
    probe = f"curl -s -m 3 -o /dev/null http://127.0.0.1:{DAEMON_PORT}/api/daemon/status"
    return (await asyncio.to_thread(host.run, probe, 30)).returncode == 0


@step("start_reachy_daemon")
async def start_reachy_daemon(ctx: ScenarioContext, ready_within_s: float = 120.0) -> None:
    """Start the real reachy-mini daemon WITH media on the robot's machine, all on loopback.

    Runs `python -m assistant_robot_reachy.daemon` from the synced checkout (see
    `code_synced_to_edge_host`). Fails if a daemon this scenario did not start already answers
    there (the test stops only what it started). No motion: the robot is not woken up and not
    put to sleep.
    """
    host = host_of(ctx)
    if await _daemon_answers(host):
        raise AssertionError(
            f"a reachy-mini daemon is already answering on {host.label} port {DAEMON_PORT}; "
            "stop it first (this test starts and stops its own)"
        )
    await _start_daemon(ctx, host, ready_within_s)
    ctx.state["daemon_started"] = True


@step("reachy_daemon_running")
async def reachy_daemon_running(ctx: ScenarioContext, ready_within_s: float = 120.0) -> None:
    """Make sure the real daemon runs: reuse one that already answers, else start one (with
    media, on loopback). Teardown stops only a daemon this scenario started."""
    await ensure_reachy_daemon(ctx, ready_within_s)


async def ensure_reachy_daemon(ctx: ScenarioContext, ready_within_s: float = 120.0) -> str:
    """The daemon's URL as seen from here; starts the daemon only if none answers (the sim's
    is always started: a daemon already answering on this PC is not the simulated robot's)."""
    host = host_of(ctx)
    if ctx.sim:
        await start_reachy_daemon(ctx, ready_within_s)
    elif await _daemon_answers(host):
        await _daemon_tunnel(ctx, host)
        ctx.state["daemon_started"] = False
        print(f"reusing the reachy-mini daemon already running on {host.label}")
    else:
        await _start_daemon(ctx, host, ready_within_s)
        ctx.state["daemon_started"] = True
    return ctx.state["daemon_url"]


async def _rest_before_daemon_stops(host: eh.EdgeHost) -> None:
    if rested := await asyncio.to_thread(eh.rest_robot, host):
        print(f"robot put to rest before the daemon stopped: {rested}")


async def _start_daemon(ctx: ScenarioContext, host: eh.EdgeHost, ready_within_s: float) -> None:
    ready = rf"Uvicorn running on http://127\.0\.0\.1:{DAEMON_PORT}"
    if ctx.sim:  # Pollen's daemon on the MuJoCo robot, from .venv-sim
        await sim.start_daemon(
            sim_robot(ctx),
            list(DAEMON_ARGS),
            name=DAEMON,
            ready_timeout=ready_within_s,
            before_stop=lambda: _rest_before_daemon_stops(host),
        )
        ctx.state["daemon_url"] = f"http://127.0.0.1:{DAEMON_PORT}"
        return
    # Whatever ends the scenario: the motors are disabled before the daemon stops.
    ctx.processes.before_stop[DAEMON] = lambda: _rest_before_daemon_stops(host)
    if host.ssh is None:
        await ctx.processes.start(
            DAEMON,
            [sys.executable, "-m", DAEMON_MODULE, *DAEMON_ARGS],
            env={k: str(Path.home() / v[2:]) if v.startswith("~/") else v
                 for k, v in DAEMON_ENV.items()},
            ready_line=ready,
            ready_timeout=ready_within_s,
        )  # fmt: skip
        ctx.state["daemon_url"] = f"http://127.0.0.1:{DAEMON_PORT}"
        return
    if ctx.state.get("edge_sha") is None:
        raise AssertionError("code not synced to the edge host; use code_synced_to_edge_host")
    args = ["-m", DAEMON_MODULE, *DAEMON_ARGS]
    if await is_mac(host):  # camera and microphone: inside Reachy Edge.app
        argv = app_argv(DAEMON, args, env=DAEMON_ENV)
    else:
        argv = [f"{eh.REMOTE_VENV}/bin/python", *args]
    await ctx.processes.start(
        DAEMON,
        argv,
        env=DAEMON_ENV,
        ssh=host.ssh,
        remote_cwd=eh.EDGE_DIR,
        ready_line=ready,
        ready_timeout=ready_within_s,
    )
    await _daemon_tunnel(ctx, host)


async def _daemon_tunnel(ctx: ScenarioContext, host: eh.EdgeHost) -> None:
    """`ssh -L` from a free local port to the daemon API (none needed for a local robot)."""
    if host.ssh is None:
        ctx.state["daemon_url"] = f"http://127.0.0.1:{DAEMON_PORT}"
        return
    local_port = _free_port()
    await ctx.processes.start(
        "tunnel:daemon",
        _ssh_tunnel_argv(host.ssh, "-L", f"{local_port}:127.0.0.1:{DAEMON_PORT}"),
    )
    url = f"http://127.0.0.1:{local_port}"
    deadline = time.monotonic() + 15
    while True:
        try:
            await _get(f"{url}/api/daemon/status", 3)
            break
        except (OSError, urllib.error.URLError) as exc:
            tunnel = ctx.processes.get("tunnel:daemon")
            if not tunnel.running or time.monotonic() > deadline:
                raise AssertionError(
                    f"daemon API not reachable through the SSH tunnel: {exc}\n{tunnel.output}"
                ) from None
            await asyncio.sleep(0.3)
    ctx.state["daemon_url"] = url
    print(f"daemon API on {host.label} reached through ssh -L at {url}")


# ---------------------------------------------------------------- Reachy Edge.app (macOS)


APP_RUN = "~/assistant-edge/src/scripts/edge_app_run.sh"
"""Runs a module of the synced checkout inside Reachy Edge.app as a per-run LaunchAgent."""
APP_EXECUTABLE = "Reachy Edge.app/Contents/MacOS/reachy-edge"
_OS: dict[str | None, str] = {}


async def is_mac(host: eh.EdgeHost) -> bool:
    """Whether the robot's machine runs macOS (asked once per host)."""
    if host.ssh not in _OS:
        done = await asyncio.to_thread(host.run, "uname -s", 30)
        _OS[host.ssh] = done.stdout.strip()
    return _OS[host.ssh] == "Darwin"


def app_argv(
    name: str, args: list[str], *, env: dict[str, str] | None = None, stdin: bool = False
) -> list[str]:
    """argv (run over SSH) that runs `python <args>` inside Reachy Edge.app on a macOS edge
    host. The app (built by edge_host_bootstrap.sh) is the process macOS holds responsible for
    the camera and microphone; its output streams back, after an `APP-JOB pid=` line."""
    argv = ["/bin/sh", APP_RUN, name, *(["--stdin"] if stdin else [])]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    return [*argv, "--", *args]


def app_job_pgid(proc: ManagedProcess) -> int | None:
    """The process group of the app job behind `proc` (from its `APP-JOB` line), if any."""
    for line in proc.lines:
        if line.startswith("APP-JOB "):
            fields = dict(f.split("=", 1) for f in line.split()[1:] if "=" in f)
            pid = fields.get("pid", "")
            return int(pid) if pid.isdigit() else None
    return None


_RESPONSIBLE = """
import ctypes, os, subprocess, sys
lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
lib.responsibility_get_pid_responsible_for_pid.argtypes = [ctypes.c_int]
def path(pid):
    buf = ctypes.create_string_buffer(4096)
    n = lib.proc_pidpath(pid, buf, 4096)
    return buf.value.decode() if n > 0 else "?"
pids = subprocess.run(["pgrep", "-g", sys.argv[1]], capture_output=True, text=True).stdout
for pid in pids.split():
    rpid = lib.responsibility_get_pid_responsible_for_pid(int(pid))
    print(pid, path(int(pid)), "->", rpid, path(rpid), sep="\t")
"""


@step("responsible_process_is_app")
async def responsible_process_is_app(ctx: ScenarioContext, process: str = DAEMON) -> None:
    """macOS: every process of `process` (`daemon` or `client:<id>`) has Reachy Edge.app as
    its responsible process, so the app's camera and microphone permission applies (and
    the shared Python needs none)."""
    host = host_of(ctx)
    if not await is_mac(host):
        print(f"{host.label} is not macOS: no responsible-process rule")
        return
    pgid = app_job_pgid(ctx.processes.get(process))
    if pgid is None:
        raise AssertionError(f"{process} was not started inside Reachy Edge.app (no APP-JOB line)")
    python = '"$HOME"/assistant-edge/src/.venv-assistant/bin/python'
    script = f"{python} -c {shlex.quote(_RESPONSIBLE)} {pgid}"
    done = await asyncio.to_thread(host.run, script, 30)
    rows = [host.scrub(line).split("\t") for line in done.stdout.splitlines() if line.strip()]
    if not rows:
        raise AssertionError(f"no processes in group {pgid}: {host.scrub(done.stderr)}")
    for pid, exe, _arrow, rpid, responsible in rows:
        print(f"{pid} {exe} -> responsible {rpid} {responsible}")
        if not responsible.endswith(APP_EXECUTABLE):
            raise AssertionError(f"{exe} (pid {pid}) is the responsibility of {responsible}")
    if not any("/python" in exe for _, exe, *_ in rows):
        raise AssertionError(f"no Python process inside the app job: {rows}")


def _daemon_url(ctx: ScenarioContext) -> str:
    url = ctx.state.get("daemon_url")
    if url is None:
        raise AssertionError("no daemon started; use start_reachy_daemon first")
    return url


@step("robot_motors_are")
async def robot_motors_are(
    ctx: ScenarioContext,
    mode: Literal["enabled", "disabled", "gravity_compensation"],
    within_s: float = 10.0,
) -> None:
    """The motors' control mode (the daemon's GET /api/motors/status) is `mode` within
    `within_s`; printed with the time since `edge_agent_crashes`, if it ran."""
    started = time.monotonic()
    while True:
        got = (await _get(f"{_daemon_url(ctx)}/api/motors/status")).get("mode")
        if got == mode:
            break
        if time.monotonic() - started > within_s:
            raise AssertionError(f"motors {got!r} after {within_s} s, want {mode!r}")
        await asyncio.sleep(0.25)
    crashed = ctx.state.get("crashed_at")
    since = f", {time.monotonic() - crashed:.1f} s after the crash" if crashed else ""
    print(f"motors {mode}{since}")


@step("daemon_printed")
async def daemon_printed(
    ctx: ScenarioContext, text: str, count: int = 1, within_s: float = 10.0
) -> None:
    """The daemon (on the robot's machine) printed a line containing `text` (e.g. the motor
    watchdog's `WATCHDOG rested`) at least `count` times, within `within_s`."""
    proc = ctx.processes.get(DAEMON)
    started = time.monotonic()
    while len(found := [line for line in proc.lines if text in line]) < count:
        if time.monotonic() - started > within_s:
            dog = [line for line in proc.lines if line.startswith("WATCHDOG")]
            tail = "\n".join([*proc.lines[-10:], "its WATCHDOG lines:", *dog])
            want = f"{text!r} {len(found)}x (want {count})"
            raise AssertionError(f"the daemon printed {want}:\n{tail}")
        await asyncio.sleep(0.2)
    print(found[count - 1])


@step("daemon_status_is")
async def daemon_status_is(
    ctx: ScenarioContext, state: str = "running", version: str | None = None
) -> None:
    """GET /api/daemon/status: the daemon is in `state`, its backend is ready, no error."""
    status = await _get(f"{_daemon_url(ctx)}/api/daemon/status")
    print(f"daemon status: {json.dumps(status)[:400]}")
    if status.get("state") != state:
        raise AssertionError(f"daemon state {status.get('state')!r}, want {state!r}: {status}")
    if status.get("error"):
        raise AssertionError(f"daemon reports an error: {status['error']}")
    backend = status.get("backend_status") or {}
    if not backend.get("ready"):
        raise AssertionError(f"daemon backend not ready: {backend}")
    if version is not None and status.get("version") != version:
        raise AssertionError(f"daemon version {status.get('version')!r}, want {version!r}")


@step("robot_state_read")
async def robot_state_read(ctx: ScenarioContext, control_mode: str | None = None) -> None:
    """GET /api/state/full: the robot reports a head pose, body yaw and both antennas.

    `control_mode` (e.g. `disabled`) also checks the motor mode, which proves nothing moved.
    """
    state = await _get(f"{_daemon_url(ctx)}/api/state/full")
    print(f"robot state: {json.dumps(state)}")
    pose = state.get("head_pose") or {}
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        if not isinstance(pose.get(key), int | float):
            raise AssertionError(f"head_pose.{key} missing or not a number: {state}")
    if not isinstance(state.get("body_yaw"), int | float):
        raise AssertionError(f"body_yaw missing: {state}")
    antennas = state.get("antennas_position")
    if not (isinstance(antennas, list) and len(antennas) == 2):
        raise AssertionError(f"antennas_position is not two numbers: {state}")
    if control_mode is not None and state.get("control_mode") != control_mode:
        raise AssertionError(f"control_mode {state.get('control_mode')!r}, want {control_mode!r}")


@step("stop_reachy_daemon")
async def stop_reachy_daemon(ctx: ScenarioContext) -> None:
    """Stop the daemon this scenario started (SIGTERM): it shuts down cleanly, stops answering."""
    url = _daemon_url(ctx)
    daemon: ManagedProcess = ctx.processes.get(DAEMON)
    await _rest_before_daemon_stops(host_of(ctx))
    await daemon.stop(grace_s=REMOTE_STOP_GRACE_S)
    if "Daemon stopped successfully" not in daemon.output:
        raise AssertionError(f"daemon did not report a clean stop:\n{daemon.output[-3000:]}")
    try:
        await _get(f"{url}/api/daemon/status", 3)
    except (OSError, urllib.error.URLError):
        return
    raise AssertionError("the daemon API still answers after the daemon was stopped")


@step("edge_host_clean")
async def edge_host_clean(ctx: ScenarioContext) -> None:
    """Stop what this scenario runs on the robot's machine, the daemon last (so the body can
    put the robot to rest through it); the motors must then be disabled, and nothing from
    ~/assistant-edge may still be running there (no orphans left by the SSH launcher). On the
    sim: none of the sim robot's processes (its daemon, the edge agents next to it).

    Motors left enabled are a failure, and the robot is put to rest (SDK `goto_sleep`, torque
    off) before the daemon stops: their torque outlives the daemon and would start the next
    scenario awake."""
    host = host_of(ctx)
    on_robot = sim_robot(ctx).agents if ctx.sim else set[str]()
    procs = [
        proc
        for proc in reversed(ctx.processes.processes)
        if isinstance(proc, RemoteProcess) or proc.name == DAEMON or proc.name in on_robot
    ]
    for proc in procs:
        if proc.name != DAEMON:
            # A reachy edge agent rests the robot as it stops (up to about 10 s): a simulated
            # robot's agent, a local process, gets as long as a remote one.
            await proc.stop(grace_s=REMOTE_STOP_GRACE_S)
    rested = await asyncio.to_thread(eh.rest_robot, host)
    for proc in procs:
        if proc.name == DAEMON:
            await proc.stop()
    # The sim's own processes only: this PC may run other edge-dir processes (other sessions).
    left = await asyncio.to_thread(sim.robot_processes if ctx.sim else lambda: eh.leftovers(host))
    if left:
        raise AssertionError(f"still running on {host.label}: {left}")
    if rested:
        motion = [
            f"{proc.name}: {line}"
            for proc in procs
            for line in proc.lines
            if line.startswith(("MOTION", "BODY-", "STOPPED", "Traceback")) or "attention" in line
        ]
        raise AssertionError(
            f"the scenario left the motors enabled (put to rest: {rested}); the edge's motion "
            "and health lines:\n" + "\n".join(motion[-30:])
        )


# ---------------------------------------------------------------- EdgeLink across machines


FORWARD_FAILED = "remote port forwarding failed"
"""What ssh logs when the edge host refused our `-R` forward (the port was taken)."""
FORWARD_SETTLE_S = 1.5
"""After the port is seen listening: how long our ssh -R gets to report a refused forward."""


async def reverse_tunnel(ctx: ScenarioContext, port: int) -> int:
    """`ssh -R` so 127.0.0.1:<returned port> on the edge host reaches 127.0.0.1:`port` here.

    Both ends stay on loopback (plain ws:// never crosses the LAN). One tunnel per local port.
    """
    tunnels: dict[int, int] = ctx.state.setdefault("reverse_tunnels", {})
    if port in tunnels:
        return tunnels[port]
    host = host_of(ctx)
    if host.ssh is None:
        return port
    # A port from REVERSE_PORTS (not `-R 0`), so a forward a crashed runner left is findable.
    listening = (
        "lsof -nP -iTCP:{p} -sTCP:LISTEN >/dev/null 2>&1 || "
        "ss -ltn 2>/dev/null | grep -q '127.0.0.1:{p} '"
    )
    last_output = ""
    for attempt in range(8):
        remote = random.choice(eh.REVERSE_PORTS)
        tunnel = await ctx.processes.start(
            f"tunnel:link:{port}:{attempt}",
            _ssh_tunnel_argv(host.ssh, "-R", f"{remote}:127.0.0.1:{port}"),
        )
        deadline = time.monotonic() + 20
        while tunnel.running and time.monotonic() < deadline:
            done = await asyncio.to_thread(host.run, listening.format(p=remote), 30)
            if done.returncode == 0:
                # Something listens there, but it may be someone else's listener that took the
                # port first: then OUR ssh -R logs the refusal and exits (ExitOnForwardFailure).
                await asyncio.sleep(FORWARD_SETTLE_S)
                if tunnel.running and FORWARD_FAILED not in tunnel.output:
                    tunnels[port] = remote
                break
            await asyncio.sleep(0.3)
        if port in tunnels:
            break
        await tunnel.stop()  # forward refused (port taken there) or never came up: next port
        last_output = tunnel.output
    else:
        raise AssertionError(
            f"no ssh -R tunnel to {host.label} came up; last output:\n{last_output}"
        )
    print(f"EdgeLink: {host.label} 127.0.0.1:{tunnels[port]} -> here 127.0.0.1:{port} (ssh -R)")
    return tunnels[port]
