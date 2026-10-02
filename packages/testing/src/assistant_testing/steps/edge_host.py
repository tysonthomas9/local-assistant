"""Edge-host steps: the real robot on whichever machine it is plugged into.

The host comes from `assistant_testing.edge_host.resolve`: this machine when the robot is
attached here, else the SSH alias from config `[test.edge_host] ssh` / ASSISTANT_EDGE_HOST.
Nothing here is simulated: the bootstrap script, `git push`, `uv sync`, the reachy-mini daemon
and the SSH tunnels are all real, and every process is stopped in teardown.

The daemon's HTTP API stays on 127.0.0.1 on the edge host; the PC reaches it through an
`ssh -L` tunnel. EdgeLink stays on 127.0.0.1 on the PC; an edge-host process reaches it through
an `ssh -R` tunnel (S7 replaces the tunnel with TLS and pairing).
"""

import asyncio
import json
import random
import re
import shlex
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from assistant_testing import edge_host as eh
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.processes import (
    ManagedProcess,
    RemoteProcess,
    home_scrubber,
    ssh_tag_options,
)

DAEMON = "daemon"
DAEMON_PORT = 8000
DAEMON_ARGS = (
    "--no-media",  # S3 brings up media; meanwhile no WebRTC signalling port is opened
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

    Fails (never skips) when neither has the robot's USB serial device.
    """
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


@step("edge_host_bootstrapped")
async def edge_host_bootstrapped(ctx: ScenarioContext, reachy_mini: str) -> None:
    """Run scripts/edge_host_bootstrap.sh there (idempotent: installs once, then validates).

    Checks the pinned reachy-mini version and that the script saw the robot.
    """
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


@step("code_synced_to_edge_host")
async def code_synced_to_edge_host(ctx: ScenarioContext) -> None:
    """Ship the commit under test (HEAD of this checkout) and `uv sync` the edge packages there.

    `git push` to the edge host's bare repo over SSH (never GitHub), check it out in
    ~/assistant-edge/src, then `uv sync --locked` only the edge/robot packages into
    .venv-assistant and import them from that env.
    """
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


@step("start_reachy_daemon")
async def start_reachy_daemon(ctx: ScenarioContext, ready_within_s: float = 90.0) -> None:
    """Start the real reachy-mini daemon on the robot's machine, API on 127.0.0.1 only.

    Fails if a daemon this scenario did not start already answers there (the test stops only
    what it started). No motion: the robot is not woken up and not put to sleep.
    """
    host = host_of(ctx)
    probe = f"curl -s -m 3 -o /dev/null http://127.0.0.1:{DAEMON_PORT}/api/daemon/status"
    if (await asyncio.to_thread(host.run, probe, 30)).returncode == 0:
        raise AssertionError(
            f"a reachy-mini daemon is already answering on {host.label} port {DAEMON_PORT}; "
            "stop it first (this test starts and stops its own)"
        )
    argv = [eh.DAEMON_BIN, *DAEMON_ARGS]
    ready = rf"Uvicorn running on http://127\.0\.0\.1:{DAEMON_PORT}"
    if host.ssh is None:
        await ctx.processes.start(
            DAEMON,
            [str(Path.home() / eh.DAEMON_BIN.removeprefix("~/")), *DAEMON_ARGS],
            env={k: str(Path.home() / v[2:]) if v.startswith("~/") else v
                 for k, v in DAEMON_ENV.items()},
            ready_line=ready,
            ready_timeout=ready_within_s,
        )  # fmt: skip
        ctx.state["daemon_url"] = f"http://127.0.0.1:{DAEMON_PORT}"
        return
    await ctx.processes.start(
        DAEMON,
        argv,
        env=DAEMON_ENV,
        ssh=host.ssh,
        remote_cwd=eh.EDGE_DIR,
        ready_line=ready,
        ready_timeout=ready_within_s,
    )
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


def _daemon_url(ctx: ScenarioContext) -> str:
    url = ctx.state.get("daemon_url")
    if url is None:
        raise AssertionError("no daemon started; use start_reachy_daemon first")
    return url


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
    await daemon.stop(grace_s=20)
    if "Daemon stopped successfully" not in daemon.output:
        raise AssertionError(f"daemon did not report a clean stop:\n{daemon.output[-3000:]}")
    try:
        await _get(f"{url}/api/daemon/status", 3)
    except (OSError, urllib.error.URLError):
        return
    raise AssertionError("the daemon API still answers after the daemon was stopped")


@step("edge_host_clean")
async def edge_host_clean(ctx: ScenarioContext) -> None:
    """Stop what this scenario runs on the robot's machine; then nothing from ~/assistant-edge
    may still be running there (no orphans left by the SSH launcher)."""
    host = host_of(ctx)
    for proc in ctx.processes.processes:
        if isinstance(proc, RemoteProcess) or proc.name == DAEMON:
            await proc.stop()
    left = await asyncio.to_thread(eh.leftovers, host)
    if left:
        raise AssertionError(f"still running on {host.label}: {left}")


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
