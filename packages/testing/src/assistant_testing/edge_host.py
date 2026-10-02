"""The edge host: the machine the robot is plugged into, when that is not this one.

The brain, the models and the test runner stay on this PC; the device side (the reachy-mini
daemon and, from S3, the edge agent) runs where the robot's USB cable is. The tests reach that
machine only by an SSH alias (config `[test.edge_host] ssh`, or the env var ASSISTANT_EDGE_HOST)
defined in ~/.ssh/config, so no address or user name is ever in the repo.

    python -m assistant_testing.edge_host check   # which host has the robot; exit 1 if none
    python -m assistant_testing.edge_host sweep   # stop anything left running from the edge dir

A robot attached to this machine is always used first. Everything on the edge host lives in
`~/assistant-edge/` (see scripts/edge_host_bootstrap.sh). EdgeLink and the daemon API stay on
loopback on both machines; they are joined by SSH tunnels (`-L`/`-R`) until S7 adds TLS.
"""

import glob
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from assistant_core.config import load_config
from assistant_testing.processes import ssh_argv

ENV_VAR = "ASSISTANT_EDGE_HOST"
EDGE_DIR = "~/assistant-edge"
"""The edge host's working dir (`~` = the edge host's home)."""
REMOTE_VENV = "~/assistant-edge/src/.venv-assistant"
"""The synced checkout's env (UV_PROJECT_ENVIRONMENT=.venv-assistant)."""
DAEMON_BIN = "~/assistant-edge/daemon/bin/reachy-mini-daemon"
SYNC_REF = "refs/heads/under-test"
EDGE_PACKAGES = ("assistant-edge", "assistant-robot-reachy")
"""What `uv sync` installs on the edge host: the edge/robot packages and their deps only."""
ROBOT_GLOBS = ("/dev/ttyACM*", "/dev/cu.usbmodem*")
"""Serial devices of a USB-attached Reachy Mini (Linux, macOS)."""
_SWEEP_PATTERN = "[/]assistant-edge/(daemon|src)/"
"""pgrep/pkill -f pattern (an extended regex on macOS and Linux) for processes run from the edge
dir; `[/]` keeps it from matching the shell that runs it."""


@dataclass(frozen=True)
class EdgeHost:
    """Where the robot is: `ssh` is the alias of the edge host, or None for this machine."""

    ssh: str | None

    @property
    def remote(self) -> bool:
        return self.ssh is not None

    @property
    def label(self) -> str:
        return f"edge host {self.ssh!r} (over SSH)" if self.ssh else "this machine"

    def sh(self, script: str) -> list[str]:
        """argv that runs the POSIX sh `script` on the edge host."""
        if self.ssh is None:
            return ["/bin/sh", "-c", script]
        return ssh_argv(self.ssh, script)

    def run(self, script: str, timeout_s: float = 60.0) -> subprocess.CompletedProcess[str]:
        """Run `script` there and return its result (stdout and stderr captured)."""
        return subprocess.run(
            self.sh(script), capture_output=True, text=True, timeout=timeout_s, check=False
        )


def local_robot() -> str | None:
    """The serial device of a robot attached to this machine, if any."""
    for pattern in ROBOT_GLOBS:
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def configured_alias(repo_root: Path, environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    alias = env.get(ENV_VAR, "").strip()
    if alias:
        return alias
    return load_config(repo_root / "config", environ={}).test.edge_host.ssh.strip()


def resolve(repo_root: Path, environ: Mapping[str, str] | None = None) -> EdgeHost:
    """This machine if a robot is attached here, else the configured edge host (if any)."""
    if local_robot() is not None:
        return EdgeHost(None)
    alias = configured_alias(repo_root, environ)
    return EdgeHost(alias or None)


ROBOT_PROBE = (
    "for d in " + " ".join(ROBOT_GLOBS) + '; do if [ -e "$d" ]; then echo "$d"; exit 0; fi; done; '
    "exit 1"
)


def robot_device(host: EdgeHost) -> str | None:
    """The robot's serial device on `host` (None: no robot there, or SSH failed)."""
    if not host.remote:
        return local_robot()
    done = host.run(ROBOT_PROBE, timeout_s=30)
    return done.stdout.strip() or None if done.returncode == 0 else None


def sweep(host: EdgeHost) -> list[str]:
    """Stop every process still running from the edge dir; return what was found."""
    pattern = shlex.quote(_SWEEP_PATTERN)
    script = (
        f"found=$(pgrep -fl {pattern} || true); "
        f'if [ -n "$found" ]; then printf "%s\\n" "$found"; '
        f"pkill -TERM -f {pattern}; sleep 3; pkill -KILL -f {pattern}; fi; true"
    )
    done = host.run(script, timeout_s=60)
    return [line for line in done.stdout.splitlines() if line.strip()]


def leftovers(host: EdgeHost) -> list[str]:
    """Processes still running from the edge dir (none after a clean teardown)."""
    done = host.run(f"pgrep -fl {shlex.quote(_SWEEP_PATTERN)} || true", timeout_s=30)
    return [line for line in done.stdout.splitlines() if line.strip()]


def _check(repo_root: Path) -> int:
    device = local_robot()
    if device is not None:
        print(f"robot: {device} on this machine (used before any edge host)")
        return 0
    alias = configured_alias(repo_root)
    if not alias:
        print(
            "robot: none on this machine ({}), and no edge host configured "
            "([test.edge_host] ssh or {})".format(", ".join(ROBOT_GLOBS), ENV_VAR)
        )
        return 1
    host = EdgeHost(alias)
    reach = host.run("true", timeout_s=30)
    if reach.returncode != 0:
        print(
            f"robot: none on this machine; edge host {alias!r} unreachable over SSH: "
            f"{reach.stderr.strip()}"
        )
        return 1
    device = robot_device(host)
    if device is None:
        print(
            f"robot: none on this machine; edge host {alias!r} reachable but has no "
            f"{' or '.join(ROBOT_GLOBS)}"
        )
        return 1
    print(f"robot: {device} on edge host {alias!r} (over SSH)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    repo_root = Path.cwd()
    if args[:1] == ["check"]:
        return _check(repo_root)
    if args[:1] == ["sweep"]:
        host = resolve(repo_root)
        found = sweep(host)
        for line in found:
            print(f"stopped leftover on {host.label}: {line}")
        print(f"sweep: {len(found)} leftover process(es) on {host.label}")
        return 0
    print("usage: python -m assistant_testing.edge_host check|sweep", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
