"""The edge host: the machine the robot is plugged into, when that is not this one.

The brain, the models and the test runner stay on this PC; the device side (the reachy-mini
daemon and, from S3, the edge agent) runs where the robot's USB cable is. The tests reach that
machine only by an SSH alias (config `[test.edge_host] ssh`, or the env var ASSISTANT_EDGE_HOST)
defined in ~/.ssh/config, so no address or user name is ever in the repo.

    python -m assistant_testing.edge_host check   # which host has the robot; exit 1 if none
    python -m assistant_testing.edge_host sweep   # stop leftovers: edge-dir processes, test ssh
                                                  # clients/tunnels here, their forwards there

A robot attached to this machine is always used first. Everything on the edge host lives in
`~/assistant-edge/` (see scripts/edge_host_bootstrap.sh). EdgeLink and the daemon API stay on
loopback on both machines; they are joined by SSH tunnels (`-L`/`-R`) until S7 adds TLS.
"""

import glob
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from assistant_core.config import load_config
from assistant_testing.processes import TEST_SSH_PATTERN, die_with_parent, ssh_argv

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
REVERSE_PORTS = range(47000, 48000)
"""Edge-host loopback ports for `ssh -R` tunnels (EdgeLink). A listener in this range owned by
sshd is a test tunnel, so the sweep can find one a crashed runner left behind."""


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

    def sh(self, script: str, *, tag: bool = True) -> list[str]:
        """argv that runs the POSIX sh `script` on the edge host (ssh tagged as a test run)."""
        if self.ssh is None:
            return ["/bin/sh", "-c", script]
        return ssh_argv(self.ssh, script, tag=tag)

    def run(
        self, script: str, timeout_s: float = 60.0, *, tag: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run `script` there and return its result (stdout and stderr captured)."""
        return subprocess.run(
            self.sh(script, tag=tag),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            preexec_fn=die_with_parent(),
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


def _local_test_ssh() -> list[str]:
    """ssh clients and tunnels of any test run still alive on this machine ("pid argv")."""
    done = subprocess.run(
        ["pgrep", "-af", TEST_SSH_PATTERN], capture_output=True, text=True, check=False
    )
    return [line for line in done.stdout.splitlines() if line.strip()]


_FORWARDS = (
    'lsof -nP -a -u "$(id -un)" -c sshd -iTCP:{lo}-{hi} -sTCP:LISTEN 2>/dev/null'
    " | awk 'NR > 1 {{print $2, $1, $9}}' | sort -u"
)


def _edge_forwards(host: EdgeHost) -> list[str]:
    """sshd listeners for test `ssh -R` tunnels on the edge host ("pid sshd addr:port")."""
    if host.ssh is None:
        return []
    lo, hi = REVERSE_PORTS.start, REVERSE_PORTS.stop - 1
    done = host.run(_FORWARDS.format(lo=lo, hi=hi), timeout_s=30, tag=False)
    return [line for line in done.stdout.splitlines() if line.strip()]


def _kill_local(lines: list[str]) -> None:
    pids = [int(line.split()[0]) for line in lines]
    for sig in ("TERM", "KILL"):
        alive = [pid for pid in pids if _alive(pid)]
        if not alive:
            return
        subprocess.run(["kill", f"-{sig}", *map(str, alive)], capture_output=True, check=False)
        time.sleep(2)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep(host: EdgeHost) -> list[str]:
    """Stop and return every leftover of a test run, wherever it is:

    - processes still running from the edge dir on the edge host;
    - tagged ssh clients and tunnels on this machine (`-o SetEnv=ASSISTANT_TEST_RUN=...`);
    - sshd listeners for test `ssh -R` tunnels on the edge host (REVERSE_PORTS).

    Its own ssh calls are untagged, so it never finds itself.
    """
    found = [f"edge forward: {line}" for line in _edge_forwards(host)]
    local = _local_test_ssh()
    found += [f"this machine: {line}" for line in local]
    _kill_local(local)
    pattern = shlex.quote(_SWEEP_PATTERN)
    script = (
        f"found=$(pgrep -fl {pattern} || true); "
        f'if [ -n "$found" ]; then printf "%s\\n" "$found"; '
        f"pkill -TERM -f {pattern}; sleep 3; pkill -KILL -f {pattern}; fi; true"
    )
    done = host.run(script, timeout_s=60, tag=False)
    found += [f"edge process: {line}" for line in done.stdout.splitlines() if line.strip()]
    # The forwards close with their ssh client; kill any sshd that still holds one.
    deadline = time.monotonic() + 10
    while (forwards := _edge_forwards(host)) and time.monotonic() < deadline:
        time.sleep(1)
    if forwards:
        pids = " ".join(sorted({line.split()[0] for line in forwards}))
        host.run(f"kill -TERM {pids} 2>/dev/null; sleep 2; kill -KILL {pids} 2>/dev/null; true",
                 timeout_s=30, tag=False)  # fmt: skip
        found += [f"edge forward (killed): {line}" for line in forwards]
    return found


def leftovers(host: EdgeHost) -> list[str]:
    """Processes still running from the edge dir (none after a clean teardown)."""
    done = host.run(f"pgrep -fl {shlex.quote(_SWEEP_PATTERN)} || true", timeout_s=30, tag=False)
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
            print(f"stopped leftover ({host.label}): {line}")
        print(f"sweep: {len(found)} leftover process(es) (this machine and {host.label})")
        return 0
    print("usage: python -m assistant_testing.edge_host check|sweep", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
