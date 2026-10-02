"""Launch real subprocesses for e2e scenarios and always stop them again.

Each process runs in its own session (process group), so stopping it also stops anything it
spawned. `ProcessGroup.stop_all` is called by the feature runner's teardown, whatever happens.

`ProcessGroup.start(..., ssh="alias")` runs the process on another machine (the edge host the
robot is plugged into) through SSH instead: the same object streams its output back, takes
stdin, is signalled (STOP/CONT/KILL reach the remote process group) and is stopped in teardown.
"""

import asyncio
import contextlib
import ctypes
import os
import re
import shlex
import signal
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

RUN_TAG_VAR = "ASSISTANT_TEST_RUN"
RUN_ID = os.environ.get(RUN_TAG_VAR) or uuid.uuid4().hex[:12]
"""Tags every ssh process this test run starts (`-o SetEnv=ASSISTANT_TEST_RUN=<id>`), so a sweep
finds ssh clients and tunnels a crashed runner left behind (`edge_host sweep`)."""
TEST_SSH_PATTERN = rf"^ssh .*SetEnv={RUN_TAG_VAR}="
"""pgrep -f pattern for the ssh processes of any test run."""

_PR_SET_PDEATHSIG = 1
_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def die_with_parent() -> Callable[[], None] | None:
    """`preexec_fn`: on Linux the child gets SIGTERM when the thread that started it dies, so
    ssh clients and tunnels never outlive a killed test runner. None (no-op) elsewhere."""
    if _LIBC is None:
        return None
    libc, parent = _LIBC, os.getpid()

    def setup() -> None:
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() != parent:  # the parent died before prctl took effect
            os.kill(os.getpid(), signal.SIGTERM)

    return setup


def ssh_tag_options() -> list[str]:
    """ssh options that mark a process as started by this test run."""
    return ["-o", f"SetEnv={RUN_TAG_VAR}={RUN_ID}"]


@dataclass(frozen=True, slots=True)
class CompletedRun:
    name: str
    argv: tuple[str, ...]
    returncode: int
    output: str


@dataclass
class ManagedProcess:
    """A running subprocess whose merged stdout/stderr is collected line by line."""

    name: str
    argv: tuple[str, ...]
    proc: asyncio.subprocess.Process
    env: dict[str, str] | None = None
    cwd: Path | None = None
    lines: list[str] = field(default_factory=list)
    _new_line: asyncio.Event = field(default_factory=asyncio.Event)
    _reader: asyncio.Task[None] | None = None

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    @property
    def running(self) -> bool:
        return self.proc.returncode is None

    def start_reader(self) -> None:
        """Start collecting output (called by `ProcessGroup.start`)."""
        self._reader = asyncio.create_task(self._read(), name=f"read:{self.name}")

    async def _read(self) -> None:
        stream = self.proc.stdout
        if stream is None:
            return
        while raw := await stream.readline():
            self.lines.append(raw.decode(errors="replace").rstrip("\n"))
            self._new_line.set()
        self._new_line.set()

    async def wait_for_line(self, pattern: str, timeout_s: float) -> str:
        """Wait until an output line matches the regex `pattern`; return that line."""
        regex = re.compile(pattern)
        seen = 0

        async def scan() -> str:
            nonlocal seen
            while True:
                for line in self.lines[seen:]:
                    seen += 1
                    if regex.search(line):
                        return line
                if not self.running and self._reader is not None and self._reader.done():
                    raise RuntimeError(
                        f"{self.name} exited ({self.proc.returncode}) before printing "
                        f"/{pattern}/; output:\n{self.output}"
                    )
                self._new_line.clear()
                await self._new_line.wait()

        try:
            return await asyncio.wait_for(scan(), timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"{self.name} did not print /{pattern}/ within {timeout_s}s; output:\n{self.output}"
            ) from None

    async def wait_for_output(
        self,
        match: Callable[[str], bool],
        timeout_s: float,
        *,
        skip: Callable[[int], bool] = lambda index: False,
        what: str = "a matching line",
    ) -> tuple[int, str]:
        """Wait for an output line (not rejected by `skip(index)`) that `match` accepts.

        Returns (index, line). Unlike `wait_for_line`, it keeps waiting after the process
        exits only until its output is fully read, then fails with the output so far.
        """

        async def scan() -> tuple[int, str]:
            seen = 0
            while True:
                while seen < len(self.lines):
                    index, line = seen, self.lines[seen]
                    seen += 1
                    if not skip(index) and match(line):
                        return index, line
                if not self.running and self._reader is not None and self._reader.done():
                    raise RuntimeError(
                        f"{self.name} exited ({self.proc.returncode}) before printing "
                        f"{what}; output:\n{self.output}"
                    )
                self._new_line.clear()
                await self._new_line.wait()

        try:
            return await asyncio.wait_for(scan(), timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"{self.name} did not print {what} within {timeout_s}s; output:\n{self.output}"
            ) from None

    async def write_line(self, text: str) -> None:
        """Write one line to the process's stdin (it must have been started with stdin=True)."""
        stream = self.proc.stdin
        if stream is None:
            raise RuntimeError(f"{self.name} was started without stdin")
        if not self.running:
            raise RuntimeError(f"{self.name} is not running; output:\n{self.output}")
        stream.write(text.encode() + b"\n")
        await stream.drain()

    def send_signal(self, sig: signal.Signals) -> None:
        """Send a real signal to the process group (SIGKILL, SIGSTOP, SIGCONT, ...)."""
        os.killpg(self.proc.pid, sig)

    async def wait(self, timeout_s: float) -> int:
        code = await asyncio.wait_for(self.proc.wait(), timeout_s)
        if self._reader is not None:
            await self._reader
        return code

    async def stop(self, grace_s: float = 5.0) -> int | None:
        """SIGTERM the process group, then SIGKILL it if it has not exited after `grace_s`."""
        if self.running:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
                os.killpg(self.proc.pid, signal.SIGCONT)  # a frozen (SIGSTOP) process too
            try:
                await asyncio.wait_for(self.proc.wait(), grace_s)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                await self.proc.wait()
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)  # stray children of an exited leader
        if self._reader is not None:
            await self._reader
        return self.proc.returncode


REMOTE_PGID_MARK = "@@edge-pgid "
"""First line a remote process prints (from the launcher, before exec): its process group."""

_REMOTE_LAUNCHER = (
    "use POSIX (); POSIX::setsid(); $| = 1; "
    'print "' + REMOTE_PGID_MARK.replace("@", "\\@") + '", getpgrp(), "\\n"; '
    'exec { $ARGV[0] } @ARGV or die "exec $ARGV[0]: $!\\n";'
)
"""Perl (present on macOS and Linux): new session, report the process group, exec argv."""


def sh_word(arg: str) -> str:
    """Quote one argument for a POSIX shell; a leading `~/` stays relative to $HOME."""
    if arg == "~":
        return '"$HOME"'
    if arg.startswith("~/"):
        return '"$HOME"/' + shlex.quote(arg[2:])
    return shlex.quote(arg)


def ssh_argv(ssh: str, script: str, *, tty_free: bool = True, tag: bool = True) -> list[str]:
    """argv that runs the POSIX sh `script` on `ssh` (an alias from ~/.ssh/config).

    The remote login shell may be zsh, so the script always runs under /bin/sh. `tag` marks it
    as a test-run process (the sweep's own ssh calls are untagged).
    """
    argv = ["ssh", "-o", "BatchMode=yes", *(ssh_tag_options() if tag else [])]
    if tty_free:
        argv.append("-T")
    return [*argv, ssh, "exec /bin/sh -c " + shlex.quote(script)]


def remote_command(
    argv: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None
) -> str:
    """The sh script that launches `argv` remotely in its own process group."""
    env_words = [f"{k}={sh_word(v)}" for k, v in (env or {}).items()]
    words = " ".join(sh_word(a) for a in argv)
    cd = f"cd {sh_word(cwd)} || exit 97; " if cwd else ""
    launcher = shlex.quote(_REMOTE_LAUNCHER)
    return f"{cd}exec /usr/bin/env {' '.join(env_words)} perl -e {launcher} -- {words}"


@dataclass
class RemoteProcess(ManagedProcess):
    """A process on the edge host, run through `ssh`; `proc` is the local ssh client.

    Signals and stop go to the remote process group (`kill -SIG -- -pgid` over SSH).
    """

    ssh: str = ""
    command: tuple[str, ...] = ()
    remote_cwd: str | None = None
    remote_env: dict[str, str] | None = None
    remote_pgid: int | None = None
    _pgid_known: asyncio.Event = field(default_factory=asyncio.Event)

    async def _read(self) -> None:
        stream = self.proc.stdout
        if stream is None:
            return
        while raw := await stream.readline():
            line = raw.decode(errors="replace").rstrip("\n")
            if self.remote_pgid is None and line.startswith(REMOTE_PGID_MARK):
                self.remote_pgid = int(line.removeprefix(REMOTE_PGID_MARK))
                self._pgid_known.set()
                continue
            self.lines.append(line)
            self._new_line.set()
        self._new_line.set()
        self._pgid_known.set()

    async def wait_started(self, timeout_s: float) -> None:
        """Wait until the remote launcher reported the process group."""
        try:
            await asyncio.wait_for(self._pgid_known.wait(), timeout_s)
        except TimeoutError:
            raise TimeoutError(
                f"{self.name}: no process group from {self.ssh} within {timeout_s}s; "
                f"output:\n{self.output}"
            ) from None
        if self.remote_pgid is None:
            raise RuntimeError(
                f"{self.name}: could not start on {self.ssh} (ssh exit "
                f"{self.proc.returncode}); output:\n{self.output}"
            )

    def _kill_argv(self, *signals: str) -> list[str]:
        pgid = self.remote_pgid
        script = "; ".join(f"kill -{sig} -- -{pgid} 2>/dev/null" for sig in signals) + "; true"
        return ssh_argv(self.ssh, script)

    def send_signal(self, sig: signal.Signals) -> None:
        """Send a real signal to the remote process group (SIGKILL, SIGSTOP, SIGCONT, ...)."""
        if self.remote_pgid is None:
            raise RuntimeError(f"{self.name}: remote process group unknown")
        name = sig.name.removeprefix("SIG")
        done = subprocess.run(
            self._kill_argv(name),
            capture_output=True,
            text=True,
            timeout=30,
            preexec_fn=die_with_parent(),
        )
        if done.returncode != 0:
            raise RuntimeError(f"kill -{name} on {self.ssh} failed: {done.stderr.strip()}")

    async def _remote_kill(self, *signals: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            *self._kill_argv(*signals),
            preexec_fn=die_with_parent(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.wait(), 30)
        except TimeoutError:
            proc.kill()
            await proc.wait()

    async def stop(self, grace_s: float = 10.0) -> int | None:
        """SIGTERM the remote group, SIGKILL it after `grace_s`; then end the ssh client."""
        if self.remote_pgid is not None:
            if self.running:
                await self._remote_kill("TERM", "CONT")
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.proc.wait(), grace_s)
            await self._remote_kill("KILL")  # stragglers of the group, always
        if self.running:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                await self.proc.wait()
        if self._reader is not None:
            await self._reader
        return self.proc.returncode


class ProcessGroup:
    """All processes one scenario started. `stop_all` stops them in reverse start order."""

    def __init__(self, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> None:
        self.cwd = cwd
        self.env = dict(os.environ if env is None else env)
        self.processes: list[ManagedProcess] = []

    async def start(
        self,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        ready_line: str | None = None,
        ready_timeout: float = 30.0,
        stdin: bool = False,
        ssh: str | None = None,
        remote_cwd: str | None = None,
    ) -> ManagedProcess:
        """Start a process. With `stdin=True` steps can type into it (`write_line`).

        With `ssh` (an alias from ~/.ssh/config) it runs on that host instead: `argv`, `env`
        (only these variables) and `remote_cwd` (`~/...` allowed) apply there.
        """
        if ssh is not None:
            return await self._start_remote(
                name, argv, ssh, env, remote_cwd, ready_line, ready_timeout, stdin
            )
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
            cwd=cwd or self.cwd,
            env={**self.env, **(env or {})},
            start_new_session=True,
            preexec_fn=die_with_parent(),
        )
        managed = ManagedProcess(
            name=name,
            argv=tuple(argv),
            proc=proc,
            env=dict(env) if env is not None else None,
            cwd=cwd,
        )
        managed.start_reader()
        self.processes.append(managed)
        if ready_line is not None:
            await managed.wait_for_line(ready_line, ready_timeout)
        return managed

    async def _start_remote(
        self,
        name: str,
        argv: Sequence[str],
        ssh: str,
        env: Mapping[str, str] | None,
        remote_cwd: str | None,
        ready_line: str | None,
        ready_timeout: float,
        stdin: bool,
    ) -> ManagedProcess:
        local = ssh_argv(ssh, remote_command(argv, cwd=remote_cwd, env=env))
        proc = await asyncio.create_subprocess_exec(
            *local,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
            env=self.env,
            start_new_session=True,
            preexec_fn=die_with_parent(),
        )
        managed = RemoteProcess(
            name=name,
            argv=tuple(local),
            proc=proc,
            ssh=ssh,
            command=tuple(argv),
            remote_cwd=remote_cwd,
            remote_env=dict(env) if env is not None else None,
        )
        managed.start_reader()
        self.processes.append(managed)
        await managed.wait_started(30)
        if ready_line is not None:
            await managed.wait_for_line(ready_line, ready_timeout)
        return managed

    async def run(
        self,
        name: str,
        argv: Sequence[str],
        *,
        timeout_s: float = 60.0,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> CompletedRun:
        """Run a process to completion (it is still stopped on teardown if it hangs)."""
        managed = await self.start(name, argv, env=env, cwd=cwd)
        try:
            code = await managed.wait(timeout_s)
        except TimeoutError:
            await managed.stop()
            raise TimeoutError(f"{name} did not finish within {timeout_s}s") from None
        return CompletedRun(name=name, argv=tuple(argv), returncode=code, output=managed.output)

    async def restart(
        self, name: str, *, ready_line: str | None = None, ready_timeout: float = 30.0
    ) -> ManagedProcess:
        """Start a new process with the same argv, env and cwd as the last one named `name`.

        The old one must have exited (e.g. it was killed); `get(name)` returns the new one.
        """
        old = self.get(name)
        if old.running:
            raise RuntimeError(f"{name} is still running (pid {old.pid}); kill or stop it first")
        if isinstance(old, RemoteProcess):
            return await self.start(
                name,
                old.command,
                env=old.remote_env,
                ready_line=ready_line,
                ready_timeout=ready_timeout,
                stdin=old.proc.stdin is not None,
                ssh=old.ssh,
                remote_cwd=old.remote_cwd,
            )
        return await self.start(
            name,
            old.argv,
            env=old.env,
            cwd=old.cwd,
            ready_line=ready_line,
            ready_timeout=ready_timeout,
            stdin=old.proc.stdin is not None,
        )

    def get(self, name: str) -> ManagedProcess:
        for managed in reversed(self.processes):
            if managed.name == name:
                return managed
        raise KeyError(f"no process named {name!r} was started in this scenario")

    async def stop_all(self) -> None:
        errors: list[BaseException] = []
        for managed in reversed(self.processes):
            try:
                await managed.stop()
            except BaseException as exc:  # keep stopping the others
                errors.append(exc)
        if errors:
            raise errors[0]
