"""Launch real subprocesses for e2e scenarios and always stop them again.

Each process runs in its own session (process group), so stopping it also stops anything it
spawned. `ProcessGroup.stop_all` is called by the feature runner's teardown, whatever happens.
"""

import asyncio
import contextlib
import os
import re
import signal
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


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
    ) -> ManagedProcess:
        """Start a process. With `stdin=True` steps can type into it (`write_line`)."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
            cwd=cwd or self.cwd,
            env={**self.env, **(env or {})},
            start_new_session=True,
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
