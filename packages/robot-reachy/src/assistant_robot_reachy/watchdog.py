"""The motor watchdog: next to the daemon, it rests the robot when no edge agent looks after it.

The reachy body of the edge agent touches `HEARTBEAT` every `HEARTBEAT_EVERY_S` while its link
to the brain is up (`ReachyBody.link_changed`), and removes it when it stops. The watchdog
runs in the daemon's process (`python -m assistant_robot_reachy.daemon`), not in the edge
agent that might die. It arms on the first heartbeat written after it started. Armed, once the
heartbeat is older than `STALE_S` or gone (the edge agent died, froze or stopped, or its link
or the SSH tunnel under it dropped), it disarms, and if the motors are on it puts the robot to
rest the way the tests do: the SDK's `goto_sleep` through the daemon's own REST API, then the
motors off. A fresh heartbeat (the agent back, its link up again) arms it again.

It prints one line per event: `WATCHDOG armed`, `WATCHDOG fired reason=... motors=...` and
`WATCHDOG rested motors=...` (or `WATCHDOG error ...`).
"""

import json
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

HEARTBEAT = Path.home() / ".cache" / "assistant-reachy" / "edge-link.alive"
"""Touched by the edge agent's reachy body while its link is up (same machine, same user)."""
HEARTBEAT_EVERY_S = 0.5
STALE_S = 3.0
"""A heartbeat this old means nobody looks after the robot: rest it."""
POLL_S = 0.5
SLEEP_WAIT_S = 15.0
"""How long `goto_sleep` may run before the motors are turned off anyway."""


def emit(tag: str, *words: str, **fields: object) -> None:
    parts = [tag, *words, *(f"{k}={v}" for k, v in fields.items())]
    sys.stdout.write(" ".join(parts) + "\n")
    sys.stdout.flush()


def touch(path: Path = HEARTBEAT) -> None:
    """One heartbeat (the edge side)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def clear(path: Path = HEARTBEAT) -> None:
    """No more heartbeats (the edge side, when it stops)."""
    path.unlink(missing_ok=True)


class DaemonApi:
    """The daemon's REST API on this machine: the motors' mode and the robot's rest."""

    def __init__(self, base: str) -> None:
        self.base = base

    def _call(self, path: str, method: str = "GET", timeout_s: float = 5.0) -> Any:
        request = urllib.request.Request(f"{self.base}{path}", method=method)
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read() or b"null")

    def motors(self) -> str:
        """The motors' control mode: enabled, disabled or gravity_compensation."""
        return str(self._call("/api/motors/status").get("mode"))

    def rest(self) -> str:
        """`goto_sleep`, wait for it to end, then the motors off; the motors' mode after."""
        self._call("/api/move/play/goto_sleep", "POST")
        deadline = time.monotonic() + SLEEP_WAIT_S
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if self._call("/api/move/running") == []:
                break
        self._call("/api/motors/set_mode/disabled", "POST")
        return self.motors()


class Watchdog:
    """The decision, one `check` at a time (`run` loops it in a thread)."""

    def __init__(
        self,
        motors: Callable[[], str],
        rest: Callable[[], str],
        path: Path = HEARTBEAT,
        stale_s: float = STALE_S,
        started: float | None = None,
    ) -> None:
        self.motors, self.rest, self.path, self.stale_s = motors, rest, path, stale_s
        self.started = time.time() if started is None else started
        self.armed = False
        self.seen = 0.0

    def _mtime(self) -> float | None:
        try:
            return self.path.stat().st_mtime
        except FileNotFoundError:
            return None

    def check(self, now: float | None = None) -> str | None:
        """What was done (None: nothing). `now` is wall-clock time, like the file's mtime."""
        now = time.time() if now is None else now
        beat = self._mtime()
        if beat is not None and beat > max(self.started, self.seen):
            self.seen = beat
            if not self.armed:
                self.armed = True
                emit("WATCHDOG", "armed")
                return "armed"
        if not self.armed:
            return None
        if beat is not None and now - beat <= self.stale_s:
            return None
        reason = "gone" if beat is None else f"stale_{now - beat:.1f}s"
        mode = self.motors()  # if the daemon does not answer: still armed, the next check retries
        self.armed = False
        emit("WATCHDOG", "fired", reason=reason, motors=mode)
        if mode == "disabled":
            return "fired: already at rest"
        try:
            after = self.rest()
        except Exception:
            self.armed = True  # not rested: the next check tries again
            raise
        emit("WATCHDOG", "rested", motors=after)
        return f"rested: motors {after}"

    def run(self, stop: threading.Event, poll_s: float = POLL_S) -> None:
        last = ""
        while not stop.wait(poll_s):
            try:
                self.check()
            except Exception as exc:  # the daemon busy or gone: try again (said once)
                detail = f"{type(exc).__name__}: {exc}"
                if detail != last:
                    emit("WATCHDOG", "error", detail=json.dumps(detail))
                last = detail
            else:
                last = ""


def start(api_base: str, path: Path = HEARTBEAT) -> threading.Event:
    """Run the watchdog in a daemon thread; set the returned event to stop it."""
    api = DaemonApi(api_base)
    dog = Watchdog(api.motors, api.rest, path)
    stop = threading.Event()
    threading.Thread(target=dog.run, args=(stop,), name="motor-watchdog", daemon=True).start()
    return stop
