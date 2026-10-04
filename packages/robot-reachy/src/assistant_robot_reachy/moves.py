"""MovementManager: Pollen's conversation-app motion loop (moves.py, Apache-2.0), ported.

One worker thread owns the robot's pose while it is awake: about 60 times a second it samples
the current primary move (or holds the last pose) and calls `ReachyMini.set_target` once.
Moves run one after the other from a queue: attention poses (`GotoMove`, ours: small and
slow) and Pollen's recorded emotion moves (`RecordedMove`, each reached from the present pose
by a short `GotoMove` to its first frame, like the SDK's `play_move`). Speech-driven head
wobble is the SDK's own (`enable_wobbling`: offsets from the played audio, composed with this
target on the daemon side), on while the manager runs.

Kept from Pollen's manager: the sequential queue, the antenna freeze while listening (blended
back over 0.4 s), the face-tracking speaking handoff (tracking pauses while the assistant
speaks, the look-at pose anchoring queued moves; tracking itself is off by default, as in
Pollen's app). Left out: the idle breathing move (the user's choice: the head holds still
between moves) and dances.

Safety, on top of Pollen's: each tick first asks `alive()` (the body's link heartbeat is
fresh); when it is not, the loop stops commanding at once, so the motor watchdog next to the
daemon rests the robot unopposed. A `set_target` failing for longer than `FAIL_STOP_S` stops
the loop and calls `on_failure` (the arbiter then sleeps the robot with its motors off); so
does a move that raises.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

CONTROL_HZ = 60.0
ANTENNA_BLEND_S = 0.4
FAIL_STOP_S = 1.0
"""`set_target` failing this long (the daemon gone, the robot unplugged): stop and rest."""

END_EPSILON_S = 1e-3
"""A recorded move's last pose is evaluated this much before its end (the SDK's last frame)."""

Pose = tuple[np.ndarray, tuple[float, float], float]
"""(head 4x4, (left, right) antennas in radians, body yaw in radians)."""


class Move(Protocol):
    @property
    def duration(self) -> float: ...

    def evaluate(self, t: float) -> Pose: ...


def _pose(head: Any, antennas: Any, body_yaw: Any) -> Pose:
    head = np.eye(4) if head is None else np.array(head, dtype=float)
    pair = (0.0, 0.0) if antennas is None else (float(antennas[0]), float(antennas[1]))
    return head, pair, 0.0 if body_yaw is None else float(body_yaw)


@dataclass
class GotoMove:
    """From `start` to `target` over `duration` (linear in time, the SDK's pose interpolation)."""

    start: Pose
    target: Pose
    seconds: float
    label: str = "goto"

    @property
    def duration(self) -> float:
        return self.seconds

    def evaluate(self, t: float) -> Pose:
        from reachy_mini.utils.interpolation import linear_pose_interpolation

        s = 1.0 if self.seconds <= 0 else max(0.0, min(1.0, t / self.seconds))
        head = linear_pose_interpolation(self.start[0], self.target[0], s)
        a0, a1 = self.start[1], self.target[1]
        antennas = (a0[0] + (a1[0] - a0[0]) * s, a0[1] + (a1[1] - a0[1]) * s)
        yaw = self.start[2] + (self.target[2] - self.start[2]) * s
        return np.array(head, dtype=float), antennas, yaw


class RecordedMove:
    """One of Pollen's recorded emotion moves (`RecordedMoves.get(name)`)."""

    def __init__(self, name: str, move: Any) -> None:
        self.label, self.move = name, move

    @property
    def duration(self) -> float:
        return float(self.move.duration)

    def evaluate(self, t: float) -> Pose:
        # The SDK raises at or past the last recorded frame: its end is just before it.
        return _pose(*self.move.evaluate(max(0.0, min(t, self.duration - END_EPSILON_S))))


@dataclass
class Played:
    """A move's run on the manager's clock (this machine's monotonic clock)."""

    label: str
    started: float | None = None
    ended: float | None = None
    cancelled: bool = False
    done: threading.Event = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.done = threading.Event()


class MovementManager:
    """The 60 Hz loop of one awake robot (`mini` is a connected `reachy_mini.ReachyMini`)."""

    def __init__(
        self,
        mini: Any,
        *,
        alive: Callable[[], bool] = lambda: True,
        on_failure: Callable[[str], None] | None = None,
        hz: float = CONTROL_HZ,
    ) -> None:
        self.mini = mini
        self.alive = alive
        self.on_failure = on_failure
        self.period = 1.0 / hz
        self._lock = threading.Lock()
        self._queue: deque[tuple[Move, Played]] = deque()
        self._current: tuple[Move, Played] | None = None
        self._pose: Pose = (np.eye(4), (0.0, 0.0), 0.0)
        self._commanded: Pose = self._pose
        self._listening = False
        self._frozen: tuple[float, float] = (0.0, 0.0)
        self._blend = 1.0
        self._blend_t = time.monotonic()
        self._tracking = False
        self._speaking = False
        self._anchor: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stopped_reason: str | None = None
        self.wobbling = False

    # ------------------------------------------------------------ lifecycle

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, pose: Pose, *, wobble: bool = True) -> None:
        """Start commanding from `pose` (the robot's present pose: no jump)."""
        if self.running:
            return
        with self._lock:
            self._pose = self._commanded = (pose[0].copy(), pose[1], pose[2])
            self._frozen = pose[1]
            self._blend, self._blend_t = 1.0, time.monotonic()
        self.stopped_reason = None
        self._stop.clear()
        if wobble:
            try:
                self.mini.enable_wobbling()
                self.wobbling = True
            except Exception as exc:
                log.warning("enable_wobbling failed: %s", exc)
        self._thread = threading.Thread(target=self._loop, name="reachy-moves", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop (moves cancelled, the pose held where it is), wobble and tracking off."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self.clear()
        if self.wobbling:
            self.wobbling = False
            try:
                self.mini.disable_wobbling()
            except Exception as exc:
                log.warning("disable_wobbling failed: %s", exc)
        if self._tracking:
            self._tracking = False
            try:
                self.mini.stop_head_tracking()
            except Exception as exc:
                log.warning("stop_head_tracking failed: %s", exc)

    # ------------------------------------------------------------ commands (any thread)

    def queue(self, move: Move) -> Played:
        played = Played(getattr(move, "label", type(move).__name__))
        with self._lock:
            self._queue.append((move, played))
        return played

    def clear(self) -> None:
        """Cancel the current move and every queued one (the pose stays where it is)."""
        with self._lock:
            pending = list(self._queue)
            self._queue.clear()
            if self._current is not None:
                pending.insert(0, self._current)
                self._current = None
        for _, played in pending:
            played.cancelled = True
            played.done.set()

    def queued_s(self) -> float:
        """How long the current and queued moves take altogether (at most)."""
        with self._lock:
            moves = [m for m, _ in self._queue] + ([self._current[0]] if self._current else [])
            return sum(m.duration for m in moves)

    def last_pose(self) -> Pose:
        """Where the queue ends: the last queued move's target, else the present pose."""
        with self._lock:
            moves = [m for m, _ in self._queue] or ([self._current[0]] if self._current else [])
            if moves:
                last = moves[-1]
                return last.evaluate(last.duration)
            return self._pose

    def set_listening(self, listening: bool) -> None:
        """Listening freezes the antennas where they are; they blend back afterwards."""
        with self._lock:
            if self._listening == listening:
                return
            self._listening = listening
            if listening:
                self._frozen = self._commanded[1]
            self._blend, self._blend_t = 0.0, time.monotonic()

    def set_head_tracking(self, enabled: bool) -> None:
        if self._tracking == enabled:
            return
        self._tracking, self._anchor, self._speaking = enabled, None, False
        try:
            if enabled:
                self.mini.start_head_tracking(weight=1.0)
            else:
                self.mini.stop_head_tracking()
        except Exception as exc:
            log.warning("head tracking toggle failed: %s", exc)

    def set_speaking(self, speaking: bool) -> None:
        """Tracking pauses while the assistant speaks, the look-at pose anchoring moves."""
        if not self._tracking or self._speaking == speaking:
            return
        self._speaking = speaking
        try:
            if speaking and self.mini.get_tracked_face(wait=False).detected:
                self._anchor = np.array(self.mini.get_current_head_pose(), dtype=float)
                self.mini.start_head_tracking(weight=0.0)
            elif not speaking:
                self._anchor = None
                self.mini.start_head_tracking(weight=1.0)
        except Exception as exc:
            log.warning("head tracking speaking handoff failed: %s", exc)

    # ------------------------------------------------------------ the loop

    def _next_pose(self, now: float) -> Pose:
        with self._lock:
            current = self._current
            if current is not None and now - (current[1].started or now) >= current[0].duration:
                self._pose = current[0].evaluate(current[0].duration)
                current[1].ended = now
                current[1].done.set()
                current = self._current = None
            if current is None and self._queue:
                current = self._current = self._queue.popleft()
                current[1].started = now
            if current is not None:
                move, played = current
                self._pose = move.evaluate(now - (played.started or now))
            head, antennas, yaw = self._pose
            if self._anchor is not None:
                from reachy_mini.utils.interpolation import compose_world_offset

                if current is None:
                    head = self._anchor.copy()
                elif isinstance(current[0], RecordedMove):
                    head = compose_world_offset(self._anchor, head)
            if self._listening:
                antennas = self._frozen
            elif self._blend < 1.0:
                self._blend = min(1.0, self._blend + (now - self._blend_t) / ANTENNA_BLEND_S)
                f = self._blend
                antennas = (
                    self._frozen[0] * (1 - f) + antennas[0] * f,
                    self._frozen[1] * (1 - f) + antennas[1] * f,
                )
            self._blend_t = now
            return head, antennas, yaw

    def _loop(self) -> None:
        failing_since: float | None = None
        while not self._stop.is_set():
            tick = time.monotonic()
            if not self.alive():
                self.stopped_reason = "link down"
                log.warning("movement loop stopped: the link heartbeat is not fresh")
                self.clear()
                return
            try:
                head, antennas, yaw = self._next_pose(tick)
            except Exception as exc:  # a move that cannot be evaluated: stop, rest
                log.exception("movement loop: a move failed")
                self.stopped_reason = f"move failed: {type(exc).__name__}: {exc}"
                self.clear()
                if self.on_failure is not None:
                    self.on_failure(self.stopped_reason)
                return
            try:
                self.mini.set_target(head=head, antennas=list(antennas), body_yaw=yaw)
            except Exception as exc:
                failing_since = failing_since or tick
                if tick - failing_since >= FAIL_STOP_S:
                    self.stopped_reason = f"set_target failing: {type(exc).__name__}: {exc}"
                    self.clear()
                    if self.on_failure is not None:
                        self.on_failure(self.stopped_reason)
                    return
            else:
                failing_since = None
                with self._lock:
                    self._commanded = (head, antennas, yaw)
            self._stop.wait(max(0.0, self.period - (time.monotonic() - tick)))
