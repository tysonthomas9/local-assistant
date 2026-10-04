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
speaks, the look-at pose anchoring queued moves). Left out: the idle breathing move (the
user's choice: the head holds still between moves) and dances.

Person tracking (S8g). Pollen's daemon-side face tracker (YuNet; `Tracker`: enable with a
weight, disable, is a face seen) owns the head while it follows a face, within Pollen's own
limits. Under it, every move of ours is composed onto a "look-at base" (`compose_world_offset`,
as Pollen's app anchors queued moves at the tracked pose), so attention poses and recorded
moves play on top of where the robot looks instead of fighting it:

- speaking pauses tracking (weight 0) at the captured look-at pose (`hold_tracked`: while
  Pollen's tracker steers the head, the base becomes the measured head pose, so nothing moves;
  with no face seen it captures nothing, so servo lag is never frozen into the base) and
  resumes it (weight 1) afterwards;
- `HOLD_AFTER_S` without a face, the head holds where it is (the base is captured and the
  tracker re-armed, before Pollen's tracker would recenter the head at 2 s);
- a voice turn (`turn_toward`) glides the base to at most `MAX_TURN_DEG` of yaw toward the
  voice, slowly (`TURN_DEG_PER_S`), until the tracker finds a face there;
- tracking off holds the pose, then glides the base back to neutral at `TURN_DEG_PER_S`;
  `bake_base` folds the base into the held pose (before a move back to neutral).

The monitor thread (`TRACK_POLL_S`) reports each tracking state change through `on_event`
(`TRACKING state=on|paused|hold|off`).

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

TRACK_POLL_S = 0.1
"""How often the tracking monitor asks the daemon whether a face is seen."""
HOLD_AFTER_S = 1.5
POLLEN_LOST_S = 2.0  # Pollen's tracker keeps aiming this long after the face was last seen
"""No face this long: hold the pose. Pollen's tracker recenters the head 2 s after it last saw
a face; holding (and re-arming it) a little before keeps that recentering from starting."""
MAX_TURN_DEG = 10.0
"""A voice turn is ours: at most this much yaw from neutral (the safety rule of our moves)."""
TURN_DEG_PER_S = 10.0
"""How fast our glides of the look-at base go (the attention poses' speed: 10 degrees in 1 s)."""
MIN_GLIDE_S = 1.0

Pose = tuple[np.ndarray, tuple[float, float], float]
"""(head 4x4, (left, right) antennas in radians, body yaw in radians)."""


class Tracker(Protocol):
    """Pollen's daemon-side face tracker (`start_head_tracking(weight)` and friends)."""

    def enable(self, weight: float) -> bool:
        """Track with `weight` (0 pauses it); whether the daemon could (it has a camera)."""
        ...

    def disable(self) -> None: ...

    def face(self) -> bool:
        """Whether the tracker sees a face now."""
        ...


def rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle between two poses' rotations, in degrees."""
    r = np.asarray(a, dtype=float)[:3, :3].T @ np.asarray(b, dtype=float)[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def glide_s(degrees: float) -> float:
    """How long a glide of ours over `degrees` takes (`TURN_DEG_PER_S`, at least `MIN_GLIDE_S`)."""
    return max(MIN_GLIDE_S, abs(degrees) / TURN_DEG_PER_S)


def yaw_pose(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    pose = np.eye(4)
    pose[:2, :2] = [[c, -s], [s, c]]
    return pose


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
        tracker: Tracker | None = None,
        on_event: Callable[..., None] | None = None,
    ) -> None:
        self.mini = mini
        self.alive = alive
        self.on_failure = on_failure
        self.tracker = tracker
        self.on_event = on_event or (lambda tag, **fields: None)
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
        self._raw_head: np.ndarray = np.eye(4)
        """The head of the current tick before the look-at base."""
        self._base: np.ndarray | None = None
        self._glide: tuple[np.ndarray, np.ndarray | None, float, float] | None = None
        """A glide of the base: (from, to (None: neutral), start, seconds)."""
        self._track_lock = threading.RLock()
        self._track_thread: threading.Thread | None = None
        self.track_state = "off"
        self.camera = False
        self._last_face = 0.0
        self._searching_since = 0.0
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
            self._raw_head, self._base, self._glide = pose[0].copy(), None, None
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
        self.set_head_tracking(False, reason="stopped", hold=False)

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

    # ------------------------------------------------------------ person tracking

    @property
    def tracking(self) -> bool:
        return self._tracking

    def _state(self, state: str, **fields: object) -> None:
        self.track_state = state
        self.on_event("TRACKING", state=state, **fields)

    def _enable(self, weight: float) -> bool:
        assert self.tracker is not None
        try:
            return self.tracker.enable(weight)
        except Exception as exc:
            log.warning("head tracking weight %s failed: %s", weight, exc)
            return False

    def set_head_tracking(self, enabled: bool, *, reason: str = "", hold: bool = True) -> None:
        """Pollen's face tracker on (weight 1, or 0 while speaking) or off. Off holds the
        pose first (`hold`) so nothing jumps when the tracker lets go of the head."""
        if self.tracker is None:
            return
        with self._track_lock:
            if self._tracking == enabled:
                return
            if not enabled:
                self._tracking = False
                if hold:
                    self.hold_tracked()
                self._last_face = 0.0
                try:
                    self.tracker.disable()
                except Exception as exc:
                    log.warning("stop head tracking failed: %s", exc)
                self._state("off", reason=reason or "off")
                return
            self._tracking = True
            weight = 0.0 if self._speaking else 1.0
            self.camera = self._enable(weight)
            self._searching_since = time.monotonic()
            state = "paused" if self._speaking else "on"
            self._state(state, weight=f"{weight:g}", camera=str(self.camera).lower(),
                        reason=reason or "on")  # fmt: skip
            thread = self._track_thread
            if thread is None or not thread.is_alive():
                self._track_thread = threading.Thread(
                    target=self._track_loop, name="reachy-tracking", daemon=True
                )
                self._track_thread.start()

    def set_speaking(self, speaking: bool) -> None:
        """Tracking pauses (weight 0) while the assistant speaks, at the captured look-at
        pose, and resumes (weight 1) afterwards."""
        with self._track_lock:
            if self._speaking == speaking:
                return
            self._speaking = speaking
            if not self._tracking:
                return
            if speaking:
                self.hold_tracked()
                self._enable(0.0)
                self._state("paused", weight="0", reason="speaking")
            else:
                self._enable(1.0)
                self._searching_since = time.monotonic()
                self._state("on", weight="1", reason="spoke")

    def _track_loop(self) -> None:
        while not self._stop.is_set():
            if not self._tracking:
                return
            try:
                seen = bool(self.tracker.face()) if self.tracker is not None else False
            except Exception:
                seen = False
            now = time.monotonic()
            with self._track_lock:
                if not self._tracking:
                    return
                if seen:
                    self._last_face = now
                if self._speaking:
                    pass
                elif seen and self.track_state == "hold":
                    self._state("on", weight="1", face="true", reason="face")
                elif not seen and self.track_state == "on":
                    lost = now - max(self._last_face, self._searching_since)
                    if lost >= HOLD_AFTER_S:
                        self.hold_tracked()
                        self._last_face = 0.0
                        self._enable(0.0)  # re-armed: Pollen's tracker forgets the lost face
                        self._enable(1.0)
                        self._state("hold", after_s=f"{lost:.2f}", reason="no_face")
            time.sleep(TRACK_POLL_S)

    # ------------------------------------------------------------ the look-at base

    def _base_at(self, now: float) -> np.ndarray | None:
        """The look-at base now (under `_lock`); a finished glide settles."""
        glide = self._glide
        if glide is None:
            return self._base
        from reachy_mini.utils.interpolation import linear_pose_interpolation

        start, target, t0, seconds = glide
        s = 1.0 if seconds <= 0 else max(0.0, min(1.0, (now - t0) / seconds))
        end = np.eye(4) if target is None else target
        if s >= 1.0:
            self._base, self._glide = target, None
            return target
        self._base = np.array(linear_pose_interpolation(start, end, s), dtype=float)
        return self._base

    def hold_tracked(self) -> None:
        """Hold the pose Pollen's tracker turned the head to, if it is steering it (the camera
        is on and a face was seen within its lost timeout); otherwise the base stays as it is:
        the measured pose then differs from ours only by servo lag and steady error."""
        steering = self.camera and time.monotonic() - self._last_face < POLLEN_LOST_S
        if steering:
            self.hold_present()

    def hold_present(self) -> None:
        """Hold the head where it is now: the base becomes the measured head pose relative to
        the current move's own pose (no motion when the tracker lets go of the head)."""
        try:
            measured = np.array(self.mini.get_current_head_pose(), dtype=float)
        except Exception as exc:
            log.warning("hold: the head pose could not be read: %s", exc)
            return
        with self._lock:
            raw = self._raw_head
            base = np.eye(4)
            base[:3, :3] = raw[:3, :3].T @ measured[:3, :3]
            base[:3, 3] = measured[:3, 3] - raw[:3, 3]
            self._base, self._glide = base, None

    def turn_toward(self, yaw_deg: float) -> tuple[float, float]:
        """Glide the base to `yaw_deg` (capped at `MAX_TURN_DEG`) of yaw from neutral, slowly;
        (the yaw it turns to, the glide's seconds)."""
        yaw = max(-MAX_TURN_DEG, min(MAX_TURN_DEG, float(yaw_deg)))
        return yaw, self._glide_to(yaw_pose(yaw))

    def glide_home(self) -> float:
        """Glide the base back to neutral, slowly; its seconds."""
        return self._glide_to(None)

    def _glide_to(self, target: np.ndarray | None) -> float:
        now = time.monotonic()
        with self._lock:
            start = self._base_at(now)
            start = np.eye(4) if start is None else start
            seconds = glide_s(rotation_deg(start, np.eye(4) if target is None else target))
            self._base, self._glide = start, (start, target, now, seconds)
        return seconds

    def bake_base(self) -> float:
        """Fold the base into the held pose (no motion): a move back to neutral from here
        starts where the head is. Degrees the head is turned from the base-free pose."""
        with self._lock:
            base = self._base_at(time.monotonic())
            if base is None:
                return 0.0
            from reachy_mini.utils.interpolation import compose_world_offset

            head, antennas, yaw = self._pose
            baked = np.array(compose_world_offset(base, head), dtype=float)
            self._pose = (baked, antennas, yaw)
            self._raw_head, self._base, self._glide = baked, None, None
            return rotation_deg(head, baked)

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
            self._raw_head = head
            base = self._base_at(now)
            if base is not None:
                from reachy_mini.utils.interpolation import compose_world_offset

                head = np.array(compose_world_offset(base, head), dtype=float)
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
                # Nothing of ours may move the head now: the tracker lets go of it too.
                self.set_head_tracking(False, reason="link_down", hold=False)
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
