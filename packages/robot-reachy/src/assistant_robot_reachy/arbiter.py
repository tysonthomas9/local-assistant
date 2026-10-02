"""MotionArbiter: the one owner of the robot's motion, shaped like Pollen's MovementManager.

Phase 3 drops Pollen's MovementManager (continuous breathing, speech wobble, emotion queue,
face tracking) in behind this facade; until then it calls the SDK directly and does only
small, slow, bounded moves. The safety rules of every move:

- the robot must be reachable (the daemon answers and its backend is ready), else nothing moves;
- the motor torque state is read first; torque is enabled only for the move (enabling pins
  the targets to the present pose, so nothing snaps) and restored afterwards;
- the head and antennas return to the pose they started from;
- head moves are at most `MAX_HEAD_DEG` and antenna moves at most `MAX_ANTENNA_DEG`, over at
  least `MIN_MOVE_S` per leg; the body never turns (`body_yaw=None`);
- on any exception the arbiter goes back to the start pose and restores the torque state
  (best effort) and re-raises, so the caller reports the failure.

TODO(phase 3): replace the direct SDK calls with Pollen's MovementManager (breathing in
`breathe()`, speech-driven head wobble via `enable_wobbling` in `set_speaking(True)`, face
tracking in `set_listening(True)`, the full emotion library in `queue_emotion`).
"""

import json
import logging
import math
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MAX_HEAD_DEG = 10.0
MAX_ANTENNA_DEG = 20.0
MIN_MOVE_S = 0.6
DAEMON_URL = "http://127.0.0.1:8000"


class RobotUnavailable(RuntimeError):
    """The robot (its daemon) is not reachable: no motion is attempted."""


def daemon_json(path: str, timeout_s: float = 3.0, base: str = DAEMON_URL) -> Any:
    with urllib.request.urlopen(f"{base}{path}", timeout=timeout_s) as response:
        return json.loads(response.read())


def robot_ready(base: str = DAEMON_URL) -> tuple[bool, str]:
    """Whether the daemon runs with its backend ready (the robot is connected)."""
    try:
        status = daemon_json("/api/daemon/status", base=base)
    except (OSError, ValueError) as exc:
        return False, f"daemon not reachable: {exc}"
    backend = status.get("backend_status") or {}
    if status.get("state") != "running" or not backend.get("ready") or status.get("error"):
        return False, f"daemon state {status.get('state')!r}, backend {backend}"
    return True, "daemon running, backend ready"


def with_pitch_offset(pose: np.ndarray, deg: float) -> np.ndarray:
    """`pose` with only its pitch changed by `deg` (xyz Euler angles, as the daemon reports)."""
    from scipy.spatial.transform import Rotation

    roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
    target = pose.copy()
    target[:3, :3] = Rotation.from_euler("xyz", [roll, pitch + math.radians(deg), yaw]).as_matrix()
    return target


def pitch_deg(pose: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    return math.degrees(Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")[1])


class MotionArbiter:
    """Serialises every move of one robot (`mini` is a connected `reachy_mini.ReachyMini`)."""

    def __init__(self, mini: Any, *, daemon_url: str = DAEMON_URL) -> None:
        self.mini = mini
        self.daemon_url = daemon_url
        self.listening = False
        self.speaking = False
        self._lock = threading.Lock()
        self.emotions: dict[str, Callable[[float], None]] = {
            "yes": self._nod,
            "happy": self._wiggle,
        }

    # ------------------------------------------------------------ the facade

    def set_listening(self, on: bool) -> None:
        """TODO(phase 3): face tracking / attentive posture while listening."""
        self.listening = on

    def set_speaking(self, on: bool) -> None:
        """TODO(phase 3): speech-driven head wobble (`ReachyMini.enable_wobbling`)."""
        self.speaking = on

    def breathe(self) -> None:
        """Idle. TODO(phase 3): Pollen's continuous breathing motion."""
        self.listening = self.speaking = False

    def goto_sleep(self) -> None:
        """The SDK's sleep pose; torque off afterwards."""
        with self._move(restore_pose=False, torque_after=False):
            self.mini.goto_sleep()

    def wake_up(self) -> None:
        """The SDK's wake-up pose; torque stays on."""
        with self._move(restore_pose=False, torque_after=True):
            self.mini.wake_up()

    def queue_emotion(self, name: str, intensity: float = 1.0) -> bool:
        """Play the emotion `name` now; False if this body has no move for it."""
        move = self.emotions.get(name)
        if move is None:
            return False
        move(max(0.0, min(1.0, intensity)))
        return True

    # ------------------------------------------------------------ moves

    def _torque_on(self) -> bool:
        state = daemon_json("/api/state/full", base=self.daemon_url)
        return state.get("control_mode") != "disabled"

    @contextmanager
    def _move(
        self, *, restore_pose: bool = True, torque_after: bool | None = None
    ) -> Iterator[tuple[np.ndarray, list[float]]]:
        """Run one move: robot checked, torque on for it, start pose restored, torque restored.

        `torque_after` overrides the restored torque state (sleep: off, wake-up: on). On an
        exception the start pose and torque state are restored too, then it is re-raised.
        """
        with self._lock:
            ok, detail = robot_ready(self.daemon_url)
            if not ok:
                raise RobotUnavailable(detail)
            was_on = self._torque_on()
            head = np.array(self.mini.get_current_head_pose(), dtype=float)
            antennas = [float(a) for a in self.mini.get_present_antenna_joint_positions()]
            if not was_on:
                self.mini.enable_motors()
            failed = True
            try:
                yield head, antennas
                failed = False
            finally:
                try:
                    if restore_pose or failed:
                        self.mini.goto_target(
                            head=head, antennas=antennas, duration=MIN_MOVE_S, body_yaw=None
                        )
                        time.sleep(0.2)
                finally:
                    keep_on = was_on if torque_after is None or failed else torque_after
                    if not keep_on:
                        self.mini.disable_motors()

    def _nod(self, intensity: float) -> None:
        """One nod of `MAX_HEAD_DEG * intensity` degrees (pitch towards level), and back."""
        deg = MAX_HEAD_DEG * intensity
        with self._move() as (head, _):
            toward_level = -deg if pitch_deg(head) > 0 else deg
            target = with_pitch_offset(head, toward_level)
            self.mini.goto_target(head=target, duration=MIN_MOVE_S + 0.2, body_yaw=None)

    def _wiggle(self, intensity: float) -> None:
        """Both antennas move `MAX_ANTENNA_DEG * intensity` towards upright and back, twice."""
        delta = math.radians(MAX_ANTENNA_DEG * intensity)
        with self._move() as (_, antennas):
            toward = [a - math.copysign(delta, a) if abs(a) > delta else a for a in antennas]
            for _ in range(2):
                self.mini.goto_target(antennas=toward, duration=MIN_MOVE_S, body_yaw=None)
                self.mini.goto_target(antennas=antennas, duration=MIN_MOVE_S, body_yaw=None)
