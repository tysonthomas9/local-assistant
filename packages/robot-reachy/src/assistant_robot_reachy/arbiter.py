"""MotionArbiter: the one owner of the robot's motion, shaped like Pollen's MovementManager.

Phase 3 drops Pollen's MovementManager (continuous breathing, speech wobble, emotion queue,
face tracking) in behind this facade; until then it calls the SDK directly. Expressions are
Pollen's recorded emotion moves (`EMOTION_MOVES`, played with `ReachyMini.play_move`). The
safety rules of every move:

- the robot must be reachable (the daemon answers and its backend is ready), else nothing moves;
- the motor torque state is read first; torque is enabled only for the move and restored
  afterwards;
- a robot at rest (torque off) follows the SDK's standard pattern: `wake_up()` to the neutral
  pose, the move from neutral, back to neutral, `goto_sleep()`, torque off; an awake robot
  moves from, and returns to, the pose it is in;
- Pollen's own tested motions (`wake_up()`, `goto_sleep()`, the recorded emotion moves) are
  the exception to the limits below; moves WE author are at most `MAX_HEAD_DEG` (head) and
  `MAX_ANTENNA_DEG` (antennas), over at least `MIN_MOVE_S` per leg, and never turn the body;
- on any exception the arbiter ends with `goto_sleep()` and torque off (best effort) and
  re-raises, so the caller reports the failure.

TODO(phase 3): replace the direct SDK calls with Pollen's MovementManager (breathing in
`breathe()`, speech-driven head wobble via `enable_wobbling` in `set_speaking(True)`, face
tracking in `set_listening(True)`, the full emotion library in `queue_emotion`).
"""

import json
import logging
import os
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

MAX_HEAD_DEG = 10.0
MAX_ANTENNA_DEG = 20.0
MIN_MOVE_S = 0.6
DAEMON_URL = "http://127.0.0.1:8000"

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
"""Pollen's recorded emotion moves (Hugging Face dataset; cached by edge_host_bootstrap.sh)."""

EMOTION_MOVES: dict[str, str] = {
    "happy": "cheerful1",
    "excited": "enthusiastic1",
    "loving": "loving1",
    "grateful": "grateful1",
    "success": "success1",
    "thinking": "thoughtful1",
    "attentive": "attentive1",
    "confused": "confused1",
    "uncertain": "uncertain1",
    "sad": "sad1",
    "downcast": "downcast1",
    "lonely": "lonely1",
    "angry": "furious1",
    "irritated": "irritated1",
    "displeased": "displeased1",
    "disgusted": "disgusted1",
    "scared": "scared1",
    "anxious": "anxiety1",
    "surprised": "surprised1",
    "amazed": "amazed1",
    "calming": "serenity1",
    "relief": "relief1",
    "impatient": "impatient1",
    "embarrassed": "shy1",
    "bored": "boredom1",
    "tired": "tired1",
    "sleepy": "sleep1",
    "yes": "yes1",
    "yes_understanding": "understanding1",
    "no": "no1",
    "no_sad": "no_sad1",
    "no_excited": "no_excited1",
    "welcoming": "welcoming1",
    "greeting": "welcoming2",
    "go_away": "go_away1",
    "helpful": "helpful1",
    "dance": "dance1",
    "electric": "electric1",
    "dying": "dying1",
}
"""`express` intent (assistant_contracts.ExpressName) -> Pollen recorded move. Not mapped yet
(answered "no expression"): random, no_firm, goodbye. TODO(phase 3): config/bodies/reachy.toml."""

INITIAL_GOTO_S = 1.0
"""How long the SDK takes to reach a recorded move's first frame from neutral."""

ATTENTION_POSES: dict[str, tuple[float, float]] = {
    "listening": (0.0, -5.0),
    "thinking": (7.0, -3.0),
    "speaking": (0.0, 0.0),
}
"""Attention state -> head (roll, pitch) in degrees from the neutral pose: listening tilts the
head up a little toward the user, thinking tilts it sideways, speaking faces the user. All
within `MAX_HEAD_DEG` (the rotation is about 7.6 degrees at most)."""
ATTENTION_MOVE_S = 1.0
"""Each attention pose is reached over one second (slow, at least `MIN_MOVE_S`)."""
REST_STATES = ("idle", "muted", "sleeping")


class RobotUnavailable(RuntimeError):
    """The robot (its daemon) is not reachable: no motion is attempted."""


class EmotionLibraryMissing(RuntimeError):
    """Pollen's emotions dataset is not in the local Hugging Face cache: nothing moves."""


class MoveIncomplete(RuntimeError):
    """A recorded move was cancelled or ended early."""


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


class MotionArbiter:
    """Serialises every move of one robot (`mini` is a connected `reachy_mini.ReachyMini`)."""

    def __init__(self, mini: Any, *, daemon_url: str = DAEMON_URL) -> None:
        self.mini = mini
        self.daemon_url = daemon_url
        self.listening = False
        self.speaking = False
        self._lock = threading.Lock()
        self.emotions: dict[str, str] = dict(EMOTION_MOVES)
        self._library: Any = None
        self.last_move: dict[str, Any] | None = None
        self._neutral: np.ndarray | None = None
        """The neutral head pose while the robot attends (a turn is running), else None."""
        self._neutral_t = 0.0
        """When the neutral pose was read (this machine's monotonic clock)."""
        self._sleep_after = False
        """The robot was at rest when it began attending: back to sleep, torque off at idle."""

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

    @property
    def attending(self) -> bool:
        """An attention pose is held (the robot is awake for a turn)."""
        return self._neutral is not None

    def attend(self, state: str) -> dict[str, Any] | None:
        """Show an attention state with a small, slow head pose (`ATTENTION_POSES`).

        The first active state wakes a robot at rest (torque on, `wake_up()` to neutral) and
        remembers the neutral pose; each pose is then reached from neutral's frame over
        `ATTENTION_MOVE_S`. `idle` goes back to neutral and, if the robot was at rest before,
        `goto_sleep()` with torque off; `muted` and `sleeping` always end asleep. Returns what
        was done, with the time the pose was reached on this machine's monotonic clock (for a
        sampler next to the daemon), or None if nothing moved. Any error ends with
        `goto_sleep()` and torque off, and is re-raised.
        """
        with self._lock:
            ok, detail = robot_ready(self.daemon_url)
            if not ok:
                raise RobotUnavailable(detail)
            try:
                return self._attend(state)
            except Exception:
                self._neutral, self._sleep_after = None, False
                for cleanup in (self.mini.goto_sleep, self.mini.disable_motors):
                    try:
                        cleanup()
                    except Exception:
                        log.exception("%s after a failed attention move", cleanup.__name__)
                raise

    def _attend(self, state: str) -> dict[str, Any] | None:
        from reachy_mini.utils import create_head_pose

        started = time.monotonic()
        if state in ATTENTION_POSES:
            if self._neutral is None:
                at_rest = not self._torque_on()
                if at_rest:
                    self.mini.enable_motors()
                    self.mini.wake_up()
                self._sleep_after = at_rest
                self._neutral = np.array(self.mini.get_current_head_pose(), dtype=float)
                self._neutral_t = time.monotonic()
            roll, pitch = ATTENTION_POSES[state]
            target = self._neutral.copy()
            delta = create_head_pose(roll=roll, pitch=pitch, degrees=True)
            target[:3, :3] = self._neutral[:3, :3] @ delta[:3, :3]
            self.mini.goto_target(head=target, duration=ATTENTION_MOVE_S, body_yaw=None)
            return {
                "state": state,
                "roll_deg": roll,
                "pitch_deg": pitch,
                "t_neutral": round(self._neutral_t, 3),
                "t_start": round(started, 3),
                "t_reached": round(time.monotonic(), 3),
            }
        if state not in REST_STATES:
            return None
        neutral, sleep = self._neutral, self._sleep_after or state != "idle"
        neutral_t = self._neutral_t
        self._neutral, self._sleep_after = None, False
        if neutral is None and state == "idle":
            return None
        if neutral is not None:
            self.mini.goto_target(head=neutral, duration=ATTENTION_MOVE_S, body_yaw=None)
        reached = time.monotonic()
        if sleep:
            self.mini.goto_sleep()
            self.mini.disable_motors()
        return {
            "state": state,
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "t_neutral": round(neutral_t, 3) if neutral is not None else None,
            "t_start": round(started, 3),
            "t_reached": round(reached, 3),
            "asleep": sleep,
        }

    def goto_sleep(self) -> None:
        """The SDK's sleep pose; torque off afterwards."""
        with self._move(restore_pose=False, torque_after=False, from_neutral=False):
            self.mini.goto_sleep()

    def wake_up(self) -> None:
        """The SDK's wake-up pose; torque stays on."""
        with self._move(restore_pose=False, torque_after=True, from_neutral=False):
            self.mini.wake_up()

    def queue_emotion(self, name: str, intensity: float = 1.0) -> bool:
        """Play Pollen's recorded move for the emotion `name` now (to its end); False if this
        body has no move for it. `intensity` is not used by recorded moves."""
        del intensity
        move_name = self.emotions.get(name)
        if move_name is None:
            return False
        move = self._moves().get(move_name)
        with self._move():
            started = time.monotonic()
            self.mini.play_move(move, initial_goto_duration=INITIAL_GOTO_S)
            took = time.monotonic() - started
            cancelled = bool(getattr(self.mini, "_move_cancelled", False))
            if cancelled or took < move.duration:
                raise MoveIncomplete(
                    f"{move_name}: cancelled={cancelled}, took {took:.2f} s of {move.duration:.2f}"
                )
        # t_start/t_end: this machine's monotonic clock, so a sampler next to the daemon can
        # pick out exactly the samples taken while the move played.
        self.last_move = {"move": move_name, "duration_s": round(move.duration, 2)}
        self.last_move["played_s"] = round(took - INITIAL_GOTO_S, 2)
        self.last_move["t_start"] = round(started, 3)
        self.last_move["t_end"] = round(started + took, 3)
        return True

    def _moves(self) -> Any:
        """Pollen's emotions library, from the local Hugging Face cache only."""
        if self._library is None:
            from reachy_mini.motion.recorded_move import RecordedMoves

            try:
                self._library = RecordedMoves(EMOTIONS_DATASET)
            except Exception as exc:
                raise EmotionLibraryMissing(
                    f"{EMOTIONS_DATASET} is not cached in HF_HOME="
                    f"{os.environ.get('HF_HOME', '~/.cache/huggingface')}: run "
                    f"scripts/edge_host_bootstrap.sh ({type(exc).__name__}: {exc})"
                ) from exc
        return self._library

    # ------------------------------------------------------------ moves

    def _torque_on(self) -> bool:
        state = daemon_json("/api/state/full", base=self.daemon_url)
        return state.get("control_mode") != "disabled"

    @contextmanager
    def _move(
        self,
        *,
        restore_pose: bool = True,
        torque_after: bool | None = None,
        from_neutral: bool = True,
    ) -> Iterator[tuple[np.ndarray, list[float]]]:
        """Run one move: robot checked, torque on for it, start pose restored, torque restored.

        A robot at rest (torque off) follows the SDK's own pattern when `from_neutral`:
        `wake_up()` to the neutral pose, our small move from there, back to neutral, then
        `goto_sleep()` and torque off. A robot already awake moves from where it is and is put
        back there. `torque_after` overrides the restored torque state (sleep: off, wake-up:
        on). On any exception the robot ends asleep with torque off, and it is re-raised.
        """
        with self._lock:
            ok, detail = robot_ready(self.daemon_url)
            if not ok:
                raise RobotUnavailable(detail)
            was_on = self._torque_on()
            woke = False
            failed = True
            try:
                if not was_on:
                    self.mini.enable_motors()
                    if from_neutral:
                        self.mini.wake_up()
                        woke = True
                head = np.array(self.mini.get_current_head_pose(), dtype=float)
                antennas = [float(a) for a in self.mini.get_present_antenna_joint_positions()]
                yield head, antennas
                if restore_pose:
                    self.mini.goto_target(
                        head=head, antennas=antennas, duration=MIN_MOVE_S, body_yaw=None
                    )
                    time.sleep(0.2)
                if woke:
                    self.mini.goto_sleep()
                failed = False
            finally:
                keep_on = was_on if torque_after is None else torque_after
                if failed:
                    # Any failure once motion setup began (wake_up() included): best-effort
                    # goto_sleep(), then torque off; cleanup errors never hide the original.
                    for cleanup in (self.mini.goto_sleep, self.mini.disable_motors):
                        try:
                            cleanup()
                        except Exception:
                            log.exception("%s after a failed move", cleanup.__name__)
                elif not keep_on:
                    self.mini.disable_motors()
