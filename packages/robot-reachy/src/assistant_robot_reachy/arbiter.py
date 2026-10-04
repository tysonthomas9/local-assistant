"""MotionArbiter: the one owner of the robot's motion, on Pollen's MovementManager.

While the robot is awake, `moves.MovementManager` (Pollen's conversation-app motion loop,
ported: no breathing) holds its pose and plays every move from one queue: our attention poses
and Pollen's recorded emotion moves (`EMOTION_MOVES`: all 42 `express` intents, mapped as
Pollen's app maps them). Speech-driven head wobble is the SDK's (`enable_wobbling`) while the
manager runs. The robot wakes once (the first attention state or expression) and stays awake;
`rest` (the body's idle timeout, stopping) puts it back to sleep. The safety rules:

- the robot must be reachable (the daemon answers and its backend is ready), else nothing moves;
- the motor torque state is read first; a robot at rest wakes with the SDK's `wake_up()`;
- Pollen's own tested motions (`wake_up()`, `goto_sleep()`, the recorded emotion moves, the
  speech wobble) are the exception to the limits below; moves WE author are at most
  `MAX_HEAD_DEG` (head) and `MAX_ANTENNA_DEG` (antennas), over at least `MIN_MOVE_S` per leg,
  and never turn the body;
- the manager stops commanding as soon as the body's link heartbeat is stale (the motor
  watchdog then rests the robot unopposed);
- on any exception (or a manager whose `set_target` keeps failing) the arbiter ends with
  `goto_sleep()` and torque off (best effort) and re-raises, so the caller reports the failure.
"""

import json
import logging
import os
import random
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from assistant_robot_reachy.moves import GotoMove, MovementManager, Played, Pose, RecordedMove

log = logging.getLogger(__name__)

MAX_HEAD_DEG = 10.0
MAX_ANTENNA_DEG = 20.0
MIN_MOVE_S = 0.6
DAEMON_URL = "http://127.0.0.1:8000"

EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
"""Pollen's recorded emotion moves (Hugging Face dataset; cached by edge_host_bootstrap.sh)."""

EMOTION_MOVES: dict[str, str] = {
    "happy": "laughing2",
    "excited": "dance3",
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
    "angry": "rage1",
    "irritated": "irritated1",
    "displeased": "displeased1",
    "disgusted": "disgusted1",
    "scared": "scared1",
    "anxious": "anxiety1",
    "surprised": "surprised1",
    "amazed": "amazed1",
    "calming": "calming1",
    "relief": "relief1",
    "impatient": "impatient2",
    "embarrassed": "shy1",
    "bored": "boredom2",
    "tired": "exhausted1",
    "sleepy": "sleep1",
    "yes": "yes1",
    "yes_understanding": "understanding2",
    "no": "no1",
    "no_sad": "no_sad1",
    "no_excited": "no_excited1",
    "no_firm": "no1",
    "welcoming": "welcoming2",
    "greeting": "welcoming2",
    "goodbye": "loving1",
    "go_away": "go_away1",
    "helpful": "helpful1",
    "dance": "dance2",
    "electric": "electric1",
    "dying": "dying1",
}
"""`express` intent (assistant_contracts.ExpressName) -> Pollen recorded move: the first choice
of Pollen's app (`_INTENT_TO_MOVES` in tools/play_emotion.py). `random` is one of
`CURATED_MOVES`. `[body.reachy] expressions` adds or replaces entries."""

CURATED_MOVES: tuple[str, ...] = (
    "anxiety1", "boredom2", "dance2", "dance3", "downcast1", "dying1", "exhausted1",
    "grateful1", "helpful1", "loving1", "rage1", "reprimand1", "resigned1", "sad1", "sad2",
    "scared1", "sleep1", "surprised1", "thoughtful1", "welcoming2", "amazed1", "attentive1",
    "attentive2", "boredom1", "confused1", "disgusted1", "displeased1", "displeased2", "fear1",
    "impatient2", "irritated1", "irritated2", "laughing1", "laughing2", "lonely1", "no1",
    "no_excited1", "no_sad1", "reprimand2", "shy1", "success1", "success2", "surprised2",
    "thoughtful2", "uncertain1", "understanding2", "yes1",
)  # fmt: skip
"""Pollen's curated pool for `random` (`_CURATED_DEFAULT_MOVES`)."""

MOVE_WAIT_SLACK_S = 5.0
"""How much longer than its own duration (plus the moves queued before it) a move may take."""

INITIAL_GOTO_S = 1.0
"""How long a recorded move takes to reach its first frame (as the SDK's `play_move`)."""

GOTO_CLOCK_RACE = "time value is out of range [0,1]"
"""The daemon's goto (reachy-mini 1.10.0, `Backend.play_move` on a `GotoMove`) loops
`while time.time() - t0 < duration` and then reads the clock again for `t`; when the second
read lands past the end (a late thread switch at the last tick, or a wall-clock step),
`time_trajectory(t / duration)` raises this ValueError and the goto task fails, although the
head already got its setpoints up to the end."""
GOTO_ATTEMPTS = 3


def goto_retrying(goto: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Run one SDK goto; the daemon's end-of-goto clock race (`GOTO_CLOCK_RACE`) re-issues it
    to the same target over the same duration (from the present pose: the rest of the way, or
    nothing left), at most `GOTO_ATTEMPTS` times. Any other error is raised at once."""
    for attempt in range(1, GOTO_ATTEMPTS + 1):
        try:
            goto(*args, **kwargs)
            return
        except Exception as exc:
            if GOTO_CLOCK_RACE not in str(exc) or attempt == GOTO_ATTEMPTS:
                raise
            log.warning("goto hit the daemon's clock race (attempt %d), re-issued", attempt)


def connect_mini(**options: Any) -> Any:
    """A connected `ReachyMini` whose every goto (ours, and the SDK's own inside `wake_up()`,
    `goto_sleep()` and `play_move()`'s initial goto) survives the daemon's clock race."""
    from reachy_mini import ReachyMini

    class _ReachyMini(ReachyMini):
        def goto_target(self, *args: Any, **kwargs: Any) -> None:
            goto_retrying(super().goto_target, *args, **kwargs)

    return _ReachyMini(**options)


ATTENTION_POSES: dict[str, tuple[float, float]] = {
    "listening": (0.0, -5.0),
    "thinking": (7.0, -3.0),
    "speaking": (0.0, 0.0),
}
"""Attention state -> head (roll, pitch) in degrees from the neutral pose: listening tilts the
head up a little toward the user, thinking tilts it sideways, speaking faces the user. All
within `MAX_HEAD_DEG` (the rotation is about 7.6 degrees at most)."""
ATTENTION_MOVE_S = 1.0
WOKEN_NEUTRAL = np.eye(4)
"""The head pose wake_up() goes to (the SDK's INIT_HEAD_POSE): neutral after waking."""
SETTLE_S = 0.6
SETTLE_PAUSE_S = 0.3
"""Each attention pose is reached over one second (slow, at least `MIN_MOVE_S`)."""
REST_STATES = ("idle", "rest", "muted", "sleeping")
"""`idle` keeps an attending robot awake at neutral (between turns of a conversation); `rest`
(the body's idle timeout, or stopping) ends attending: neutral, then asleep with the motors
off if the robot was at rest before; `muted` and `sleeping` always end asleep."""


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
    # Pollen's MuJoCo backend (a simulated robot) reports no `ready`: it is up once the
    # daemon runs.
    ready = backend.get("ready", status.get("simulation_enabled") is True)
    if status.get("state") != "running" or not ready or status.get("error"):
        return False, f"daemon state {status.get('state')!r}, backend {backend}"
    return True, "daemon running, backend ready"


@dataclass(frozen=True)
class QueuedEmotion:
    """An emotion's recorded move on the manager's queue (and the goto to its first frame)."""

    move: str
    duration: float
    start: Played
    played: Played


class MotionArbiter:
    """Serialises every move of one robot (`mini` is a connected `reachy_mini.ReachyMini`);
    while the robot is awake its `MovementManager` holds the pose and plays the moves.
    `alive()` says whether the body's link heartbeat is fresh (the manager stops commanding
    when it is not, leaving the robot to the motor watchdog)."""

    def __init__(
        self, mini: Any, *, daemon_url: str = DAEMON_URL, alive: Callable[[], bool] | None = None
    ) -> None:
        self.mini = mini
        self.daemon_url = daemon_url
        self.listening = False
        self.speaking = False
        self._lock = threading.Lock()
        self.emotions: dict[str, str] = dict(EMOTION_MOVES)
        self._library: Any = None
        self.last_move: dict[str, Any] | None = None
        self.manager = MovementManager(mini, alive=alive or (lambda: True), on_failure=self._failed)
        self._neutral: np.ndarray | None = None
        """The neutral head pose while the robot is awake for the conversation, else None."""
        self._neutral_antennas: tuple[float, float] = (0.0, 0.0)
        self._neutral_t = 0.0
        """When the neutral pose was read (this machine's monotonic clock)."""
        self._sleep_after = False
        """The robot was at rest when it woke: back to sleep, torque off, at `rest`."""
        self.failure: str | None = None

    # ------------------------------------------------------------ the facade

    def set_listening(self, on: bool) -> None:
        """Listening freezes the antennas (Pollen's manager)."""
        self.listening = on
        self.manager.set_listening(on)

    def set_speaking(self, on: bool) -> None:
        """Speaking: the speech wobble follows the played audio by itself (SDK); with face
        tracking on, tracking pauses meanwhile."""
        self.speaking = on
        self.manager.set_speaking(on)

    def breathe(self) -> None:
        """Neither listening nor speaking. No breathing motion (left out on purpose)."""
        self.listening = self.speaking = False
        self.manager.set_listening(False)
        self.manager.set_speaking(False)

    @property
    def attending(self) -> bool:
        """The robot is awake for the conversation (the manager holds its pose)."""
        return self._neutral is not None and self.manager.running

    def attend(self, state: str) -> dict[str, Any] | None:
        """Show an attention state with a small, slow head pose (`ATTENTION_POSES`).

        The first active state wakes a robot at rest (torque on, `wake_up()` to neutral,
        the movement manager started) and remembers the neutral pose; each pose is then
        reached from neutral's frame over `ATTENTION_MOVE_S`, played by the manager after
        whatever move it is playing. `idle` goes back to neutral and stays awake (the body puts
        it to `rest` after its idle timeout); `rest` goes back to neutral, stops the manager
        and, if the robot was at rest before, `goto_sleep()` with torque off; `muted` and
        `sleeping` always end asleep. A robot found with its motors off, or its manager
        stopped (the link dropped, the watchdog rested it), is woken afresh. Returns what was
        done, with the time the pose was reached on this machine's monotonic clock (for a
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
                self._rest_safely("a failed attention move")
                raise

    def _attend(self, state: str) -> dict[str, Any] | None:
        from reachy_mini.utils import create_head_pose

        started = time.monotonic()
        if state in ATTENTION_POSES:
            neutral = self._awake()
            roll, pitch = ATTENTION_POSES[state]
            target = neutral.copy()
            delta = create_head_pose(roll=roll, pitch=pitch, degrees=True)
            target[:3, :3] = neutral[:3, :3] @ delta[:3, :3]
            reached = self._goto(target, self._neutral_antennas, state)
            return {
                "state": state,
                "roll_deg": roll,
                "pitch_deg": pitch,
                "t_neutral": round(self._neutral_t, 3),
                "t_start": round(started, 3),
                "t_reached": round(reached, 3),
            }
        if state not in REST_STATES:
            return None
        if state == "idle":  # stay awake at neutral between turns
            if self._neutral is None or not self._still_awake():
                return None  # not awake: nothing to do
            reached = self._goto(self._neutral, self._neutral_antennas, state)
            return {
                "state": state,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "t_neutral": round(self._neutral_t, 3),
                "t_start": round(started, 3),
                "t_reached": round(reached, 3),
                "asleep": False,
            }
        neutral, neutral_t = self._neutral, self._neutral_t
        sleep = self._sleep_after or state in ("muted", "sleeping")
        if neutral is not None and self.manager.running:
            self.manager.clear()
            reached = self._goto(neutral, self._neutral_antennas, state)
        else:
            reached = time.monotonic()
        self.manager.stop()
        self._neutral, self._sleep_after = None, False
        if neutral is None and state == "rest":
            return None
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

    def _still_awake(self) -> bool:
        """The motors are on (else the robot was rested meanwhile: the watchdog); a manager
        stopped by a stale heartbeat starts again from the present pose."""
        if not self._torque_on():
            self.manager.stop()
            self._neutral, self._sleep_after = None, False
            return False
        if not self.manager.running:
            self.manager.start(self._present())
        return True

    def _present(self) -> Pose:
        head = np.array(self.mini.get_current_head_pose(), dtype=float)
        left, right = (float(a) for a in self.mini.get_present_antenna_joint_positions())
        return head, (left, right), 0.0

    def _awake(self) -> np.ndarray:
        """Wake the robot if it is not awake (once per conversation); the neutral pose."""
        if self._neutral is not None and self._still_awake():
            return self._neutral
        at_rest = not self._torque_on()
        if at_rest:
            self.mini.enable_motors()
            self.mini.wake_up()
            # wake_up() ends with a quick 20 deg roll and back (0.2 s each): settle slowly at
            # its neutral pose before attending from there.
            self.mini.goto_target(head=WOKEN_NEUTRAL, duration=SETTLE_S, body_yaw=None)
            time.sleep(SETTLE_PAUSE_S)
        present = self._present()
        self._neutral = WOKEN_NEUTRAL.copy() if at_rest else present[0]
        self._neutral_antennas = present[1]
        self._sleep_after = at_rest
        self._neutral_t = time.monotonic()
        self.failure = None
        self.manager.start(present)
        return self._neutral

    def _goto(self, head: np.ndarray, antennas: tuple[float, float], label: str) -> float:
        """Move to `head` over `ATTENTION_MOVE_S` after the queued moves; when it was reached."""
        played = self.manager.queue(
            GotoMove(self.manager.last_pose(), (head, antennas, 0.0), ATTENTION_MOVE_S, label)
        )
        self._wait(played, ATTENTION_MOVE_S)
        return played.ended or time.monotonic()

    def _wait(self, played: Played, seconds: float) -> None:
        """Wait for a queued move to end (the moves before it included)."""
        if not played.done.wait(seconds + MOVE_WAIT_SLACK_S + self.manager.queued_s()):
            raise MoveIncomplete(f"{played.label}: did not end in time")
        if played.cancelled:
            raise MoveIncomplete(f"{played.label}: cancelled ({self.manager.stopped_reason})")

    def queue_emotion(self, name: str, intensity: float = 1.0) -> bool:
        """Play Pollen's recorded move for the emotion `name` and wait for its end
        (`enqueue_emotion`, then `finish_emotion`); False if this body has no move for it."""
        queued = self.enqueue_emotion(name, intensity)
        if queued is None:
            return False
        self.finish_emotion(queued)
        return True

    def enqueue_emotion(self, name: str, intensity: float = 1.0) -> "QueuedEmotion | None":
        """Queue Pollen's recorded move for the emotion `name` on the movement manager, after
        the moves before it (waking the robot if it is at rest; it then stays awake); None if
        this body has no move for it. `random` picks one of Pollen's curated moves.
        `intensity` is not used by recorded moves."""
        del intensity
        move_name = random.choice(CURATED_MOVES) if name == "random" else self.emotions.get(name)
        if move_name is None:
            return None
        move = RecordedMove(move_name, self._moves().get(move_name))
        with self._lock:
            ok, detail = robot_ready(self.daemon_url)
            if not ok:
                raise RobotUnavailable(detail)
            try:
                self._awake()
                start = self.manager.queue(
                    GotoMove(
                        self.manager.last_pose(), move.evaluate(0.0), INITIAL_GOTO_S, move_name
                    )
                )
                played = self.manager.queue(move)
            except Exception:
                self._rest_safely("a failed expression")
                raise
        return QueuedEmotion(move_name, move.duration, start, played)

    def finish_emotion(self, queued: "QueuedEmotion") -> dict[str, Any]:
        """Wait for a queued emotion's move to end (the moves before it included); its
        `last_move` record (t_start/t_end on this machine's monotonic clock, so a sampler next
        to the daemon can pick out exactly the samples taken while it played)."""
        try:
            self._wait(queued.played, INITIAL_GOTO_S + queued.duration)
        except Exception:
            with self._lock:
                self._rest_safely("a failed expression")
            raise
        t_start = queued.start.started or time.monotonic()
        t_end = queued.played.ended or time.monotonic()
        record = {"move": queued.move, "duration_s": round(queued.duration, 2)}
        record["played_s"] = round(t_end - (queued.played.started or t_start), 2)
        record["t_start"] = round(t_start, 3)
        record["t_end"] = round(t_end, 3)
        self.last_move = record
        return record

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

    # ------------------------------------------------------------ safety

    def _failed(self, reason: str) -> None:
        """The manager's loop gave up (`set_target` kept failing): rest, motors off."""
        self.failure = reason
        log.error("movement manager stopped: %s", reason)
        self._neutral, self._sleep_after = None, False
        self.manager.stop()  # from the loop's own thread: wobble off, no join
        self._cleanup("a failing movement loop")

    def _rest_safely(self, after: str) -> None:
        self.manager.stop()
        self._neutral, self._sleep_after = None, False
        self._cleanup(after)

    def _cleanup(self, after: str) -> None:
        for cleanup in (self.mini.goto_sleep, self.mini.disable_motors):
            try:
                cleanup()
            except Exception:
                log.exception("%s after %s", cleanup.__name__, after)

    def _torque_on(self) -> bool:
        state = daemon_json("/api/state/full", base=self.daemon_url)
        return state.get("control_mode") != "disabled"
