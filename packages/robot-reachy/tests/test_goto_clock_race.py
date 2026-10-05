"""The daemon's end-of-goto clock race: every SDK goto is re-issued, nothing else is.

reachy-mini 1.10.0's daemon fails a goto task with "time value is out of range [0,1]" when its
second clock read lands past the goto's end. `connect_mini` gives a `ReachyMini` whose
`goto_target` (used by our moves and by the SDK's own `wake_up()`, `goto_sleep()` and
`play_move()`) re-issues such a goto. Here the SDK's real `wake_up()` / `goto_sleep()` run
against a stand-in for the one daemon call they make, `ReachyMini.goto_target`'s task.
"""

from typing import Any

import pytest
from reachy_mini import ReachyMini

from assistant_robot_reachy.arbiter import GOTO_ATTEMPTS, connect_mini, goto_retrying

RACE = Exception("Task failed with error: time value is out of range [0,1]")


class _Daemon:
    """The goto tasks the daemon ran; `failures` are raised, in order, before successes."""

    def __init__(self, *failures: Exception) -> None:
        self.failures = list(failures)
        self.gotos: list[float] = []

    def goto_target(self, _mini: Any, *_args: Any, duration: float = 0.5, **_: Any) -> None:
        self.gotos.append(duration)
        if self.failures:
            raise self.failures.pop(0)


class _Media:
    def play_sound(self, _name: str) -> None:
        pass


def _mini(monkeypatch: pytest.MonkeyPatch, daemon: _Daemon) -> Any:
    monkeypatch.setattr(ReachyMini, "__init__", lambda self, **_: None)
    monkeypatch.setattr(ReachyMini, "goto_target", daemon.goto_target)
    monkeypatch.setattr(ReachyMini, "media", property(lambda self: _Media()))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return connect_mini(connection_mode="localhost_only")


def test_the_sdks_wake_up_survives_the_clock_race(monkeypatch: pytest.MonkeyPatch) -> None:
    daemon = _Daemon(RACE)
    _mini(monkeypatch, daemon).wake_up()
    # wake_up(): neutral over 2 s (raced, re-issued), the 20 deg roll, back to neutral.
    assert daemon.gotos == [2, 2, 0.2, 0.2]


def test_the_sdks_goto_sleep_survives_the_clock_race(monkeypatch: pytest.MonkeyPatch) -> None:
    daemon = _Daemon(RACE, RACE)
    mini = _mini(monkeypatch, daemon)
    monkeypatch.setattr(ReachyMini, "get_current_joint_positions", lambda self: ([0.0] * 7, []))
    mini.goto_sleep()
    assert daemon.gotos == [1, 1, 1, 2]


def test_a_race_on_every_attempt_fails_the_move(monkeypatch: pytest.MonkeyPatch) -> None:
    daemon = _Daemon(*[RACE] * GOTO_ATTEMPTS)
    with pytest.raises(Exception, match=r"out of range \[0,1\]"):
        _mini(monkeypatch, daemon).wake_up()
    assert daemon.gotos == [2] * GOTO_ATTEMPTS


def test_any_other_goto_error_is_not_retried() -> None:
    calls: list[int] = []

    def goto() -> None:
        calls.append(1)
        raise TimeoutError("Task did not complete in time.")

    with pytest.raises(TimeoutError):
        goto_retrying(goto)
    assert calls == [1]


def test_the_sdk_goto_fails_just_past_its_end() -> None:
    """The root cause, in the SDK itself: a goto evaluated a hair past its duration raises the
    error the daemon reports (its loop reads the clock twice; the second read can be late)."""
    import numpy as np
    from reachy_mini.motion.goto import GotoMove
    from reachy_mini.utils.interpolation import InterpolationTechnique

    goto = GotoMove(
        start_head_pose=np.eye(4),
        target_head_pose=np.eye(4),
        start_antennas=np.zeros(2),
        target_antennas=np.zeros(2),
        start_body_yaw=0.0,
        target_body_yaw=0.0,
        duration=1.0,
        method=InterpolationTechnique.MIN_JERK,
    )
    goto.evaluate(1.0)
    with pytest.raises(ValueError, match=r"time value is out of range \[0,1\]"):
        goto.evaluate(1.0 + 1e-6)
