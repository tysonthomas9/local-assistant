"""The motor watchdog's decisions (the hardware run is e2e/features/robot/motor_watchdog.yaml)."""

import os
from pathlib import Path

import pytest

from assistant_robot_reachy import watchdog
from assistant_robot_reachy.daemon import _api_base


class Robot:
    def __init__(self, mode: str = "enabled", fail: bool = False) -> None:
        self.mode, self.fail, self.rests = mode, fail, 0

    def motors(self) -> str:
        return self.mode

    def rest(self) -> str:
        if self.fail:
            raise OSError("daemon busy")
        self.rests += 1
        self.mode = "disabled"
        return self.mode


def beat(path: Path, at: float) -> None:
    watchdog.touch(path)
    os.utime(path, (at, at))


def dog(path: Path, robot: Robot, started: float = 100.0) -> watchdog.Watchdog:
    return watchdog.Watchdog(robot.motors, robot.rest, path, stale_s=3.0, started=started)


def test_no_heartbeat_since_it_started_means_unarmed(tmp_path: Path) -> None:
    path, robot = tmp_path / "alive", Robot()
    beat(path, 50.0)  # an old run's file
    wd = dog(path, robot)
    assert wd.check(now=200.0) is None
    assert robot.rests == 0


def test_a_stale_heartbeat_rests_a_robot_whose_motors_are_on(tmp_path: Path) -> None:
    path, robot = tmp_path / "alive", Robot()
    wd = dog(path, robot)
    beat(path, 101.0)
    assert wd.check(now=101.5) == "armed"
    assert wd.check(now=103.9) is None
    assert wd.check(now=104.5) == "rested: motors disabled"
    assert robot.rests == 1
    assert wd.check(now=110.0) is None  # disarmed: once
    beat(path, 111.0)
    assert wd.check(now=111.2) == "armed"  # the agent is back


def test_a_removed_heartbeat_fires_at_once(tmp_path: Path) -> None:
    path, robot = tmp_path / "alive", Robot(mode="disabled")
    wd = dog(path, robot)
    beat(path, 101.0)
    wd.check(now=101.1)
    watchdog.clear(path)
    assert wd.check(now=101.6) == "fired: already at rest"
    assert robot.rests == 0  # motors off: nothing moves


def test_a_failed_rest_is_tried_again(tmp_path: Path) -> None:
    path, robot = tmp_path / "alive", Robot(fail=True)
    wd = dog(path, robot)
    beat(path, 101.0)
    wd.check(now=101.1)
    with pytest.raises(OSError, match="daemon busy"):
        wd.check(now=105.0)
    robot.fail = False
    assert wd.check(now=105.5) == "rested: motors disabled"


def test_the_api_port_comes_from_the_daemon_arguments() -> None:
    assert _api_base(["--fastapi-port", "8123"]) == "http://127.0.0.1:8123"
    assert _api_base(["--fastapi-port=8124"]) == "http://127.0.0.1:8124"
    assert _api_base([]) == "http://127.0.0.1:8000"
