"""MotionArbiter safety: a failed move always ends with goto_sleep() and the motors off; the
robot wakes once and stays awake; moves play through the movement manager.

The daemon's HTTP API is a real local server; the SDK side is a recorder of the calls the
arbiter makes (no robot is needed or moved).
"""

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from assistant_robot_reachy.arbiter import MotionArbiter, MoveIncomplete, RobotUnavailable


class _Daemon(BaseHTTPRequestHandler):
    control_mode = "disabled"

    def do_GET(self) -> None:
        if self.path == "/api/daemon/status":
            body: dict[str, Any] = {"state": "running", "backend_status": {"ready": True}}
        elif self.path == "/api/state/full":
            body = {"control_mode": self.control_mode}
        else:
            self.send_error(404)
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@pytest.fixture
def daemon_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Daemon)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class RecordingMini:
    """Records the SDK calls; `fail` names calls that raise."""

    def __init__(self, *fail: str) -> None:
        self.calls: list[str] = []
        self.fail = set(fail)
        self.targets = 0

    def _call(self, name: str) -> None:
        if not self.calls or self.calls[-1] != name or name != "set_target":
            self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    def enable_motors(self) -> None:
        self._call("enable_motors")

    def disable_motors(self) -> None:
        self._call("disable_motors")
        _Daemon.control_mode = "disabled"

    def wake_up(self) -> None:
        self._call("wake_up")
        _Daemon.control_mode = "enabled"

    def goto_sleep(self) -> None:
        self._call("goto_sleep")

    def goto_target(self, **_: Any) -> None:
        self._call("goto_target")

    def set_target(self, **_: Any) -> None:
        self.targets += 1
        self._call("set_target")

    def enable_wobbling(self) -> None:
        self._call("enable_wobbling")

    def disable_wobbling(self) -> None:
        self._call("disable_wobbling")

    def get_current_head_pose(self) -> np.ndarray:
        return np.eye(4)

    def get_present_antenna_joint_positions(self) -> list[float]:
        return [0.0, 0.0]


class _Move:
    """Like the SDK's recorded move: evaluating at or past its end raises (`fail`: while it
    plays)."""

    duration = 0.2
    fail = False

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, float]:
        if t >= self.duration or (self.fail and t > 0.0):
            raise RuntimeError("Tried to evaluate recorded move beyond its duration.")
        return np.eye(4), np.zeros(2), 0.0


class _Library:
    def get(self, name: str) -> _Move:
        del name
        return _Move()


@pytest.fixture(autouse=True)
def _at_rest() -> None:
    _Daemon.control_mode = "disabled"


def _arbiter(mini: RecordingMini, url: str) -> MotionArbiter:
    arbiter = MotionArbiter(mini, daemon_url=url)
    arbiter._library = _Library()
    return arbiter


def test_an_expression_from_rest_wakes_once_plays_through_the_manager_and_stays_awake(
    daemon_url: str,
) -> None:
    mini = RecordingMini()
    arbiter = _arbiter(mini, daemon_url)
    assert arbiter.queue_emotion("yes") is True
    assert arbiter.queue_emotion("no_firm") is True
    assert arbiter.attending
    assert mini.calls == [
        "enable_motors", "wake_up", "goto_target", "enable_wobbling", "set_target"
    ]  # fmt: skip
    assert arbiter.last_move is not None
    assert arbiter.last_move["move"] == "no1"
    assert arbiter.attend("rest") is not None
    assert mini.calls[-3:] == ["disable_wobbling", "goto_sleep", "disable_motors"]
    assert not arbiter.attending


def test_queued_emotions_play_one_after_the_other(daemon_url: str) -> None:
    mini = RecordingMini()
    arbiter = _arbiter(mini, daemon_url)
    first = arbiter.enqueue_emotion("yes")
    second = arbiter.enqueue_emotion("no")
    assert first is not None
    assert second is not None
    done = [arbiter.finish_emotion(first), arbiter.finish_emotion(second)]
    assert [d["move"] for d in done] == ["yes1", "no1"]
    assert done[1]["t_start"] >= done[0]["t_end"] - 0.02
    assert mini.calls.count("wake_up") == 1
    arbiter.attend("rest")


def test_every_express_intent_has_a_move() -> None:
    from assistant_contracts import ExpressName
    from assistant_robot_reachy.arbiter import EMOTION_MOVES

    assert {name.value for name in ExpressName} - {"random"} == set(EMOTION_MOVES)


def test_idle_keeps_the_robot_awake(daemon_url: str) -> None:
    mini = RecordingMini()
    arbiter = _arbiter(mini, daemon_url)
    assert arbiter.attend("listening") is not None
    idle = arbiter.attend("idle")
    assert idle is not None
    assert idle["asleep"] is False
    assert arbiter.attending
    assert "goto_sleep" not in mini.calls
    arbiter.attend("rest")


def test_wake_up_failing_from_rest_still_sleeps_and_turns_the_motors_off(daemon_url: str) -> None:
    mini = RecordingMini("wake_up")
    with pytest.raises(RuntimeError, match="wake_up failed"):
        _arbiter(mini, daemon_url).queue_emotion("yes")
    assert mini.calls == ["enable_motors", "wake_up", "goto_sleep", "disable_motors"]


def test_a_failing_movement_loop_sleeps_and_turns_the_motors_off(daemon_url: str) -> None:
    mini = RecordingMini("set_target")
    with pytest.raises(MoveIncomplete):
        _arbiter(mini, daemon_url).queue_emotion("yes")
    assert mini.calls[-2:] == ["goto_sleep", "disable_motors"]
    assert "disable_wobbling" in mini.calls


def test_a_move_that_raises_sleeps_and_turns_the_motors_off(daemon_url: str) -> None:
    mini = RecordingMini()
    arbiter = _arbiter(mini, daemon_url)
    _Move.fail = True
    try:
        with pytest.raises(MoveIncomplete):
            arbiter.queue_emotion("yes")
    finally:
        _Move.fail = False
    assert mini.calls[-2:] == ["goto_sleep", "disable_motors"]
    assert not arbiter.manager.running


def test_a_stale_heartbeat_stops_commanding(daemon_url: str) -> None:
    mini = RecordingMini()
    alive = threading.Event()
    alive.set()
    arbiter = MotionArbiter(mini, daemon_url=daemon_url, alive=alive.is_set)
    arbiter.attend("listening")
    alive.clear()
    time.sleep(0.2)
    sent = mini.targets
    time.sleep(0.2)
    assert mini.targets == sent
    assert not arbiter.manager.running
    assert "goto_sleep" not in mini.calls  # the watchdog rests it, unopposed


def test_a_failing_cleanup_keeps_the_original_error_and_turns_the_motors_off(
    daemon_url: str,
) -> None:
    mini = RecordingMini("wake_up", "goto_sleep")
    with pytest.raises(RuntimeError, match="wake_up failed"):
        _arbiter(mini, daemon_url).queue_emotion("yes")
    assert mini.calls == ["enable_motors", "wake_up", "goto_sleep", "disable_motors"]


def test_without_the_daemon_nothing_moves() -> None:
    mini = RecordingMini()
    with pytest.raises(RobotUnavailable):
        _arbiter(mini, "http://127.0.0.1:9").queue_emotion("yes")
    assert mini.calls == []
