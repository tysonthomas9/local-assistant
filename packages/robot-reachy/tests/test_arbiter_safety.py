"""MotionArbiter safety: a failed move always ends with goto_sleep() and the motors off.

The daemon's HTTP API is a real local server; the SDK side is a recorder of the calls the
arbiter makes (no robot is needed or moved).
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import pytest

from assistant_robot_reachy.arbiter import MotionArbiter, RobotUnavailable


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
        self._move_cancelled = False

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} failed")

    def enable_motors(self) -> None:
        self._call("enable_motors")

    def disable_motors(self) -> None:
        self._call("disable_motors")

    def wake_up(self) -> None:
        self._call("wake_up")

    def goto_sleep(self) -> None:
        self._call("goto_sleep")

    def goto_target(self, **_: Any) -> None:
        self._call("goto_target")

    def play_move(self, move: Any, **_: Any) -> None:
        del move
        self._call("play_move")

    def get_current_head_pose(self) -> np.ndarray:
        return np.eye(4)

    def get_present_antenna_joint_positions(self) -> list[float]:
        return [0.0, 0.0]


class _Move:
    duration = 0.0


class _Library:
    def get(self, name: str) -> _Move:
        del name
        return _Move()


def _arbiter(mini: RecordingMini, url: str) -> MotionArbiter:
    arbiter = MotionArbiter(mini, daemon_url=url)
    arbiter._library = _Library()
    return arbiter


def test_a_move_from_rest_wakes_plays_and_sleeps_with_motors_off(daemon_url: str) -> None:
    mini = RecordingMini()
    assert _arbiter(mini, daemon_url).queue_emotion("yes") is True
    assert mini.calls == [
        "enable_motors",
        "wake_up",
        "play_move",
        "goto_target",
        "goto_sleep",
        "disable_motors",
    ]


def test_wake_up_failing_from_rest_still_sleeps_and_turns_the_motors_off(daemon_url: str) -> None:
    mini = RecordingMini("wake_up")
    with pytest.raises(RuntimeError, match="wake_up failed"):
        _arbiter(mini, daemon_url).queue_emotion("yes")
    assert mini.calls == ["enable_motors", "wake_up", "goto_sleep", "disable_motors"]


def test_a_failing_move_sleeps_and_turns_the_motors_off(daemon_url: str) -> None:
    mini = RecordingMini("play_move")
    with pytest.raises(RuntimeError, match="play_move failed"):
        _arbiter(mini, daemon_url).queue_emotion("yes")
    assert mini.calls[-2:] == ["goto_sleep", "disable_motors"]


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
