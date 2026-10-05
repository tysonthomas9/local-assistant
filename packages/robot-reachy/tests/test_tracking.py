"""Person tracking in the movement manager and the arbiter (S8g): the tracker's states, the
look-at base (a hold that does not move the head, the voice turn's cap and speed, the slow way
back) and the tracker switched off on every way to rest.

The daemon's HTTP API is a real local server (with the tracking routes); the SDK side is a
recorder of the calls (no robot is needed or moved).
"""

import itertools
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import numpy as np
import pytest

from assistant_robot_reachy import moves
from assistant_robot_reachy.arbiter import MotionArbiter
from assistant_robot_reachy.moves import (
    MAX_TURN_DEG,
    TURN_DEG_PER_S,
    MovementManager,
    glide_s,
    rotation_deg,
    yaw_pose,
)


class _Daemon(BaseHTTPRequestHandler):
    control_mode = "disabled"
    face = False
    camera = True
    calls: ClassVar[list[str]] = []

    def _send(self, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/daemon/status":
            self._send({"state": "running", "backend_status": {"ready": True}})
        elif self.path == "/api/state/full":
            self._send({"control_mode": self.control_mode})
        elif self.path == "/api/media/tracking/face":
            self._send({"status": "ok", "face_target": {"detected": _Daemon.face}})
        elif self.path == "/api/state/doa":
            self._send({"angle": 0.0, "speech_detected": True})  # Pollen's 0 rad: the left
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/media/tracking/enable":
            _Daemon.calls.append(f"enable {body.get('weight')}")
            self._send({"status": "ok", "enabled": _Daemon.camera})
        elif self.path == "/api/media/tracking/disable":
            _Daemon.calls.append("disable")
            self._send({"status": "ok", "enabled": False})
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


@pytest.fixture
def daemon_url() -> Iterator[str]:
    _Daemon.control_mode, _Daemon.face, _Daemon.camera, _Daemon.calls = "disabled", False, True, []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Daemon)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class RecordingMini:
    """Records the SDK calls; the measured head pose is `head` (what the tracker holds)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.head = np.eye(4)
        self.targets: list[np.ndarray] = []

    def enable_motors(self) -> None:
        self.calls.append("enable_motors")

    def disable_motors(self) -> None:
        self.calls.append("disable_motors")
        _Daemon.control_mode = "disabled"

    def wake_up(self) -> None:
        self.calls.append("wake_up")
        _Daemon.control_mode = "enabled"

    def goto_sleep(self) -> None:
        self.calls.append("goto_sleep")

    def goto_target(self, **_: Any) -> None:
        pass

    def set_target(self, head: Any = None, **_: Any) -> None:
        self.targets.append(np.array(head, dtype=float))

    def enable_wobbling(self) -> None:
        pass

    def disable_wobbling(self) -> None:
        pass

    def get_current_head_pose(self) -> np.ndarray:
        return self.head.copy()

    def get_present_antenna_joint_positions(self) -> list[float]:
        return [0.0, 0.0]


def _events() -> tuple[list[dict[str, Any]], Any]:
    seen: list[dict[str, Any]] = []
    return seen, lambda tag, **fields: seen.append({"tag": tag, **fields})


def _wait_for(check: Any, seconds: float = 3.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError("not reached in time")


def test_the_tracker_follows_the_robot_state(daemon_url: str) -> None:
    seen, on_event = _events()
    mini = RecordingMini()
    arbiter = MotionArbiter(mini, daemon_url=daemon_url, on_event=on_event)
    _Daemon.face = True
    arbiter.attend("listening")
    assert seen[0] == {"tag": "TRACKING", "state": "on", "weight": "1", "camera": "true",
                       "reason": "awake"}  # fmt: skip
    arbiter.set_speaking(True)
    assert seen[-1]["state"] == "paused"
    assert _Daemon.calls[-1] == "enable 0.0"
    arbiter.breathe()
    assert seen[-1]["state"] == "on"
    assert _Daemon.calls[-1] == "enable 1.0"
    arbiter.attend("rest")
    assert seen[-1] == {"tag": "TRACKING", "state": "off", "reason": "rest"}
    assert "disable" in _Daemon.calls
    assert mini.calls[-2:] == ["goto_sleep", "disable_motors"]


def test_without_a_face_the_pose_is_held_before_pollen_recenters(daemon_url: str) -> None:
    seen, on_event = _events()
    mini = RecordingMini()
    arbiter = MotionArbiter(mini, daemon_url=daemon_url, on_event=on_event)
    _Daemon.face = True
    arbiter.attend("listening")
    mini.head = yaw_pose(30.0)  # where Pollen's tracker had turned the head
    time.sleep(0.3)
    _Daemon.face = False  # the face leaves; Pollen's tracker keeps aiming for 2 s
    lost = time.monotonic()
    _wait_for(lambda: seen[-1]["state"] == "hold", moves.HOLD_AFTER_S + 1.0)
    held = time.monotonic() - lost
    assert moves.HOLD_AFTER_S - moves.TRACK_POLL_S <= held < moves.POLLEN_LOST_S
    time.sleep(0.1)
    # The held pose is the measured one: no jump when the tracker lets go.
    assert rotation_deg(mini.targets[-1], yaw_pose(30.0)) < 0.5
    assert _Daemon.calls[-2:] == ["enable 0.0", "enable 1.0"]  # re-armed
    _Daemon.face = True
    _wait_for(lambda: seen[-1]["state"] == "on")
    arbiter.attend("rest")


def test_without_a_tracked_face_nothing_is_captured(daemon_url: str) -> None:
    """Servo lag (the measured pose behind ours) is never frozen into the base: with no face
    seen, speaking and the no-face hold keep the head on our own pose."""
    mini = RecordingMini()
    arbiter = MotionArbiter(mini, daemon_url=daemon_url)
    arbiter.attend("listening")
    mini.head = yaw_pose(8.0)  # lagging behind, not steered by the tracker
    arbiter.set_speaking(True)
    time.sleep(0.1)
    assert rotation_deg(mini.targets[-1], np.eye(4)) < 6.0  # the listening pose, not yaw 8
    assert rotation_deg(mini.targets[-1], yaw_pose(8.0)) > 5.0
    arbiter.breathe()
    arbiter.attend("rest")


def test_a_voice_turn_is_capped_and_slow(daemon_url: str) -> None:
    mini = RecordingMini()
    arbiter = MotionArbiter(mini, daemon_url=daemon_url, tracking="voice+face")
    done = arbiter.turn_toward(75.0, "look_at")
    assert done["yaw_deg"] == MAX_TURN_DEG
    assert done["seconds"] == pytest.approx(MAX_TURN_DEG / TURN_DEG_PER_S)
    time.sleep(done["seconds"] + 0.2)
    assert rotation_deg(mini.targets[-1], yaw_pose(MAX_TURN_DEG)) < 0.5
    steps = [rotation_deg(a, b) for a, b in itertools.pairwise(mini.targets)]
    assert max(steps) * moves.CONTROL_HZ <= TURN_DEG_PER_S * 1.5
    arbiter.attend("rest")


def test_rest_from_a_tracked_pose_goes_back_slowly(daemon_url: str) -> None:
    mini = RecordingMini()
    arbiter = MotionArbiter(mini, daemon_url=daemon_url)
    arbiter.attend("listening")
    mini.head = yaw_pose(40.0)
    arbiter.manager.hold_present()
    time.sleep(0.05)
    count = len(mini.targets)
    started = time.monotonic()
    arbiter.attend("rest")
    took = time.monotonic() - started
    path = mini.targets[count:]
    assert took >= glide_s(40.0) * 0.9
    steps = [rotation_deg(a, b) for a, b in itertools.pairwise(path)]
    assert max(steps) * moves.CONTROL_HZ <= TURN_DEG_PER_S * 1.5


def test_the_brain_switches_following(daemon_url: str) -> None:
    seen, on_event = _events()
    arbiter = MotionArbiter(RecordingMini(), daemon_url=daemon_url, on_event=on_event)
    arbiter.attend("listening")
    assert arbiter.set_follow(False, "brain") is True
    assert seen[-1] == {"tag": "TRACKING", "state": "off", "reason": "brain"}
    assert arbiter.set_follow(True, "brain") is True
    assert seen[-1]["state"] == "on"
    arbiter.attend("rest")
    off = MotionArbiter(RecordingMini(), daemon_url=daemon_url, tracking="off")
    assert off.set_follow(True, "brain") is False


def test_a_stale_heartbeat_switches_the_tracker_off(daemon_url: str) -> None:
    alive = threading.Event()
    alive.set()
    seen, on_event = _events()
    arbiter = MotionArbiter(
        RecordingMini(), daemon_url=daemon_url, alive=alive.is_set, on_event=on_event
    )
    arbiter.attend("listening")
    alive.clear()
    _wait_for(lambda: seen[-1] == {"tag": "TRACKING", "state": "off", "reason": "link_down"})
    assert _Daemon.calls[-1] == "disable"


def test_the_doa_reads_as_head_yaw(daemon_url: str) -> None:
    arbiter = MotionArbiter(RecordingMini(), daemon_url=daemon_url)
    assert arbiter.read_doa() == (90.0, True)  # Pollen's 0 rad is the robot's left


def test_a_hold_does_not_move_the_head_under_an_attention_pose() -> None:
    mini = RecordingMini()
    manager = MovementManager(mini)
    pose = yaw_pose(5.0)
    manager.start((pose, (0.0, 0.0), 0.0), wobble=False)
    try:
        time.sleep(0.05)
        mini.head = yaw_pose(25.0)
        manager.hold_present()
        time.sleep(0.05)
        assert rotation_deg(mini.targets[-1], yaw_pose(25.0)) < 0.5
    finally:
        manager.stop()
