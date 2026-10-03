"""The gadget's reachy.* commands: registration, forwarding, "robot is asleep", secret."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from musegadget.executor import Account

from gadget import restrict, robot

SECRET = "s" * 64


class FakeEndpoint:
    """Stands in for MuseHandler's robot-tools endpoint (mac/robot_tools.py)."""

    def __init__(self, secret=SECRET, status=200, answer=None):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, self.headers.get("Authorization"), body))
                if self.headers.get("Authorization") != f"Bearer {secret}":
                    code, payload = 401, {"error": "unauthorized"}
                elif status != 200:
                    code, payload = status, {"error": "asleep"}
                else:
                    code, payload = 200, answer or {"ok": True, "result": {"status": "queued", "tool": body["tool"]}}
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def endpoint():
    ep = FakeEndpoint()
    yield ep
    ep.close()


def executor(url, secret=SECRET):
    return restrict.RestrictedExecutor(Account.current(), robot.RobotTools(url=url, secret=secret, timeout_s=3))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_registration_has_robot_commands_and_nothing_dangerous():
    specs = restrict.command_specs()
    for name in ("reachy.emotion", "reachy.dance", "reachy.stop_move", "reachy.look",
                 "reachy.head_tracking", "reachy.status", "device.health"):
        assert name in specs
    for name in specs:
        assert not name.startswith(("system.", "file.", "camera")) and name != "device.ota"
        assert "camera" not in name and "photo" not in name and "volume" not in name
    for name, spec in robot.command_specs().items():
        assert spec["description"] and set(spec) >= {"required", "optional"}
    assert specs["reachy.look"]["required"]["direction"]["enum"] == ["left", "right", "up", "down", "front"]
    assert "happy" in specs["reachy.emotion"]["required"]["emotion"]["enum"]
    assert "simple_nod" in specs["reachy.dance"]["optional"]["move"]["enum"]
    assert set(specs["reachy.status"]["required"]["topic"]["enum"]) == {"name", "software", "imu"}


@pytest.mark.parametrize("command,params,tool,args", [
    ("reachy.dance", {}, "dance", {}),
    ("reachy.dance", {"move": "simple_nod"}, "dance", {"move": "simple_nod"}),
    ("reachy.emotion", {"emotion": "happy"}, "play_emotion", {"emotion": "happy"}),
    ("reachy.stop_move", {}, "stop_dance", {"dummy": True}),
    ("reachy.look", {"direction": "left"}, "move_head", {"direction": "left"}),
    ("reachy.head_tracking", {"enabled": False}, "head_tracking", {"enabled": False}),
    ("reachy.status", {"topic": "imu"}, "robot_status", {"topic": "imu"}),
])
def test_forwards_to_pollen_tool(endpoint, command, params, tool, args):
    result = executor(endpoint.url).run(command, params)
    assert result == {"ok": True, "payload": {"status": "queued", "tool": tool}}
    path, auth, body = endpoint.requests[-1]
    assert (path, auth, body) == ("/tool", f"Bearer {SECRET}", {"tool": tool, "args": args})


def test_tool_error_is_returned(endpoint):
    ep = FakeEndpoint(answer={"ok": False, "result": {"error": "Dance system not available"}})
    try:
        assert executor(ep.url).run("reachy.dance", {}) == {"ok": False, "error": "Dance system not available"}
    finally:
        ep.close()


@pytest.mark.parametrize("command,params", [
    ("reachy.look", {"direction": "behind"}),
    ("reachy.look", {}),
    ("reachy.emotion", {"emotion": "happy", "volume": 100}),
    ("reachy.head_tracking", {"enabled": "yes"}),
    ("reachy.status", {"topic": "wifi"}),
    ("reachy.dance", {"move": "moonwalk"}),
])
def test_bad_params_never_reach_the_robot(endpoint, command, params):
    result = executor(endpoint.url).run(command, params)
    assert result["ok"] is False
    assert endpoint.requests == []


def test_asleep_when_no_session_is_running():
    result = executor(f"http://127.0.0.1:{free_port()}").run("reachy.dance", {})
    assert result == {"ok": False, "error": robot.ASLEEP}


def test_asleep_without_a_run_secret(endpoint):
    assert executor(endpoint.url, secret="").run("reachy.look", {"direction": "up"}) == {"ok": False, "error": robot.ASLEEP}
    assert endpoint.requests == []


def test_asleep_when_endpoint_says_so():
    ep = FakeEndpoint(status=503)
    try:
        assert executor(ep.url).run("reachy.emotion", {"emotion": "sad"}) == {"ok": False, "error": robot.ASLEEP}
    finally:
        ep.close()


def test_wrong_secret_is_refused(endpoint):
    result = executor(endpoint.url, secret="x" * 64).run("reachy.dance", {})
    assert result["ok"] is False and "secret" in result["error"]


def test_blocked_commands_still_refused(endpoint):
    ex = executor(endpoint.url)
    for command in ("system.run", "file.read", "device.ota", "camera.capture", "reachy.volume"):
        assert ex.run(command, {}) == {"ok": False, "error": f"unsupported command: {command}"}
    assert endpoint.requests == []
