"""Robot-tools endpoint tests (no robot, no Mac): secret, allowlist, asleep, forwarding into Pollen's tools.

    cd muse_gadget/mac && <app venv>/bin/python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies  # noqa: E402

import muse_handler  # noqa: E402
import robot_tools  # noqa: E402
from muse_bridge import BridgeClient  # noqa: E402
from muse_vad import EnergyVad, UtteranceSegmenter  # noqa: E402

SECRET = "a1" * 32


def load_gadget_robot():
    """The gadget's client (muse_gadget/gadget/robot.py, stdlib only), to test both ends together."""
    spec = importlib.util.spec_from_file_location("gadget_robot", HERE.parents[1] / "gadget" / "robot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def http(port, method="POST", path="/tool", body=None, secret=SECRET, content_type="application/json"):
    data = json.dumps(body).encode() if body is not None else b""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(data)}\r\nContent-Type: {content_type}\r\n"
    if secret is not None:
        head += f"Authorization: Bearer {secret}\r\n"
    writer.write(head.encode() + b"\r\n" + data)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    status_line, _, rest = raw.partition(b"\r\n")
    return int(status_line.split()[1]), json.loads(rest.split(b"\r\n\r\n", 1)[1])


def with_server(scenario, awake=True):
    calls = []

    async def dispatch(tool, args):
        calls.append((tool, args))
        return {"status": "queued", "tool": tool}

    async def main():
        server = await robot_tools.RobotToolsServer(SECRET, dispatch, lambda: awake).serve("127.0.0.1", 0)
        try:
            return await scenario(server.sockets[0].getsockname()[1])
        finally:
            server.close()
            await server.wait_closed()

    return asyncio.run(main()), calls


def test_good_request_is_dispatched():
    (status, body), calls = with_server(lambda port: http(port, body={"tool": "dance", "args": {"move": "simple_nod"}}))
    assert status == 200 and body == {"ok": True, "result": {"status": "queued", "tool": "dance"}}
    assert calls == [("dance", {"move": "simple_nod"})]


@pytest.mark.parametrize("secret", [None, "", "wrong" * 10, SECRET[:-1], SECRET + "x"])
def test_missing_or_wrong_secret_is_rejected(secret):
    (status, body), calls = with_server(lambda port: http(port, body={"tool": "dance", "args": {}}, secret=secret))
    assert (status, body, calls) == (401, {"error": "unauthorized"}, [])


@pytest.mark.parametrize("request_body", [
    {"tool": "volume_control", "args": {"volume": 100}},
    {"tool": "camera", "args": {}},
    {"tool": "go_to_sleep", "args": {}},
    {"tool": "robot_status", "args": {"topic": "wifi"}},
    {"tool": "robot_status", "args": {"topic": "account"}},
])
def test_only_allowed_tools(request_body):
    (status, body), calls = with_server(lambda port: http(port, body=request_body))
    assert (status, calls) == (403, [])


def test_asleep_when_session_is_not_up():
    (status, body), calls = with_server(lambda port: http(port, body={"tool": "dance", "args": {}}), awake=False)
    assert (status, body, calls) == (503, {"error": "asleep"}, [])


def test_refuses_non_loopback_bind():
    async def dispatch(tool, args):
        return {}

    with pytest.raises(ValueError):
        asyncio.run(robot_tools.RobotToolsServer(SECRET, dispatch, lambda: True).serve("0.0.0.0", 0))
    with pytest.raises(ValueError):
        robot_tools.RobotToolsServer("short", dispatch, lambda: True)


def test_read_secret(tmp_path, monkeypatch):
    f = tmp_path / "s.env"
    f.write_text(f"MUSE_ROBOT_TOOLS_SECRET={SECRET}\n")
    assert robot_tools.read_secret(str(f)) == SECRET
    f.write_text("MUSE_ROBOT_TOOLS_SECRET=short\n")
    assert robot_tools.read_secret(str(f)) is None
    assert robot_tools.read_secret(str(tmp_path / "missing")) is None
    monkeypatch.delenv(robot_tools.SECRET_FILE_ENV, raising=False)
    assert robot_tools.read_secret() is None


# ------------------------------------------------------------------ through MuseHandler and Pollen's tools
class FakeMovementManager:
    def __init__(self):
        self.calls = []

    def set_head_tracking(self, enabled):
        self.calls.append(("set_head_tracking", enabled))

    def clear_move_queue(self):
        self.calls.append(("clear_move_queue",))


class FakeRobot:
    class media:  # noqa: N801
        @staticmethod
        def get_output_audio_samplerate():
            return 16000


def make_handler(mm, secret=SECRET):
    deps = ToolDependencies(reachy_mini=FakeRobot(), movement_manager=mm)
    return muse_handler.MuseHandler(
        deps, bridge=BridgeClient("http://127.0.0.1:9"), transcriber=lambda a: "", synthesizer=lambda t, r: [],
        segmenter=UtteranceSegmenter(EnergyVad()), robot_tools_secret=secret, robot_tools_port=0,
    )


def test_gadget_command_runs_pollen_tool_through_the_handler(caplog):
    caplog.set_level("INFO")
    gadget_robot = load_gadget_robot()
    mm = FakeMovementManager()
    handler = make_handler(mm)

    async def main():
        startup = asyncio.create_task(handler.start_up())
        for _ in range(200):
            if handler._robot_tools_server is not None:
                break
            await asyncio.sleep(0.01)
        port = handler._robot_tools_server.sockets[0].getsockname()[1]
        client = gadget_robot.RobotTools(url=f"http://127.0.0.1:{port}", secret=SECRET, timeout_s=5)
        on = await asyncio.to_thread(client.run, "reachy.head_tracking", {"enabled": True})
        stop = await asyncio.to_thread(client.run, "reachy.stop_move", {})
        wrong = await asyncio.to_thread(
            gadget_robot.RobotTools(url=f"http://127.0.0.1:{port}", secret="b2" * 32).run, "reachy.stop_move", {})
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)
        after = await asyncio.to_thread(client.run, "reachy.head_tracking", {"enabled": False})
        return on, stop, wrong, after

    on, stop, wrong, after = asyncio.run(main())
    assert on == {"ok": True, "payload": {"status": "following"}}
    assert stop == {"ok": True, "payload": {"status": "stopped dance and cleared queue"}}
    assert wrong["ok"] is False and "secret" in wrong["error"]
    assert after == {"ok": False, "error": gadget_robot.ASLEEP}, "endpoint is gone once the session ends"
    assert mm.calls == [("set_head_tracking", True), ("clear_move_queue",)]
    logged = caplog.text
    assert "robot tool head_tracking ->" in logged and "robot tool stop_dance ->" in logged


def test_no_secret_no_endpoint():
    handler = make_handler(FakeMovementManager(), secret="")

    async def main():
        startup = asyncio.create_task(handler.start_up())
        for _ in range(100):
            if handler._is_connected():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        server = handler._robot_tools_server
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)
        return server

    assert asyncio.run(main()) is None
