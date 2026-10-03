"""client.invoke on /chat/subscribe answered with link.result, over the real Noise transport."""

import asyncio

from musegadget.executor import Account
from musegadget.link_client import DeviceDescription, encode_message

from gadget import client_invoke, restrict
from gadget.link import RobotLinkSession
from test_link import FakeVm, Pipe


class FakeRobotTools:
    def __init__(self):
        self.calls = []

    def run(self, command, params):
        self.calls.append((command, params))
        return {"ok": True, "payload": {"queued": command}}


def invoke(invoke_id, command, params_json="{}", **extra):
    return {"type": "event", "seq": 1, "event": "client.invoke",
            "payload": {"command_id": command, "invoke_id": invoke_id, "params_json": params_json,
                        "timeout_ms": 5000, **extra}}


async def started(monkeypatch=None):
    to_device, to_vm = asyncio.Queue(), asyncio.Queue()
    device_ws, vm_ws = Pipe(to_device, to_vm), Pipe(to_vm, to_device)

    async def connect(url, headers):
        return device_ws

    tools = FakeRobotTools()
    executor = restrict.RestrictedExecutor(Account.current(), robot_tools=tools)
    calls = []

    def run_command(command, params, timeout_ms):
        calls.append((command, params, timeout_ms))
        return executor.run(command, params, timeout_ms)

    device = DeviceDescription(node_id="homelink-abcdef", display_name="Reachy Mini",
                               version="0.1.0", commands=restrict.command_specs())
    session = RobotLinkSession(noise_host="gw.example", vm_id="vm", vm_auth_token="tok",
                               device=device, run_command=run_command, connect=connect)
    vm = FakeVm(vm_ws)
    stop = asyncio.Event()
    task = asyncio.ensure_future(session.run(stop))
    await vm.handshake()
    server = asyncio.ensure_future(vm.serve())
    for _ in range(200):
        if vm.register:
            break
        await asyncio.sleep(0.01)
    sub = await session.subscribe()

    async def finish():
        await sub.close()
        stop.set()
        await asyncio.wait_for(task, 2)
        server.cancel()

    return session, vm, sub, calls, tools, finish


def results(vm):
    return [m for m in vm.control_messages if m.get("method") == "link.result"]


async def wait_results(vm, n):
    for _ in range(300):
        if len(results(vm)) >= n:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)   # nothing more should follow
    return results(vm)


def test_health_and_robot_commands_answered(monkeypatch):
    monkeypatch.delenv(client_invoke.ENV, raising=False)

    async def scenario():
        session, vm, sub, calls, tools, finish = await started()
        await vm.events(invoke("inv-1", "device.health"),
                        invoke("inv-2", "reachy.dance", '{"move": "simple_nod"}'))
        got = await wait_results(vm, 2)
        by_id = {r["id"]: r for r in got}
        assert set(by_id) == {"inv-1", "inv-2"}
        assert by_id["inv-1"]["ok"] is True and by_id["inv-1"]["payload"]["hostname"] == "Reachy Mini"
        assert by_id["inv-2"] == {"method": "link.result", "id": "inv-2", "ok": True,
                                  "payload": {"queued": "reachy.dance"}}
        assert tools.calls == [("reachy.dance", {"move": "simple_nod"})]
        assert ("device.health", {}, 5000) in calls
        # The event still reaches the turn's reader.
        seen = [await sub.next(1) for _ in range(3)]
        assert [e["event"] for e in seen if e.get("type") == "event"] == ["client.invoke", "client.invoke"]
        await finish()

    asyncio.run(scenario())


def test_blocked_and_unknown_commands_get_errors(monkeypatch):
    monkeypatch.delenv(client_invoke.ENV, raising=False)

    async def scenario():
        session, vm, sub, calls, tools, finish = await started()
        await vm.events(invoke("b-1", "system.run", '{"command": "touch /tmp/x"}'),
                        invoke("b-2", "file.read", '{"path": "/etc/hostname"}'),
                        invoke("b-3", "device.ota"),
                        invoke("b-4", "something.else"),
                        invoke("b-5", "reachy.dance", "not-json"),
                        invoke("b-6", "reachy.dance", "[]"))
        got = await wait_results(vm, 6)
        assert sorted(r["id"] for r in got) == ["b-1", "b-2", "b-3", "b-4", "b-5", "b-6"]
        assert all(r["ok"] is False and r["error"] for r in got)
        assert calls == [] and tools.calls == []
        await finish()

    asyncio.run(scenario())


def test_duplicates_answered_once_and_malformed_ignored(monkeypatch):
    monkeypatch.delenv(client_invoke.ENV, raising=False)

    async def scenario():
        session, vm, sub, calls, tools, finish = await started()
        good = invoke("d-1", "reachy.emotion", '{"emotion": "happy"}')
        await vm.events(good, good,
                        {"type": "event", "event": "client.invoke", "payload": "nope"},
                        {"type": "event", "event": "client.invoke", "payload": {"command_id": "device.health"}},
                        invoke("", "device.health"),
                        invoke("bad id with spaces", "device.health"),
                        invoke("m-1", None),
                        invoke("m-2", 42))
        got = await wait_results(vm, 1)
        assert [r["id"] for r in got] == ["d-1"]
        assert len(calls) == 1
        # A link.invoke with an id already answered isn't counted twice, and vice versa.
        await vm.chunk(vm.control, encode_message(
            {"method": "link.invoke", "id": "ctl-1", "command": "device.health", "params": {}}))
        got = await wait_results(vm, 2)
        assert [r["id"] for r in got] == ["d-1", "ctl-1"]
        await vm.events(invoke("ctl-1", "device.health"))
        got = await wait_results(vm, 3)
        assert [r["id"] for r in got] == ["d-1", "ctl-1"]
        await finish()

    asyncio.run(scenario())


def test_link_invoke_still_works(monkeypatch):
    monkeypatch.delenv(client_invoke.ENV, raising=False)

    async def scenario():
        session, vm, sub, calls, tools, finish = await started()
        await vm.chunk(vm.control, encode_message(
            {"method": "link.invoke", "id": "ctl-2", "command": "reachy.look", "params": {"direction": "left"}}))
        await vm.chunk(vm.control, encode_message(
            {"method": "link.invoke", "id": "ctl-3", "command": "system.run", "params": {"command": "id"}}))
        got = await wait_results(vm, 2)
        by_id = {r["id"]: r for r in got}
        assert by_id["ctl-2"]["ok"] is True and by_id["ctl-3"]["ok"] is False
        assert tools.calls == [("reachy.look", {"direction": "left"})]
        await finish()

    asyncio.run(scenario())


def test_env_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv(client_invoke.ENV, "0")

    async def scenario():
        session, vm, sub, calls, tools, finish = await started()
        await vm.events(invoke("off-1", "device.health"), invoke("off-2", "reachy.dance", '{"move": "simple_nod"}'))
        got = await wait_results(vm, 1)
        assert got == [] and calls == []
        await finish()

    asyncio.run(scenario())


def test_parse():
    p = client_invoke.parse(invoke("x", "reachy.look", '{"direction": "up"}', timeout_ms=0))
    assert p == client_invoke.Invoke("x", "reachy.look", {"direction": "up"}, None)
    assert client_invoke.parse(invoke("x", "device.health", None)).params is None
    assert client_invoke.parse(invoke("x", "device.health", "")).params is None
    omitted = invoke("x", "device.health")
    del omitted["payload"]["params_json"]
    assert client_invoke.parse(omitted).params == {}
    assert client_invoke.parse(invoke("x", "device.health", 5)).params is None
    assert client_invoke.parse(invoke("x", "device.health", timeout_ms=True)).timeout_ms is None
    assert client_invoke.parse({"event": "link.invoke"}) is None


def test_client_invoke_first_then_link_invoke_runs_once(monkeypatch):
    monkeypatch.delenv(client_invoke.ENV, raising=False)

    async def scenario():
        session, vm, sub, calls, tools, finish = await started()
        await vm.events(invoke("r-1", "reachy.dance", '{"move": "simple_nod"}'))
        got = await wait_results(vm, 1)
        assert [r["id"] for r in got] == ["r-1"]
        await vm.chunk(vm.control, encode_message(
            {"method": "link.invoke", "id": "r-1", "command": "reachy.dance", "params": {"move": "simple_nod"}}))
        got = await wait_results(vm, 2)
        assert [r["id"] for r in got] == ["r-1"]
        assert tools.calls == [("reachy.dance", {"move": "simple_nod"})]
        await finish()

    asyncio.run(scenario())


def test_explicit_empty_params_refused_but_omitted_runs(monkeypatch):
    monkeypatch.delenv(client_invoke.ENV, raising=False)

    async def scenario():
        session, vm, sub, calls, tools, finish = await started()
        omitted = invoke("p-4", "reachy.dance")
        del omitted["payload"]["params_json"]
        await vm.events(invoke("p-1", "reachy.dance", None),
                        invoke("p-2", "reachy.dance", ""),
                        invoke("p-3", "reachy.dance", {"move": "simple_nod"}),
                        omitted)
        got = await wait_results(vm, 4)
        by_id = {r["id"]: r for r in got}
        assert set(by_id) == {"p-1", "p-2", "p-3", "p-4"}
        for refused in ("p-1", "p-2", "p-3"):
            assert by_id[refused]["ok"] is False and by_id[refused]["error"]
        assert by_id["p-4"]["ok"] is True
        assert tools.calls == [("reachy.dance", {})]   # only the omitted one ran
        await finish()

    asyncio.run(scenario())
