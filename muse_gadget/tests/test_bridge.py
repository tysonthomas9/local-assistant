import asyncio
import json

import pytest

from gadget import bridge, chat
from fakes import FakeLink

FAST = chat.TurnOptions(poll_s=0, settle_s=0, timeout_s=2)


async def http(port: int, method: str, path: str, body: bytes = b"", content_type: str | None = None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    head = f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {len(body)}\r\n"
    if content_type:
        head += f"Content-Type: {content_type}\r\n"
    writer.write(head.encode() + b"\r\n" + body)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    status_line, _, rest = raw.partition(b"\r\n")
    payload = rest.split(b"\r\n\r\n", 1)[1]
    return int(status_line.split()[1]), json.loads(payload)


def with_bridge(scenario, link=None, paired=True, options=FAST):
    async def main():
        web = bridge.Bridge(lambda: link, lambda: paired, options)
        server = await web.serve("127.0.0.1", 0, in_container=False)
        port = server.sockets[0].getsockname()[1]
        try:
            return await scenario(port)
        finally:
            server.close()
            await server.wait_closed()
    return asyncio.run(main())


def test_turn_returns_reply():
    link = FakeLink({"what time is it": ["Tea time."]})
    status, body = with_bridge(
        lambda port: http(port, "POST", "/turn", b'{"text": "what time is it"}', "application/json"), link)
    assert (status, body) == (200, {"reply": "Tea time."})


def test_not_paired_and_link_down():
    async def scenario(port):
        return await http(port, "POST", "/turn", b'{"text": "hi"}', "application/json; charset=utf-8")
    assert with_bridge(scenario, link=None, paired=False) == (503, {"error": "not_paired"})
    assert with_bridge(scenario, link=None, paired=True) == (503, {"error": "link_down"})


def test_health():
    assert with_bridge(lambda port: http(port, "GET", "/health"), None, False) == (
        200, {"paired": False, "linked": False})
    assert with_bridge(lambda port: http(port, "GET", "/health"), FakeLink(), True) == (
        200, {"paired": True, "linked": True})


def test_audio_and_non_json_get_415_bad_text_400():
    async def scenario(port):
        return [
            await http(port, "POST", "/turn", b"RIFF....WAVE", "audio/wav"),
            await http(port, "POST", "/turn", b"text=hi", "application/x-www-form-urlencoded"),
            await http(port, "POST", "/turn", b"{not json", "application/json"),
            await http(port, "POST", "/turn", b'{"text": "  "}', "application/json"),
            await http(port, "GET", "/turn"),
            await http(port, "GET", "/nope"),
        ]
    results = with_bridge(scenario, FakeLink())
    assert [s for s, _ in results] == [415, 415, 415, 400, 405, 404]


def test_timeout_is_504():
    status, body = with_bridge(
        lambda port: http(port, "POST", "/turn", b'{"text": "silence"}', "application/json"),
        FakeLink({}), options=chat.TurnOptions(poll_s=0.05, settle_s=0, timeout_s=0.3))
    assert (status, body) == (504, {"error": "timeout"})


def test_bridge_binds_only_to_loopback():
    for host in ("127.0.0.1", "::1", "localhost"):
        bridge.check_bind_host(host, in_container=False)
    for host in ("0.0.0.0", "::", "203.0.113.10", "10.0.0.1"):
        with pytest.raises(ValueError):
            bridge.check_bind_host(host, in_container=False)
    bridge.check_bind_host("0.0.0.0", in_container=True)
    with pytest.raises(ValueError):
        bridge.check_bind_host("203.0.113.10", in_container=True)


def test_serve_refuses_public_address():
    async def main():
        web = bridge.Bridge(lambda: None, lambda: False)
        with pytest.raises(ValueError):
            await web.serve("0.0.0.0", 0, in_container=False)
    asyncio.run(main())


def test_default_server_socket_is_loopback(monkeypatch):
    monkeypatch.delenv(bridge.IN_CONTAINER_ENV, raising=False)

    async def main():
        web = bridge.Bridge(lambda: None, lambda: False)
        server = await web.serve(port=0)
        addrs = {s.getsockname()[0] for s in server.sockets}
        server.close()
        await server.wait_closed()
        return addrs
    assert asyncio.run(main()) == {"127.0.0.1"}
