import asyncio
import json

import pytest

from gadget import bridge, chat
from fakes import FakeLink

FAST = chat.TurnOptions(settle_s=0, timeout_s=2)


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
        FakeLink({}), options=chat.TurnOptions(settle_s=0, timeout_s=0.3))
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


class Unfinished(FakeLink):
    """Muse starts a reply but never finishes it (e.g. it keeps calling tools)."""

    async def send_chat(self, message, session_id=None):
        ack = await super().send_chat(message, session_id)
        note = ack["response"]["result"]["message_id"]
        self.subs[-1].events += [
            self._event("delta.message_start", message_id="m-late", reply_to_message_id=note),
            self._event("delta.text_append", message_id="m-late", text="Watch this dance!"),
        ]
        return ack


def test_timeout_returns_the_text_so_far():
    """The outer guard must not fire before chat.turn's own deadline, which keeps partial text."""
    status, body = with_bridge(
        lambda port: http(port, "POST", "/turn", b'{"text": "dance"}', "application/json"),
        Unfinished({}), options=chat.TurnOptions(settle_s=0, timeout_s=0.3))
    assert (status, body) == (200, {"reply": "Watch this dance!"})


def test_outer_guard_is_later_than_the_turn_deadline():
    assert bridge.TURN_MARGIN_S > 0


async def http_lines(port: int, path: str, body: bytes):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    head = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {len(body)}\r\n"
            "Content-Type: application/json\r\n\r\n")
    writer.write(head.encode() + body)
    await writer.drain()
    raw = await reader.read()
    writer.close()
    status_line, _, rest = raw.partition(b"\r\n")
    headers, _, payload = rest.partition(b"\r\n\r\n")
    return int(status_line.split()[1]), headers.decode().lower(), [json.loads(l) for l in payload.splitlines() if l]


def test_stream_sends_sentences_then_done():
    link = FakeLink({"hi": ["Hi there. How are you?"]})   # streamed as "Hi there. H" + "ow are you?"
    status, headers, lines = with_bridge(lambda port: http_lines(port, "/turn?stream=1", b'{"text": "hi"}'), link)
    assert status == 200 and "application/x-ndjson" in headers
    assert lines == [{"text": "Hi there."}, {"text": "How are you?"}, {"done": True}]


def test_stream_timeout_after_text_just_ends():
    status, _, lines = with_bridge(lambda port: http_lines(port, "/turn?stream=1", b'{"text": "dance"}'),
                                   Unfinished({}), options=chat.TurnOptions(settle_s=0, timeout_s=0.3))
    assert status == 200 and lines == [{"text": "Watch this dance!"}, {"done": True}]


def test_stream_error_before_any_text_is_the_usual_status():
    status, _, lines = with_bridge(lambda port: http_lines(port, "/turn?stream=1", b'{"text": "silence"}'),
                                   FakeLink({}), options=chat.TurnOptions(settle_s=0, timeout_s=0.3))
    assert (status, lines) == (504, [{"error": "timeout"}])


def test_client_leaving_a_stream_stops_the_turn():
    """A barge-in on the robot closes the stream: the turn stops, so the next one doesn't wait for it."""
    class Then(FakeLink):
        async def send_chat(self, message, session_id=None):
            ack = await super().send_chat(message, session_id)
            if message == "dance":   # one sentence out at once, then Muse keeps going (tools)
                note = ack["response"]["result"]["message_id"]
                self.subs[-1].events += [
                    self._event("delta.message_start", message_id="m-late", reply_to_message_id=note),
                    self._event("delta.text_append", message_id="m-late", text="Watch this! Now the"),
                ]
            return ack

    link = Then({"what time is it": ["Tea time."]})

    async def scenario(port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        body = b'{"text": "dance"}'
        writer.write((f"POST /turn?stream=1 HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n"
                      "Content-Type: application/json\r\n\r\n").encode() + body)
        await writer.drain()
        while b'"text"' not in await reader.readline():
            pass
        writer.close()                                    # the robot's user barged in
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        reply = await http(port, "POST", "/turn", b'{"text": "what time is it"}', "application/json")
        return reply, loop.time() - t0

    (status, body), took = with_bridge(scenario, link, options=chat.TurnOptions(settle_s=0, timeout_s=5))
    assert (status, body) == (200, {"reply": "Tea time."})
    assert took < 2, "the next turn didn't wait for the abandoned one's 5 s deadline"
    assert link.subs[0].closed
