"""EdgeLink steps: a real link server console and real client consoles on 127.0.0.1.

The server runs `python -m assistant_link.server --console` and every client
`python -m assistant_link.client --console`, each a separate process on a free loopback port.
Steps type commands into their stdin and read what they print (see `assistant_link.console`
for the line format). Faults are real signals: SIGKILL, SIGSTOP, SIGCONT.

Every expectation consumes the output line it matched, so two identical messages need two
expectations, and a "reconnects" step only counts a welcome printed after the earlier ones.
"""

import asyncio
import base64
import contextlib
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from assistant_contracts.frames import FrameKind
from assistant_testing import edge_host
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.processes import ManagedProcess
from assistant_testing.steps import edge_host as edge_host_steps

SERVER = "server"
DEV_TOKEN = "e2e-dev-token"
BACKOFF_MIN_S = 0.5
BACKOFF_MAX_S = 10.0
JITTER = 0.2


@dataclass
class _Link:
    port: int
    token: str
    consumed: dict[str, set[int]] = field(default_factory=dict)
    sent_crc: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Line:
    index: int
    tag: str
    fields: dict[str, str]
    payload: dict[str, Any] | None
    text: str


def _parse(index: int, text: str) -> _Line | None:
    head, brace, rest = text.partition("{")
    tokens = head.split()
    if not tokens or not tokens[0].isupper():
        return None
    fields = dict(t.split("=", 1) for t in tokens[1:] if "=" in t)
    payload: dict[str, Any] | None = None
    if brace:
        try:
            payload = json.loads(brace + rest)
        except json.JSONDecodeError:
            return None
    return _Line(index, tokens[0], fields, payload, text)


def _subset(expected: Any, actual: Any) -> bool:
    """`expected` is contained in `actual` (dicts recursively; everything else equal)."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _subset(v, actual[k]) for k, v in expected.items()
        )
    if isinstance(expected, float | int) and isinstance(actual, float | int):
        return float(expected) == float(actual)
    return expected == actual


def _link(ctx: ScenarioContext) -> _Link:
    link = ctx.state.get("link")
    if link is None:
        raise AssertionError("no link server started; use start_link_server first")
    return link


def _client_name(client: str) -> str:
    return f"client:{client}"


def _process_name(process: str) -> str:
    return SERVER if process == SERVER else _client_name(process.removeprefix("client:"))


def _kind(name: str) -> FrameKind:
    try:
        return FrameKind(int(name, 0))
    except ValueError:
        pass
    try:
        return FrameKind[name.upper()]
    except KeyError:
        raise ValueError(f"unknown frame kind {name!r}") from None


def _kind_field(kind: FrameKind) -> str:
    return f"0x{int(kind):02x}"


async def _expect(
    ctx: ScenarioContext,
    process: str,
    tags: set[str],
    within_s: float,
    *,
    fields: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    what: str,
) -> _Line:
    """Wait for (and consume) an output line with one of `tags`, these fields and payload."""
    link = _link(ctx)
    proc = ctx.processes.get(process)
    consumed = link.consumed.setdefault(process, set())
    want = fields or {}

    def match(text: str) -> bool:
        line = _parse(-1, text)
        if line is None or line.tag not in tags:
            return False
        if any(line.fields.get(k) != v for k, v in want.items()):
            return False
        return payload is None or (line.payload is not None and _subset(payload, line.payload))

    index, text = await proc.wait_for_output(match, within_s, skip=consumed.__contains__, what=what)
    consumed.add(index)
    parsed = _parse(index, text)
    assert parsed is not None
    return parsed


def _get_lines(proc: ManagedProcess, tag: str) -> list[_Line]:
    """Every output line of `proc` with this tag so far (consumed or not)."""
    return [
        line
        for i, text in enumerate(proc.lines)
        if (line := _parse(i, text)) is not None and line.tag == tag
    ]


async def _type(ctx: ScenarioContext, process: str, command: str) -> ManagedProcess:
    proc = ctx.processes.get(process)
    await proc.write_line(command)
    return proc


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# ---------------------------------------------------------------- processes


@step("start_link_server")
async def start_link_server(
    ctx: ScenarioContext, accept_opus: bool = False, token: str = DEV_TOKEN
) -> None:
    """Start the link server console (a real LinkServer) on a free 127.0.0.1 port."""
    port = _free_port()
    ctx.state["link"] = _Link(port=port, token=token)
    argv = [sys.executable, "-m", "assistant_link.server", "--console"]
    argv += ["--host", "127.0.0.1", "--port", str(port), "--token", token]
    if accept_opus:
        argv.append("--accept-opus")
    await ctx.processes.start(SERVER, argv, stdin=True, ready_line=r"^LISTENING ", ready_timeout=30)


@step("start_link_client")
async def start_link_client(
    ctx: ScenarioContext,
    id: str,
    token: str | None = None,
    proto: str | None = None,
    opus: bool = False,
    speak_text: bool = False,
    wait: bool = True,
    where: Literal["pc", "edge_host"] = "pc",
) -> None:
    """Start a link client console (a real LinkClient) that dials the server.

    `wait` (default) waits for its welcome; with `wait: false` the scenario checks it.
    `where: edge_host` runs it on the robot's machine from the synced checkout (see
    `code_synced_to_edge_host`), dialling 127.0.0.1 there through an `ssh -R` tunnel.
    """
    link = _link(ctx)
    port, python, ssh = link.port, sys.executable, None
    if where == "edge_host":
        host = edge_host_steps.host_of(ctx)
        if host.ssh is not None:
            if ctx.state.get("edge_sha") is None:
                raise AssertionError(
                    "code not synced to the edge host; use code_synced_to_edge_host"
                )
            port = await edge_host_steps.reverse_tunnel(ctx, link.port)
            python, ssh = f"{edge_host.REMOTE_VENV}/bin/python", host.ssh
    argv = [python, "-m", "assistant_link.client", "--console", "--device-id", id]
    argv += ["--url", f"ws://127.0.0.1:{port}/edge/v1", "--token", token or link.token]
    if proto is not None:
        argv += ["--proto", proto]
    if opus:
        argv.append("--opus")
    if speak_text:
        argv.append("--speak-text")
    if ssh is not None:
        await ctx.processes.start(
            _client_name(id),
            argv,
            stdin=True,
            ssh=ssh,
            remote_cwd=f"{edge_host.EDGE_DIR}/src",
            env={"PYTHONUNBUFFERED": "1"},
        )
    else:
        await ctx.processes.start(_client_name(id), argv, stdin=True)
    if wait:
        await _expect(ctx, _client_name(id), {"WELCOME"}, 30, what="WELCOME")


@step("kill_process")
async def kill_process(
    ctx: ScenarioContext,
    process: str,
    signal_name: Literal["KILL", "STOP", "CONT", "TERM", "HUP"] = "KILL",
    within_s: float = 60.0,
) -> None:
    """Send a real signal to `server`, a client id or `daemon` (the reachy-mini daemon): KILL
    (kill -9), STOP (freeze), CONT, TERM or HUP (as when its SSH session drops). After KILL,
    TERM or HUP the process has exited (TERM and HUP: within `within_s`)."""
    name = "daemon" if process == "daemon" else _process_name(process)  # steps/edge_host DAEMON
    proc = ctx.processes.get(name)
    proc.send_signal(signal.Signals[f"SIG{signal_name}"])
    if signal_name == "KILL":
        await proc.wait(10)
    elif signal_name in ("TERM", "HUP"):
        code = await proc.wait(within_s)
        print(f"{process} exited ({code}) after SIG{signal_name}")


@step("restart_process")
async def restart_process(ctx: ScenarioContext, process: str) -> None:
    """Start a killed process again with the same command line (the server: same port;
    `daemon`: the reachy-mini daemon, ready when its API serves)."""
    name = "daemon" if process == "daemon" else _process_name(process)
    ready = {SERVER: r"^LISTENING ", "daemon": r"Uvicorn running on "}.get(name)
    await ctx.processes.restart(name, ready_line=ready, ready_timeout=120 if ready else 30)
    # The new process prints from line 0 again: what was consumed belonged to the old one.
    _link(ctx).consumed.pop(name, None)


@step("wait")
async def wait(ctx: ScenarioContext, seconds: float) -> None:
    """Let real time pass (e.g. keep a killed server down)."""
    del ctx
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------- messages


def _fields_json(fields: dict[str, Any] | None) -> str:
    return json.dumps(fields or {}, default=str)


async def _console_ok(ctx: ScenarioContext, process: str, tag: str, what: str, **want: str) -> None:
    line = await _expect(ctx, process, {tag, "CONSOLE-ERROR"}, 10, fields=want, what=what)
    if line.tag == "CONSOLE-ERROR":
        raise AssertionError(f"{process} console refused the command: {line.payload}")


@step("client_sends")
async def client_sends(
    ctx: ScenarioContext, client: str, type: str, fields: dict[str, Any] | None = None
) -> None:
    """The client sends a message of `type` with these fields."""
    name = _client_name(client)
    await _type(ctx, name, f"send {type} {_fields_json(fields)}")
    await _console_ok(ctx, name, "SENT", f"SENT {type}", type=type)


@step("server_sends")
async def server_sends(
    ctx: ScenarioContext, client: str, type: str, fields: dict[str, Any] | None = None
) -> None:
    """The server sends a message of `type` with these fields to the client."""
    await _type(ctx, SERVER, f"send {client} {type} {_fields_json(fields)}")
    await _console_ok(ctx, SERVER, "SENT", f"SENT {type}", type=type, device=client)


@step("server_receives")
async def server_receives(
    ctx: ScenarioContext,
    type: str,
    client: str | None = None,
    fields: dict[str, Any] | None = None,
    within_s: float = 5.0,
) -> None:
    """The server received a message of `type` (from `client`) containing these fields."""
    want = {"type": type, **({"device": client} if client else {})}
    await _expect(ctx, SERVER, {"RECV"}, within_s, fields=want, payload=fields, what=f"RECV {type}")


@step("client_receives")
async def client_receives(
    ctx: ScenarioContext,
    client: str,
    type: str,
    fields: dict[str, Any] | None = None,
    within_s: float = 5.0,
) -> None:
    """The client received a message of `type` containing these fields (`welcome` included)."""
    name = _client_name(client)
    if type == "welcome":
        await _expect(
            ctx,
            name,
            {"WELCOME", "RECV"},
            within_s,
            payload={"type": "welcome", **(fields or {})},
            what="welcome",
        )
        return
    await _expect(
        ctx, name, {"RECV"}, within_s, fields={"type": type}, payload=fields, what=f"RECV {type}"
    )


@step("server_connected")
async def server_connected(
    ctx: ScenarioContext,
    client: str,
    fields: dict[str, Any] | None = None,
    within_s: float = 5.0,
) -> None:
    """The server accepted the client's hello (containing these fields) and opened a session."""
    await _expect(
        ctx,
        SERVER,
        {"CONNECTED"},
        within_s,
        fields={"device": client},
        payload={"type": "hello", "device_id": client, **(fields or {})},
        what=f"CONNECTED {client}",
    )


@step("client_sends_raw")
async def client_sends_raw(
    ctx: ScenarioContext, client: str, message: dict[str, Any] | None = None, text: str = ""
) -> None:
    """Send unchecked text (a JSON `message`, or `text`) from the client: tests peer checks."""
    raw = json.dumps(message) if message is not None else text
    name = _client_name(client)
    await _type(ctx, name, f"raw {raw}")
    await _console_ok(ctx, name, "SENT-RAW", "SENT-RAW")


@step("server_sends_raw")
async def server_sends_raw(
    ctx: ScenarioContext, client: str, message: dict[str, Any] | None = None, text: str = ""
) -> None:
    """Send unchecked text (a JSON `message`, or `text`) from the server to the client."""
    raw = json.dumps(message) if message is not None else text
    await _type(ctx, SERVER, f"raw {client} {raw}")
    await _console_ok(ctx, SERVER, "SENT-RAW", "SENT-RAW", device=client)


@step("sender_refuses_locally")
async def sender_refuses_locally(
    ctx: ScenarioContext,
    client: str,
    sender: Literal["edge", "brain"],
    type: str,
    fields: dict[str, Any] | None = None,
) -> None:
    """The sender's own link refuses to send `type` (wrong direction) before it hits the wire."""
    if sender == "edge":
        name, command = _client_name(client), f"send {type} {_fields_json(fields)}"
    else:
        name, command = SERVER, f"send {client} {type} {_fields_json(fields)}"
    await _type(ctx, name, command)
    line = await _expect(ctx, name, {"CONSOLE-ERROR", "SENT"}, 10, what="the local refusal")
    assert line.tag == "CONSOLE-ERROR", f"{name} sent {type}: {line.text}"
    assert line.payload is not None, line.text
    assert "WrongDirection" in line.payload["detail"], line.text


@step("server_refuses")
async def server_refuses(
    ctx: ScenarioContext, code: str, client: str | None = None, within_s: float = 5.0
) -> None:
    """The server refused a connection or an item with this error code (auth, frame_refused)."""
    want = {"code": code, **({"device": client} if client else {})}
    await _expect(ctx, SERVER, {"REFUSED"}, within_s, fields=want, what=f"REFUSED {code}")


@step("client_refuses")
async def client_refuses(
    ctx: ScenarioContext, client: str, code: str, within_s: float = 5.0
) -> None:
    """The client refused something the server sent, with this error code."""
    name = _client_name(client)
    await _expect(ctx, name, {"REFUSED"}, within_s, fields={"code": code}, what=f"REFUSED {code}")


# ---------------------------------------------------------------- frames


async def _send_frame(
    ctx: ScenarioContext,
    process: str,
    command: str,
    sender_key: str,
    kind: FrameKind,
    expect: str,
    device: dict[str, str],
) -> None:
    await _type(ctx, process, command)
    line = await _expect(
        ctx,
        process,
        {"SENT-FRAME", "FRAME-REFUSED", "CONSOLE-ERROR"},
        10,
        what="the frame result",
    )
    if expect == "refused_locally":
        assert line.tag in {"FRAME-REFUSED", "CONSOLE-ERROR"}, f"frame was sent: {line.text}"
        return
    assert line.tag == "SENT-FRAME", f"frame not sent: {line.text}"
    assert all(line.fields.get(k) == v for k, v in device.items()), line.text
    _link(ctx).sent_crc[(sender_key, _kind_field(kind))] = line.fields["crc"]


def _frame_command(kind: str, bytes: int, stream: int, seq: int, unchecked: bool) -> str:
    opts = f"{kind} bytes={bytes} stream={stream} seq={seq}"
    return f"{opts} unchecked" if unchecked else opts


@step("client_sends_frame")
async def client_sends_frame(
    ctx: ScenarioContext,
    client: str,
    kind: str,
    bytes: int = 640,
    stream: int = 1,
    seq: int = 1,
    unchecked: bool = False,
    expect: Literal["sent", "refused_locally"] = "sent",
) -> None:
    """The client sends a binary frame. `unchecked` skips its own codec checks."""
    frame_kind = _kind(kind)
    command = "frame " + _frame_command(kind, bytes, stream, seq, unchecked)
    await _send_frame(ctx, _client_name(client), command, client, frame_kind, expect, {})


@step("server_sends_frame")
async def server_sends_frame(
    ctx: ScenarioContext,
    client: str,
    kind: str,
    bytes: int = 640,
    stream: int = 1,
    seq: int = 1,
    unchecked: bool = False,
    expect: Literal["sent", "refused_locally"] = "sent",
) -> None:
    """The server sends a binary frame to the client. `unchecked` skips its codec checks."""
    frame_kind = _kind(kind)
    command = f"frame {client} " + _frame_command(kind, bytes, stream, seq, unchecked)
    await _send_frame(ctx, SERVER, command, SERVER, frame_kind, expect, {"device": client})


async def _receive_frame(
    ctx: ScenarioContext,
    process: str,
    sender_key: str,
    kind: str,
    within_s: float,
    extra: dict[str, str],
    size: int | None,
) -> None:
    frame_kind = _kind(kind)
    want = {"kind": _kind_field(frame_kind), **extra}
    if size is not None:
        want["bytes"] = str(size)
    crc = _link(ctx).sent_crc.get((sender_key, _kind_field(frame_kind)))
    if crc is not None:
        want["crc"] = crc
    await _expect(ctx, process, {"FRAME"}, within_s, fields=want, what=f"FRAME {kind} {want}")


@step("server_receives_frame")
async def server_receives_frame(
    ctx: ScenarioContext,
    client: str,
    kind: str,
    bytes: int | None = None,
    within_s: float = 5.0,
) -> None:
    """The server received this frame kind from the client, with the payload CRC it sent."""
    await _receive_frame(ctx, SERVER, client, kind, within_s, {"device": client}, bytes)


@step("client_receives_frame")
async def client_receives_frame(
    ctx: ScenarioContext,
    client: str,
    kind: str,
    bytes: int | None = None,
    within_s: float = 5.0,
) -> None:
    """The client received this frame kind from the server, with the payload CRC it sent."""
    await _receive_frame(ctx, _client_name(client), SERVER, kind, within_s, {}, bytes)


@step("capability_negotiated")
async def capability_negotiated(
    ctx: ScenarioContext,
    client: str,
    capability: Literal["opus", "speak_text"],
    negotiated: bool = True,
) -> None:
    """Both ends agree whether `capability` is on for this client's current session."""
    proc = ctx.processes.get(_client_name(client))
    welcomes = [
        line
        for i, text in enumerate(proc.lines)
        if (line := _parse(i, text)) is not None and line.tag == "WELCOME"
    ]
    assert welcomes, f"client {client} has no welcome yet"
    welcome = welcomes[-1]
    expected = str(negotiated).lower()
    if capability == "opus":
        assert welcome.fields.get("opus") == expected, welcome.text
        server = ctx.processes.get(SERVER)
        connected = [
            line
            for i, text in enumerate(server.lines)
            if (line := _parse(i, text)) is not None
            and line.tag == "CONNECTED"
            and line.fields.get("device") == client
        ]
        assert connected, f"the server has no session for {client}"
        assert connected[-1].fields.get("opus") == expected, connected[-1].text
    else:
        assert welcome.payload is not None
        assert welcome.payload["audio"]["speak_text"] is negotiated, welcome.text


# ---------------------------------------------------------------- closes and reconnects


@step("connection_closed_with_code")
async def connection_closed_with_code(
    ctx: ScenarioContext, client: str, code: int, within_s: float = 5.0
) -> None:
    """The client's connection closed with this code; 4001/4003 also end the client (exit 2)."""
    name = _client_name(client)
    await _expect(
        ctx, name, {"CLOSED"}, within_s, fields={"code": str(code)}, what=f"CLOSED {code}"
    )
    if code in (4001, 4003):
        await _expect(ctx, name, {"GAVE-UP"}, 5, fields={"code": str(code)}, what="GAVE-UP")
        exit_code = await ctx.processes.get(name).wait(10)
        assert exit_code == 2, f"client {client} exited with {exit_code}, expected 2"


@step("server_disconnects")
async def server_disconnects(
    ctx: ScenarioContext,
    client: str,
    within_s: float = 5.0,
    code: int | None = None,
    not_before_s: float = 0.0,
    stalled: bool | None = None,
) -> None:
    """The server dropped the client's session (heartbeat timeout, close, kill).

    `not_before_s`: the drop must not come earlier than this (e.g. only the heartbeat).
    `stalled`: whether the write-stall watchdog (not the keepalive or a close) dropped it.
    """
    want = {"device": client, **({"code": str(code)} if code is not None else {})}
    if stalled is not None:
        want["stalled"] = str(stalled).lower()
    started = time.monotonic()
    await _expect(ctx, SERVER, {"DISCONNECTED"}, within_s, fields=want, what="DISCONNECTED")
    elapsed = time.monotonic() - started
    assert elapsed >= not_before_s, f"dropped after {elapsed:.1f} s, before {not_before_s} s"


@step("client_reconnects_within")
async def client_reconnects_within(ctx: ScenarioContext, client: str, seconds: float) -> None:
    """A new welcome reaches the client within `seconds`, and the server has its new session."""
    name = _client_name(client)
    welcome = await _expect(ctx, name, {"WELCOME"}, seconds, what="a new WELCOME")
    await _expect(
        ctx,
        SERVER,
        {"CONNECTED"},
        5,
        fields={"device": client, "session": welcome.fields["session"]},
        what=f"CONNECTED {client}",
    )


@step("client_retries_with_backoff")
async def client_retries_with_backoff(
    ctx: ScenarioContext, client: str, min_retries: int = 1
) -> None:
    """Every retry delay the client printed follows the backoff: 0.5 s doubling to 10 s, ±20 %."""
    proc = ctx.processes.get(_client_name(client))
    retries = [
        line
        for i, text in enumerate(proc.lines)
        if (line := _parse(i, text)) is not None and line.tag == "RETRY"
    ]
    assert len(retries) >= min_retries, f"{len(retries)} retries, expected >= {min_retries}"
    for line in retries:
        attempt, delay = int(line.fields["attempt"]), float(line.fields["delay"])
        base = min(BACKOFF_MAX_S, BACKOFF_MIN_S * 2 ** (attempt - 1))
        low = max(BACKOFF_MIN_S, base * (1 - JITTER))
        high = min(BACKOFF_MAX_S, base * (1 + JITTER))
        assert low - 1e-3 <= delay <= high + 1e-3, (
            f"retry {attempt} waited {delay} s, outside [{low:.3f}, {high:.3f}]: {line.text}"
        )


@step("client_disconnects")
async def client_disconnects(
    ctx: ScenarioContext, client: str, within_s: float = 5.0, stalled: bool | None = None
) -> None:
    """The client's connection dropped (it keeps running and retries).

    `stalled`: whether its write-stall watchdog dropped it.
    """
    want = {} if stalled is None else {"stalled": str(stalled).lower()}
    await _expect(ctx, _client_name(client), {"CLOSED"}, within_s, fields=want, what="CLOSED")


# ---------------------------------------------------------------- floods and early leavers


async def _flood(ctx: ScenarioContext, process: str, command: str) -> None:
    await _type(ctx, process, command)
    line = await _expect(ctx, process, {"FLOOD-STARTED", "CONSOLE-ERROR"}, 10, what="FLOOD")
    assert line.tag == "FLOOD-STARTED", f"{process} did not start the flood: {line.text}"


@step("server_floods")
async def server_floods(
    ctx: ScenarioContext,
    client: str,
    kind: str = "out_pcm",
    count: int = 1_000_000,
    bytes: int = 1_048_576,
    stream: int = 1,
) -> None:
    """The server sends `count` frames to the client as fast as the link takes them."""
    command = f"flood {client} {kind} {count} bytes={bytes} stream={stream}"
    await _flood(ctx, SERVER, command)


@step("client_floods")
async def client_floods(
    ctx: ScenarioContext,
    client: str,
    kind: str = "jpeg_chunk",
    count: int = 1_000_000,
    bytes: int = 1_048_576,
    stream: int = 1,
) -> None:
    """The client sends `count` frames to the server as fast as the link takes them."""
    command = f"flood {kind} {count} bytes={bytes} stream={stream}"
    await _flood(ctx, _client_name(client), command)


@step("flood_stopped")
async def flood_stopped(ctx: ScenarioContext, process: str, within_s: float = 5.0) -> None:
    """The flood from `process` ended early because its link closed (it never finished)."""
    name = _process_name(process)
    line = await _expect(
        ctx, name, {"FLOOD-STOPPED", "FLOOD-DONE"}, within_s, what="the end of the flood"
    )
    assert line.tag == "FLOOD-STOPPED", f"the flood finished, so writes never backed up: {line}"


@step("peer_leaves_before_hello")
async def peer_leaves_before_hello(
    ctx: ScenarioContext, stage: Literal["tcp", "upgraded"] = "upgraded"
) -> None:
    """A real TCP peer connects to the server and hangs up before sending hello.

    `tcp`: right after the TCP connect. `upgraded`: after a real WebSocket upgrade with the
    dev token (HTTP 101), before any message.
    """
    link = _link(ctx)
    reader, writer = await asyncio.open_connection("127.0.0.1", link.port)
    try:
        if stage == "upgraded":
            key = base64.b64encode(os.urandom(16)).decode()
            request = (
                "GET /edge/v1 HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{link.port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                f"Authorization: Bearer {link.token}\r\n\r\n"
            )
            writer.write(request.encode())
            await writer.drain()
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            assert head.startswith(b"HTTP/1.1 101"), f"upgrade refused: {head!r}"
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


@step("server_output_clean")
async def server_output_clean(ctx: ScenarioContext) -> None:
    """The server printed no traceback or handler error so far."""
    bad = [
        line
        for line in ctx.processes.get(SERVER).lines
        if "Traceback" in line or "handler failed" in line or line.startswith("ERROR")
    ]
    assert not bad, "server output has errors:\n" + "\n".join(bad)
