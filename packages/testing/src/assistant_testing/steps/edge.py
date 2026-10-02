"""Edge-agent steps: the real edge agent (`python -m assistant_edge`) with a real body.

The agent is a link client like the client console: it dials the link server console from
`steps/link.py`, and its process is named `client:<id>`, so the link steps (server_sends,
server_receives, kill_process, client_reconnects_within, ...) work on it too. Bodies are
real: `console` (prints what it would do, plays into a paced clock with no sound device) or
`reachy` (the real robot through the reachy-mini SDK, on the robot's machine).

Robot steps read the real robot state from the daemon's API (`/api/state/full`) while the
robot moves; every move is the arbiter's small, slow, bounded move (head at most 10 degrees,
antennas at most 20) that returns to the start pose and leaves the motors as they were.
"""

import asyncio
import json
import math
import sys
import time
from typing import Any, Literal

from assistant_testing import edge_host
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.processes import ManagedProcess, RemoteProcess
from assistant_testing.steps import edge_host as edge_host_steps
from assistant_testing.steps.link import (
    SERVER,
    _client_name,
    _expect,
    _get_lines,
    _Line,
    _link,
    _type,
)

DAEMON_LOOPBACK_PORTS = (8000, 8443)
LOOPBACK_HOSTS = ("127.0.0.1", "[::1]", "localhost")


def _agent(ctx: ScenarioContext, client: str) -> ManagedProcess:
    return ctx.processes.get(_client_name(client))


async def _server_command(ctx: ScenarioContext, command: str, tag: str, **want: str) -> _Line:
    await _type(ctx, SERVER, command)
    line = await _expect(ctx, SERVER, {tag, "CONSOLE-ERROR"}, 15, what=f"{tag} for {command!r}")
    if line.tag == "CONSOLE-ERROR":
        raise AssertionError(f"the server console refused {command!r}: {line.payload}")
    for key, value in want.items():
        if line.fields.get(key) != value:
            raise AssertionError(f"{line.text!r}: {key} is not {value!r}")
    return line


async def _request(
    ctx: ScenarioContext, client: str, type: str, fields: dict[str, Any], within_s: float
) -> dict[str, Any]:
    """The server sends request `type`; returns the edge's `result` for it (matched by id)."""
    sent = await _server_command(
        ctx, f"send {client} {type} {json.dumps(fields)}", "SENT", device=client, type=type
    )
    assert sent.payload is not None, sent.text
    result = await _expect(
        ctx,
        SERVER,
        {"RECV"},
        within_s,
        fields={"device": client, "type": "result"},
        payload={"re": sent.payload["id"]},
        what=f"the result of {type}",
    )
    assert result.payload is not None
    print(f"result of {type}: {json.dumps(result.payload)}")
    return result.payload


# ---------------------------------------------------------------- the agent


@step("start_edge_agent")
async def start_edge_agent(
    ctx: ScenarioContext,
    id: str,
    body: Literal["console", "reachy"] = "console",
    where: Literal["pc", "edge_host"] = "pc",
    wait: bool = True,
    energy_trigger_dbfs: float | None = None,
    within_s: float = 60.0,
) -> None:
    """Start the real edge agent with a real body; it dials the link server console.

    `body: reachy` runs on the robot's machine (`where: edge_host`) from the synced checkout;
    on a macOS edge host inside "Reachy Edge.app" (the owner of the microphone and camera
    permission, see scripts/edge_app_run.sh). It needs the daemon (`start_reachy_daemon`).
    `wait` (default) waits until the body started and the link welcomed the agent.
    """
    link = _link(ctx)
    if body == "reachy" and where != "edge_host":
        raise AssertionError("the reachy body runs where the robot is: use where: edge_host")
    port, python, ssh = link.port, sys.executable, None
    in_app = False
    if where == "edge_host":
        host = edge_host_steps.host_of(ctx)
        if host.ssh is not None:
            if ctx.state.get("edge_sha") is None:
                raise AssertionError(
                    "code not synced to the edge host; use code_synced_to_edge_host"
                )
            port = await edge_host_steps.reverse_tunnel(ctx, link.port)
            python, ssh = f"{edge_host.REMOTE_VENV}/bin/python", host.ssh
            in_app = body == "reachy" and await edge_host_steps.is_mac(host)
    args = ["-m", "assistant_edge", "--device-id", id, "--body", body]
    args += ["--url", f"ws://127.0.0.1:{port}/edge/v1", "--token", link.token]
    if energy_trigger_dbfs is not None:
        args += ["--energy-trigger-dbfs", str(energy_trigger_dbfs)]
    # The robot's microphone needs the macOS permission of Reachy Edge.app: run inside it.
    argv = edge_host_steps.app_argv(f"edge-{id}", args, stdin=True) if in_app else [python, *args]
    name = _client_name(id)
    if ssh is not None:
        await ctx.processes.start(
            name,
            argv,
            stdin=True,
            ssh=ssh,
            remote_cwd=f"{edge_host.EDGE_DIR}/src",
            env={"PYTHONUNBUFFERED": "1"},
        )
    else:
        await ctx.processes.start(name, argv, stdin=True, env={"PYTHONUNBUFFERED": "1"})
    if wait:
        line = await _expect(ctx, name, {"BODY"}, within_s, what="BODY (the body started)")
        if line.fields.get("kind") != body:
            raise AssertionError(f"started body {line.fields.get('kind')!r}, want {body!r}")
        await _expect(ctx, name, {"WELCOME"}, within_s, what="WELCOME")


@step("edge_body_is")
async def edge_body_is(
    ctx: ScenarioContext,
    client: str,
    aec: Literal["none", "sw", "hw"] | None = None,
    camera: bool | None = None,
    expressions: list[str] | None = None,
) -> None:
    """The capabilities the edge body announced (BODY line and the hello the server got).

    For the reachy body the agent also prints the XVF3800 echo-canceller state (AEC line);
    `aec: hw` checks that the board was found and its AEC runs on the speaker reference.
    """
    lines = _get_lines(_agent(ctx, client), "BODY")
    assert lines, f"edge {client} printed no BODY line"
    caps = lines[-1].payload or {}
    print(f"body capabilities: {json.dumps(caps)}")
    if aec is not None:
        got = (caps.get("audio_in") or {}).get("aec")
        assert got == aec, f"aec {got!r}, want {aec!r}"
        if aec == "hw":
            reports = _get_lines(_agent(ctx, client), "AEC")
            assert reports, "the reachy body printed no AEC line"
            board = reports[-1].payload or {}
            print(f"XVF3800 echo canceller: {json.dumps(board)}")
            assert board.get("board") == "xvf3800", f"no XVF3800 board: {board}"
            assert _first(board.get("AEC_NUM_FARENDS")) == 1, f"AEC has no far end: {board}"
    if camera is not None:
        assert (caps.get("camera") is not None) == camera, f"camera {caps.get('camera')}"
    if expressions is not None:
        got = (caps.get("motion") or {}).get("expressions") or []
        missing = set(expressions) - set(got)
        assert not missing, f"expressions {got} miss {sorted(missing)}"
    await _expect(
        ctx,
        SERVER,
        {"CONNECTED"},
        5,
        fields={"device": client},
        payload={"type": "hello", "device_id": client},
        what=f"CONNECTED {client}",
    )


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


@step("edge_types")
async def edge_types(ctx: ScenarioContext, client: str, text: str) -> None:
    """Type a line into the agent's stdin (`/ptt down`, `/mute`, ... or text for text.input)."""
    await _type(ctx, _client_name(client), text)


@step("press_push_to_talk")
async def press_push_to_talk(ctx: ScenarioContext, client: str) -> None:
    """Push-to-talk down: the agent opens a mic window (MIC-OPEN) and starts the uplink."""
    name = _client_name(client)
    await _type(ctx, name, "/ptt down")
    line = await _expect(
        ctx, name, {"MIC-OPEN", "MIC-REFUSED", "CONSOLE-ERROR"}, 10, what="MIC-OPEN"
    )
    assert line.tag == "MIC-OPEN", f"no mic window: {line.text}"


@step("release_push_to_talk")
async def release_push_to_talk(ctx: ScenarioContext, client: str) -> None:
    """Push-to-talk up: the window closes (MIC-CLOSE) and the uplink stops."""
    name = _client_name(client)
    await _type(ctx, name, "/ptt up")
    await _expect(ctx, name, {"MIC-CLOSE"}, 10, fields={"reason": "ptt"}, what="MIC-CLOSE")


@step("edge_output_clean")
async def edge_output_clean(ctx: ScenarioContext, client: str) -> None:
    """The agent printed no traceback, console error or unhandled message type so far."""
    bad = [
        line
        for line in _agent(ctx, client).lines
        if "Traceback" in line or line.startswith(("CONSOLE-ERROR", "UNHANDLED", "BODY-ERROR"))
    ]
    assert not bad, f"edge {client} output has errors:\n" + "\n".join(bad)


@step("edge_answers")
async def edge_answers(
    ctx: ScenarioContext,
    client: str,
    type: str,
    fields: dict[str, Any] | None = None,
    ok: bool = True,
    error_contains: str | None = None,
    within_s: float = 10.0,
) -> None:
    """The server sends request `type`; the edge answers with `result{ok}` for it."""
    result = await _request(ctx, client, type, fields or {}, within_s)
    assert result.get("ok") is ok, f"result ok={result.get('ok')}, want {ok}: {result}"
    if error_contains is not None:
        assert error_contains in (result.get("error") or ""), f"error: {result.get('error')!r}"


# ---------------------------------------------------------------- audio


@step("server_streams_speech")
async def server_streams_speech(
    ctx: ScenarioContext,
    client: str,
    stream: int,
    clip: str,
    rate: int = 24000,
    text: str | None = None,
) -> None:
    """The server streams speech like the brain: speak.begin, 20 ms 0x02 frames, speak.end.

    `clip`: `tone:<hz>:<seconds>` (a sine at -12 dBFS) or a 16-bit mono WAV path.
    """
    command = f"stream {client} {stream} {clip} rate={rate}"
    if text:
        command += f" text={text}"
    await _server_command(ctx, command, "STREAMED", device=client, stream=str(stream))


@step("playback_reported")
async def playback_reported(
    ctx: ScenarioContext,
    client: str,
    stream: int,
    state: Literal["started", "progress", "done", "flushed"],
    min_played_ms: int | None = None,
    max_played_ms: int | None = None,
    within_s: float = 10.0,
) -> None:
    """The server receives the edge's playback clock for `stream` in `state`.

    For `progress`, it waits for the first report with at least `min_played_ms`.
    """
    deadline = time.monotonic() + within_s
    while True:
        line = await _expect(
            ctx,
            SERVER,
            {"RECV"},
            max(0.1, deadline - time.monotonic()),
            fields={"device": client, "type": "playback"},
            payload={"stream_id": stream, "state": state},
            what=f"playback {state} for stream {stream}",
        )
        assert line.payload is not None
        played = line.payload["played_ms"]
        if state != "progress" or min_played_ms is None or played >= min_played_ms:
            break
    if min_played_ms is not None:
        assert played >= min_played_ms, f"played_ms {played} < {min_played_ms}"
    if max_played_ms is not None:
        assert played <= max_played_ms, f"played_ms {played} > {max_played_ms}"


@step("playback_paced")
async def playback_paced(ctx: ScenarioContext, client: str, stream: int, ms: int) -> None:
    """The edge played `stream` in real time: from `started` to `done` took about `ms`
    (at least 90 %, at most `ms` + 1.5 s), and `done` reports `ms` played (±5 %)."""
    server = ctx.processes.get(SERVER)
    times: dict[str, float] = {}
    played = 0
    for line in _get_lines(server, "RECV"):
        p = line.payload or {}
        if line.fields.get("device") != client or p.get("type") != "playback":
            continue
        if p.get("stream_id") == stream and p.get("state") in ("started", "done"):
            times.setdefault(p["state"], server.line_times[line.index])
            if p["state"] == "done":
                played = p["played_ms"]
    assert set(times) == {"started", "done"}, f"stream {stream}: playback states {sorted(times)}"
    took = (times["done"] - times["started"]) * 1000
    print(f"stream {stream}: started -> done in {took:.0f} ms, played_ms {played} (clip {ms} ms)")
    assert ms * 0.9 <= took <= ms + 1500, f"took {took:.0f} ms for a {ms} ms clip"
    assert abs(played - ms) <= ms * 0.05 + 40, f"played_ms {played} for a {ms} ms clip"


@step("barge_in_reported")
async def barge_in_reported(
    ctx: ScenarioContext,
    client: str,
    stream: int,
    min_played_ms: int = 0,
    max_played_ms: int | None = None,
    within_s: float = 5.0,
) -> None:
    """The server receives `vad{start, barge_in, stream_id, played_ms}` from the edge."""
    line = await _expect(
        ctx,
        SERVER,
        {"RECV"},
        within_s,
        fields={"device": client, "type": "vad"},
        payload={"state": "start", "barge_in": True, "stream_id": stream},
        what="vad barge_in",
    )
    assert line.payload is not None
    played = line.payload["played_ms"]
    print(f"barge-in: stream {stream} stopped after {played} ms played")
    assert played >= min_played_ms, f"played_ms {played} < {min_played_ms}"
    if max_played_ms is not None:
        assert played <= max_played_ms, f"played_ms {played} > {max_played_ms}"


@step("playback_stops_within")
async def playback_stops_within(
    ctx: ScenarioContext, client: str, stream: int, ms: float, local: bool = True
) -> None:
    """The edge flushed its speaker within `ms` (its FLUSHED line's took_ms), `local`ly on a
    barge-in (or on the brain's flush), and the server got `playback{flushed}` for `stream`."""
    name = _client_name(client)
    line = await _expect(
        ctx, name, {"FLUSHED"}, 5, fields={"local": str(local).lower()}, what="FLUSHED"
    )
    took = float(line.fields["took_ms"])
    print(f"speaker flushed in {took:.1f} ms ({line.text})")
    assert took <= ms, f"flush took {took:.1f} ms, more than {ms} ms"
    await playback_reported(ctx, client, stream, "flushed", within_s=5)


def _mic_frames(ctx: ScenarioContext, client: str, after_index: int = -1) -> list[_Line]:
    return [
        line
        for line in _get_lines(ctx.processes.get(SERVER), "FRAME")
        if line.index > after_index
        and line.fields.get("device") == client
        and line.fields.get("kind") == "0x01"
    ]


@step("uplink_audio_live")
async def uplink_audio_live(
    ctx: ScenarioContext,
    client: str,
    min_frames: int,
    above_dbfs: float = -100.0,
    within_s: float = 10.0,
) -> None:
    """The server received at least `min_frames` 20 ms mic frames from the edge, and they
    carry a live signal: the loudest is above `above_dbfs` and the level varies (a denied or
    muted microphone delivers digital silence, -120 dBFS)."""
    deadline = time.monotonic() + within_s
    frames = _mic_frames(ctx, client)
    while len(frames) < min_frames and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
        frames = _mic_frames(ctx, client)
    levels = [float(f.fields["dbfs"]) for f in frames]
    assert len(levels) >= min_frames, f"{len(levels)} mic frames, want >= {min_frames}"
    mean = sum(levels) / len(levels)
    print(
        f"uplink: {len(levels)} frames, mean {mean:.1f} dBFS, "
        f"min {min(levels):.1f}, max {max(levels):.1f}"
    )
    assert max(levels) > above_dbfs, f"loudest frame {max(levels):.1f} dBFS: no live signal"
    assert max(levels) - min(levels) > 0.5, "the level never changes: not a live microphone"


@step("uplink_carries_no_audio")
async def uplink_carries_no_audio(ctx: ScenarioContext, client: str, seconds: float) -> None:
    """For `seconds`, no mic frame from the edge reaches the server (no open mic window)."""
    server = ctx.processes.get(SERVER)
    start = len(server.lines) - 1
    await asyncio.sleep(seconds)
    frames = _mic_frames(ctx, client, after_index=start)
    assert not frames, f"{len(frames)} mic frames reached the server: {frames[0].text}"


# ---------------------------------------------------------------- robot motion


async def _robot_state(ctx: ScenarioContext) -> dict[str, Any]:
    return await edge_host_steps._get(
        f"{edge_host_steps._daemon_url(ctx)}/api/state/full",
        3,
    )


async def _during_request(
    ctx: ScenarioContext, client: str, type: str, fields: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Send a request and sample the robot state every ~50 ms until its result arrives."""
    samples: list[dict[str, Any]] = []
    done = asyncio.Event()

    async def sample() -> None:
        while not done.is_set():
            samples.append(await _robot_state(ctx))
            await asyncio.sleep(0.05)

    sampler = asyncio.create_task(sample())
    try:
        result = await _request(ctx, client, type, fields, 20)
    finally:
        done.set()
        await sampler
    return result, samples


def _remember_start(ctx: ScenarioContext, state: dict[str, Any]) -> None:
    ctx.state["robot_start"] = state
    pose = state["head_pose"]
    print(
        f"robot before: pitch {math.degrees(pose['pitch']):.1f} deg, antennas "
        f"{[round(math.degrees(a), 1) for a in state['antennas_position']]} deg, "
        f"motors {state.get('control_mode')}"
    )


@step("robot_nods")
async def robot_nods(ctx: ScenarioContext, client: str, degrees: float) -> None:
    """The brain sends `express yes`; the real head pitches by about `degrees` (at most 10)
    and the edge answers ok. Measured from the daemon's state while it moves."""
    if not 0 < degrees <= 10:
        raise AssertionError("robot safety: a nod is at most 10 degrees")
    before = await _robot_state(ctx)
    _remember_start(ctx, before)
    result, samples = await _during_request(
        ctx, client, "express", {"name": "yes", "intensity": degrees / 10}
    )
    assert result.get("ok") is True, f"express yes failed: {result}"
    start = before["head_pose"]["pitch"]
    peak = max(abs(math.degrees(s["head_pose"]["pitch"] - start)) for s in samples)
    print(f"nod: peak pitch change {peak:.1f} deg over {len(samples)} samples")
    assert peak >= degrees * 0.5, f"the head pitched only {peak:.1f} deg for a {degrees} deg nod"
    assert peak <= degrees + 3, f"the head pitched {peak:.1f} deg, more than {degrees} + 3"


@step("antennas_wiggle")
async def antennas_wiggle(ctx: ScenarioContext, client: str, degrees: float) -> None:
    """The brain sends `express happy`; both real antennas swing by about `degrees` (at most
    20) and back, and the edge answers ok."""
    if not 0 < degrees <= 20:
        raise AssertionError("robot safety: an antenna wiggle is at most 20 degrees")
    before = await _robot_state(ctx)
    _remember_start(ctx, before)
    result, samples = await _during_request(
        ctx, client, "express", {"name": "happy", "intensity": degrees / 20}
    )
    assert result.get("ok") is True, f"express happy failed: {result}"
    for i in (0, 1):
        start = before["antennas_position"][i]
        peak = max(abs(math.degrees(s["antennas_position"][i] - start)) for s in samples)
        print(f"antenna {i}: peak change {peak:.1f} deg")
        assert peak >= degrees * 0.5, f"antenna {i} moved only {peak:.1f} deg"
        assert peak <= degrees + 5, f"antenna {i} moved {peak:.1f} deg, more than {degrees} + 5"


@step("robot_back_at_rest")
async def robot_back_at_rest(
    ctx: ScenarioContext, head_deg: float = 2.0, antenna_deg: float = 5.0
) -> None:
    """After the move: head and antennas are back where they started (within `head_deg` /
    `antenna_deg`), and the motors are in the mode they had before (disabled at rest)."""
    before = ctx.state.get("robot_start")
    assert before is not None, "no move measured; use robot_nods or antennas_wiggle first"
    after = await _robot_state(ctx)
    for key in ("roll", "pitch", "yaw"):
        diff = abs(math.degrees(after["head_pose"][key] - before["head_pose"][key]))
        assert diff <= head_deg, f"head {key} is {diff:.1f} deg off its start pose"
    for i in (0, 1):
        diff = abs(math.degrees(after["antennas_position"][i] - before["antennas_position"][i]))
        assert diff <= antenna_deg, f"antenna {i} is {diff:.1f} deg off its start"
    assert after.get("control_mode") == before.get("control_mode"), (
        f"motors {after.get('control_mode')!r}, were {before.get('control_mode')!r}"
    )
    print(f"robot at rest again, motors {after.get('control_mode')}")


# ---------------------------------------------------------------- camera


@step("robot_camera_frame")
async def robot_camera_frame(
    ctx: ScenarioContext, client: str, slot: int = 1, max_side: int = 640, min_bytes: int = 2000
) -> None:
    """The brain asks for a snapshot; the edge sends a real JPEG on `slot` (0x03 chunks) whose
    size fits `max_side`, and its result reports the same size."""
    result = await _request(ctx, client, "snapshot", {"slot": slot, "max_side": max_side}, 20)
    assert result.get("ok") is True, f"snapshot failed: {result}"
    data = result.get("data") or {}
    jpeg = await _expect(
        ctx,
        SERVER,
        {"JPEG"},
        10,
        fields={"device": client, "slot": str(slot)},
        what=f"a whole JPEG on slot {slot}",
    )
    width, height, size = (int(jpeg.fields[k]) for k in ("width", "height", "bytes"))
    print(f"camera frame: {width}x{height}, {size} bytes")
    assert size >= min_bytes, f"JPEG of {size} bytes: too small for a camera frame"
    assert 0 < max(width, height) <= max_side, f"{width}x{height} does not fit {max_side}"
    assert (data.get("width"), data.get("height"), data.get("bytes")) == (width, height, size), (
        f"result {data} does not match the received JPEG {width}x{height} {size} bytes"
    )


# ---------------------------------------------------------------- daemon health


@step("edge_body_lost")
async def edge_body_lost(ctx: ScenarioContext, client: str, within_s: float = 10.0) -> None:
    """The robot went away: the agent prints BODY-ERROR and the server gets
    `error{body_unavailable}`; the link itself stays up."""
    await _expect(ctx, _client_name(client), {"BODY-ERROR"}, within_s, what="BODY-ERROR")
    await _expect(
        ctx,
        SERVER,
        {"RECV"},
        5,
        fields={"device": client, "type": "error"},
        payload={"code": "body_unavailable"},
        what="error body_unavailable",
    )


@step("edge_body_recovers")
async def edge_body_recovers(ctx: ScenarioContext, client: str, within_s: float = 30.0) -> None:
    """The robot is back: the agent reconnected its body on its own (BODY-OK)."""
    await _expect(ctx, _client_name(client), {"BODY-OK"}, within_s, what="BODY-OK")


@step("daemon_ports_on_loopback")
async def daemon_ports_on_loopback(
    ctx: ScenarioContext, listening: list[int] | None = None
) -> None:
    """Every socket of the daemon's processes (lsof) is on loopback: no `*` (all interfaces)
    and no LAN address, at either end; it LISTENs on 127.0.0.1 at the HTTP API (8000) and
    the WebRTC signalling port (8443); nothing on mDNS (5353)."""
    host = edge_host_steps.host_of(ctx)
    daemon = ctx.processes.get(edge_host_steps.DAEMON)
    pgid = edge_host_steps.app_job_pgid(daemon)  # inside Reachy Edge.app (macOS)
    if pgid is None:
        pgid = daemon.remote_pgid if isinstance(daemon, RemoteProcess) else daemon.proc.pid
    assert pgid is not None, "the daemon's process group is unknown"
    assert daemon.running, "the daemon this scenario started is not running"
    # -F: machine-readable fields (p pid, P protocol, n name, T tcp state); no user column.
    done = await asyncio.to_thread(host.run, f"lsof -nP -a -g {pgid} -i -F pPnT", 30)
    sockets: list[tuple[str, str, str]] = []
    proto = state = ""
    for raw in done.stdout.splitlines():
        tag, value = raw[:1], raw[1:]
        if tag == "P":
            proto, state = value, ""
        elif tag == "T" and value.startswith("ST="):
            state = value[3:]
            if sockets:
                sockets[-1] = (*sockets[-1][:2], state)
        elif tag == "n":
            sockets.append((proto, value, state))
    assert sockets, f"lsof found no sockets of the daemon (exit {done.returncode}): {done.stderr}"
    print(f"daemon sockets (lsof, process group of the daemon on {host.label}):")
    bad: list[str] = []
    for proto, name, state in sockets:
        print(f"  {proto} {name} {state}")
        for end in name.split("->"):
            address = end.rsplit(":", 1)[0]
            if address not in LOOPBACK_HOSTS:
                bad.append(f"{proto} {name}")
        if name.endswith(":5353"):
            bad.append(f"{proto} {name} (mDNS)")
    assert not bad, "daemon sockets not on loopback:\n" + "\n".join(bad)
    listens = {name for proto, name, state in sockets if proto == "TCP" and state == "LISTEN"}
    for port in listening or list(DAEMON_LOOPBACK_PORTS):
        assert f"127.0.0.1:{port}" in listens, f"no LISTEN on 127.0.0.1:{port}: {sorted(listens)}"


@step("edge_agent_fails")
async def edge_agent_fails(
    ctx: ScenarioContext, client: str, contains: str, within_s: float = 60.0
) -> None:
    """The agent refused to start: it exits non-zero, its output says `contains`, and it never
    connected (so nothing it would have driven ran)."""
    proc = _agent(ctx, client)
    code = await proc.wait(within_s)
    print(f"edge {client} exited with {code}")
    assert code not in (0, None), f"edge {client} exited with {code}, want a failure"
    assert contains in proc.output, f"edge {client} output lacks {contains!r}:\n{proc.output}"
    assert not _get_lines(proc, "WELCOME"), f"edge {client} connected anyway"


@step("client_stayed_connected")
async def client_stayed_connected(ctx: ScenarioContext, client: str) -> None:
    """The client's EdgeLink never dropped: one welcome, no close, no retry."""
    proc = _agent(ctx, client)
    welcomes = _get_lines(proc, "WELCOME")
    drops = _get_lines(proc, "CLOSED") + _get_lines(proc, "RETRY")
    assert len(welcomes) == 1, f"{len(welcomes)} welcomes: the link was re-established"
    assert not drops, f"the link dropped: {drops[0].text}"
