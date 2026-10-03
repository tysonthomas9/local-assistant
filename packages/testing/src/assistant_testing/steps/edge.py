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
import itertools
import json
import math
import os
import signal
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
    energy_trigger_feed_only: bool = False,
    vad_end_ms: int | None = None,
    record: bool = False,
    listen: Literal["wake_word", "open_mic", "push_to_talk"] = "push_to_talk",
    config_only: bool = False,
    within_s: float = 60.0,
) -> None:
    """Start the real edge agent with a real body; it dials the link server console.

    `body: reachy` runs on the robot's machine (`where: edge_host`) from the synced checkout;
    on the simulated robot from this checkout, on this PC; on a macOS edge host inside
    "Reachy Edge.app" (the owner of the microphone and camera
    permission, see scripts/edge_app_run.sh). It needs the daemon (`start_reachy_daemon`).
    `wait` (default) waits until the body started and the link welcomed the agent.
    `energy_trigger_dbfs` opens a mic window on a loud enough voice (standing in for a
    wake-word engine) and `vad_end_ms` closes it after that much quiet once speech was heard.
    `energy_trigger_feed_only` arms that trigger only on fed golden audio (`/feed`): ordinary
    room sound on the real microphone then cannot open a window before the fed utterance (the
    real microphone still flows into the brain's follow-up windows). Both go in the timings.
    `listen` is the listening mode (`--listen`): `push_to_talk` here unless a scenario asks
    for `wake_word` (the product's default) or `open_mic` (see steps/listen.py); it goes in the
    timings. `record` writes each played speech stream to a WAV where the agent runs (read and
    removed by `recorded_reply_transcript_not_empty`).
    `config_only` starts it the way a user does: only the link's address and token are given,
    everything else (device id, body, listening mode, wake word, mic cap) comes from
    config/assistant.toml; `id` then only names the process and `body` is what to expect.
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
    args = ["-m", "assistant_edge"]
    if not config_only:
        args += ["--device-id", id, "--body", body, "--listen", listen]
        ctx.state.setdefault("timings", {})["listen_mode"] = listen
    args += ["--url", f"ws://127.0.0.1:{port}/edge/v1", "--token", link.token]
    if config_only and (energy_trigger_dbfs is not None or vad_end_ms is not None or record):
        raise AssertionError("config_only takes no other agent options")
    if energy_trigger_dbfs is not None:
        args += ["--energy-trigger-dbfs", str(energy_trigger_dbfs)]
        timings = ctx.state.setdefault("timings", {})
        timings["energy_trigger"] = {
            "dbfs": energy_trigger_dbfs,
            "armed_by": "fed audio only" if energy_trigger_feed_only else "the real microphone",
        }
    if energy_trigger_feed_only:
        if energy_trigger_dbfs is None:
            raise AssertionError("energy_trigger_feed_only needs energy_trigger_dbfs")
        args += ["--energy-trigger-feed-only"]
    if vad_end_ms is not None:
        args += ["--vad-end-ms", str(vad_end_ms)]
    if record:
        run = os.environ.get("ASSISTANT_TEST_RUN") or f"run-{os.getpid()}"
        args += ["--record-dir", f".recordings/{run}-{id}"]  # in the agent's checkout
    # The robot's microphone needs the macOS permission of Reachy Edge.app: run inside it.
    # The reachy body plays Pollen's recorded moves from the edge host's offline HF cache.
    hf = {k: edge_host_steps.DAEMON_ENV[k] for k in ("HF_HOME", "HF_HUB_OFFLINE")}
    argv = (
        edge_host_steps.app_argv(f"edge-{id}", args, env=hf, stdin=True)
        if in_app
        else [python, *args]
    )
    name = _client_name(id)
    if ssh is not None:
        await ctx.processes.start(
            name,
            argv,
            stdin=True,
            ssh=ssh,
            remote_cwd=f"{edge_host.EDGE_DIR}/src",
            env={"PYTHONUNBUFFERED": "1", **hf},
        )
    else:
        # The simulated robot's processes share its own HOME (no ~/.asoundrc of a real robot;
        # the watchdog's heartbeat next to the sim daemon's) and the offline HF cache.
        sim_side = ctx.sim and where == "edge_host"
        env = edge_host_steps.sim_robot(ctx).env if sim_side else {"PYTHONUNBUFFERED": "1"}
        await ctx.processes.start(name, argv, stdin=True, env=env)
        if sim_side:
            edge_host_steps.sim_robot(ctx).agents.add(name)
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
            assert _first(board.get("AEC_NUM_FARENDS")) == 1, f"AEC has no far end: {board}" + (
                f" (the board's DSP is wedged; {REBOOT_HINT})" if _wedged(board) else ""
            )
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


REBOOT_HINT = (
    "reboot the audio board, no motion: python -m reachy_mini.media.audio_control_utils"
    " REBOOT --values 1"
)


def _wedged(board: dict[str, Any]) -> bool:
    """Every AEC parameter unreadable: the XVF3800's DSP servicer stuck answering "retry"."""
    values = [v for k, v in board.items() if k.startswith(("AEC_", "PP_"))]
    return bool(values) and all(str(v).startswith("unreadable") for v in values)


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


@step("edge_printed")
async def edge_printed(
    ctx: ScenarioContext,
    client: str,
    tag: str,
    fields: dict[str, str] | None = None,
    within_s: float = 10.0,
) -> None:
    """The agent printed a `tag` line (with these fields) that no step consumed yet."""
    want = {k: str(v) for k, v in (fields or {}).items()}
    line = await _expect(ctx, _client_name(client), {tag}, within_s, fields=want, what=tag)
    print(line.text)


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
    """For `seconds`, no mic frame from the edge reaches the server (no open mic window).

    Once the server has received the edge's `vad {state: end}`, the window is closed: no frame
    may follow that line, so the check counts from it (not from when this step starts)."""
    server = ctx.processes.get(SERVER)
    start = len(server.lines) - 1
    ends = [
        line.index
        for line in _get_lines(server, "RECV")
        if line.fields.get("device") == client
        and line.fields.get("type") == "vad"
        and (line.payload or {}).get("state") == "end"
    ]
    if ends:
        start = min(start, ends[-1])
    await asyncio.sleep(seconds)
    frames = _mic_frames(ctx, client, after_index=start)
    assert not frames, f"{len(frames)} mic frames reached the server: {frames[0].text}"


# ---------------------------------------------------------------- robot motion


async def _robot_state(ctx: ScenarioContext) -> dict[str, Any]:
    return await edge_host_steps._get(
        f"{edge_host_steps._daemon_url(ctx)}/api/state/full",
        3,
    )


SAMPLE_PERIOD_S = 0.05
"""The state sampler's target period (20 Hz)."""
MIN_SAMPLE_HZ = 8.0
"""Below this average rate during a move the measurement is starved: the step fails as such."""
MAX_SAMPLE_GAP_S = 0.4
"""The longest allowed gap between two samples during a move (a nod lasts about a second)."""

_SAMPLER = """
import json, sys, time, urllib.request
url, period = sys.argv[1] + "/api/state/full", float(sys.argv[2])
while True:
    t = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            s = json.load(response)
        keep = ("head_pose", "antennas_position", "body_yaw", "control_mode")
        print("STATE " + json.dumps({"t": round(t, 3), **{k: s.get(k) for k in keep}}), flush=True)
    except Exception as exc:
        print("STATE-ERROR " + json.dumps({"t": round(t, 3), "error": repr(exc)}), flush=True)
    time.sleep(max(0.0, period - (time.monotonic() - t)))
"""


def check_sampling(samples: list[dict[str, Any]], seconds: float) -> None:
    """Fail when `samples` (with monotonic `t`) are too sparse to observe a move of `seconds`."""
    assert len(samples) >= 2, f"the state sampler got {len(samples)} samples: no measurement"
    rate = len(samples) / max(seconds, 1e-6)
    gap = max(b["t"] - a["t"] for a, b in itertools.pairwise(samples))
    starved = f"the measurement is starved: {len(samples)} samples in {seconds:.1f} s"
    assert rate >= MIN_SAMPLE_HZ, f"{starved} ({rate:.1f} Hz, want >= {MIN_SAMPLE_HZ})"
    assert gap <= MAX_SAMPLE_GAP_S, f"{starved} (a {gap:.2f} s gap, want <= {MAX_SAMPLE_GAP_S})"


async def _during_request(
    ctx: ScenarioContext, client: str, type: str, fields: dict[str, Any], within_s: float = 20
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Send a request while a sampler ON THE ROBOT'S MACHINE reads the daemon's state every
    50 ms (no SSH tunnel or PC load in the measurement path); returns the result and the
    samples taken between sending it and receiving its result, checked for density."""
    host = edge_host_steps.host_of(ctx)
    count = ctx.state["samplers"] = ctx.state.get("samplers", 0) + 1
    name = f"state-sampler-{count}"
    url = f"http://127.0.0.1:{edge_host_steps.DAEMON_PORT}"
    args = ["-c", _SAMPLER, url, str(SAMPLE_PERIOD_S)]
    if host.ssh is not None:
        python = f"{edge_host.REMOTE_VENV}/bin/python"
        sampler = await ctx.processes.start(
            name, [python, *args], ssh=host.ssh, remote_cwd=f"{edge_host.EDGE_DIR}/src"
        )
    else:
        sampler = await ctx.processes.start(name, [sys.executable, *args])
    try:
        await _expect(ctx, name, {"STATE"}, 15, what="the robot-side state sampler")
        result = await _request(ctx, client, type, fields, within_s)
        await asyncio.sleep(3 * SAMPLE_PERIOD_S)
    finally:
        await sampler.stop()
    samples = [line.payload for line in _get_lines(sampler, "STATE") if line.payload is not None]
    span = samples[-1]["t"] - samples[0]["t"] if samples else 0.0
    errors = _get_lines(sampler, "STATE-ERROR")
    print(f"sampler: {len(samples)} samples over {span:.1f} s, {len(errors)} errors")
    check_sampling(samples, span)
    return result, samples


def _remember_start(ctx: ScenarioContext, state: dict[str, Any]) -> None:
    ctx.state["robot_start"] = state
    pose = state["head_pose"]
    print(
        f"robot before: pitch {math.degrees(pose['pitch']):.1f} deg, antennas "
        f"{[round(math.degrees(a), 1) for a in state['antennas_position']]} deg, "
        f"motors {state.get('control_mode')}"
    )


def _rotation(pose: dict[str, float]) -> list[list[float]]:
    """The rotation matrix of the daemon's head pose (xyz Euler angles, radians)."""
    cx, sx = math.cos(pose["roll"]), math.sin(pose["roll"])
    cy, sy = math.cos(pose["pitch"]), math.sin(pose["pitch"])
    cz, sz = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    return [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ]


def head_turn_deg(a: dict[str, float], b: dict[str, float]) -> float:
    """The angle (degrees) of the rotation between two head poses, whatever its axis."""
    ra, rb = _rotation(a), _rotation(b)
    trace = sum(ra[k][i] * rb[k][i] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))


def move_samples(
    samples: list[dict[str, Any]], t_start: float, t_end: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The pose when the move started (the last sample at or before `t_start`) and the
    samples taken while it played, all on the robot machine's monotonic clock."""
    before = [s for s in samples if s["t"] <= t_start]
    during = [s for s in samples if t_start <= s["t"] <= t_end]
    assert before, f"no state sample before the move started (t={t_start})"
    assert during, f"no state sample while the move played ({t_start}..{t_end})"
    return before[-1], during


@step("robot_plays_emotion")
async def robot_plays_emotion(
    ctx: ScenarioContext,
    client: str,
    emotion: str,
    move: str,
    min_head_deg: float = 0.0,
    min_antenna_deg: float = 0.0,
) -> None:
    """The brain sends `express <emotion>`; the body plays Pollen's recorded `move` to its end
    (wake_up -> move -> goto_sleep for a robot at rest) and answers ok. Measured by a sampler
    next to the daemon, over exactly the samples taken while the move played (its MOTION line
    gives the start and end on the robot machine's monotonic clock), from the pose when it
    started: the head turned by at least `min_head_deg` (the angle of the rotation, whatever
    its axis) and an antenna by at least `min_antenna_deg`."""
    before = await _robot_state(ctx)
    _remember_start(ctx, before)
    result, samples = await _during_request(ctx, client, "express", {"name": emotion}, 60)
    assert result.get("ok") is True, f"express {emotion} failed: {result}"
    motion = [
        line
        for line in _get_lines(_agent(ctx, client), "MOTION")
        if line.fields.get("express") == emotion
    ]
    assert motion, f"the body printed no MOTION line for express {emotion}"
    played = motion[-1].payload or {}
    print(f"played: {json.dumps(played)} in {motion[-1].fields.get('took_s')} s")
    assert played.get("move") == move, f"played {played.get('move')!r}, want {move!r}"
    assert played.get("played_s", 0) >= played.get("duration_s", 1e9) - 0.05, (
        f"{move} did not run to its end: {played}"
    )
    trace = [
        f"{math.degrees(s['head_pose']['pitch']):.0f}/{math.degrees(s['head_pose']['roll']):.0f}"
        for s in samples[::5]
    ]
    print(f"pitch/roll every 5th sample (deg): {' '.join(trace)}")
    ref, during = move_samples(samples, played["t_start"], played["t_end"])
    check_sampling(during, played["t_end"] - played["t_start"])
    head = max(head_turn_deg(ref["head_pose"], s["head_pose"]) for s in during)
    antenna = max(
        abs(math.degrees(s["antennas_position"][i] - ref["antennas_position"][i]))
        for s in during
        for i in (0, 1)
    )
    print(
        f"{move}: head turned {head:.1f} deg, antennas {antenna:.1f} deg over {len(during)} samples"
    )
    assert head >= min_head_deg, f"the head moved only {head:.1f} deg (want {min_head_deg})"
    assert antenna >= min_antenna_deg, (
        f"the antennas moved only {antenna:.1f} deg (want {min_antenna_deg})"
    )


def _rest_pose(state: dict[str, Any]) -> dict[str, float]:
    """Head roll/pitch, head yaw relative to the body, and the antennas, in degrees."""
    pose = state["head_pose"]
    return {
        "roll": math.degrees(pose["roll"]),
        "pitch": math.degrees(pose["pitch"]),
        "yaw_to_body": math.degrees(pose["yaw"] - (state.get("body_yaw") or 0.0)),
        "antenna0": math.degrees(state["antennas_position"][0]),
        "antenna1": math.degrees(state["antennas_position"][1]),
    }


@step("robot_back_at_rest")
async def robot_back_at_rest(
    ctx: ScenarioContext, head_deg: float = 4.0, antenna_deg: float = 5.0
) -> None:
    """After the move the robot is in the pose it started in (for a robot at rest: the sleep
    pose, after `goto_sleep()`) within `head_deg` / `antenna_deg`, and the motors are in the
    mode they had before (disabled at rest). The head's yaw is compared relative to the body,
    since `goto_sleep()` may also turn the body back straight. With the motors off the head
    settles passively into the sleep pose: measured on the real robot its pitch there varies
    between about 25 and 28 degrees from one rest to the next (the antennas within 0.2), hence
    4 degrees for the head; the sleep pose is about 25 degrees from neutral."""
    before = ctx.state.get("robot_start")
    assert before is not None, "no move measured; use robot_plays_emotion first"
    after = await _robot_state(ctx)
    start, end = _rest_pose(before), _rest_pose(after)
    print(
        "pose before -> after (deg): "
        + ", ".join(f"{k} {start[k]:.1f} -> {end[k]:.1f}" for k in start)
        + f", body yaw {math.degrees(before.get('body_yaw') or 0):.1f}"
        + f" -> {math.degrees(after.get('body_yaw') or 0):.1f}"
    )
    for key, limit in (("roll", head_deg), ("pitch", head_deg), ("yaw_to_body", head_deg)):
        diff = abs(end[key] - start[key])
        assert diff <= limit, f"head {key} is {diff:.1f} deg off its start pose"
    for key in ("antenna0", "antenna1"):
        diff = abs(end[key] - start[key])
        assert diff <= antenna_deg, f"{key} is {diff:.1f} deg off its start"
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


@step("edge_agent_crashes")
async def edge_agent_crashes(ctx: ScenarioContext, client: str) -> None:
    """The agent dies at once (kill -9 of its Python; inside Reachy Edge.app, of the app
    job's process group), so none of its own cleanup runs."""
    proc = _agent(ctx, client)
    pgid = edge_host_steps.app_job_pgid(proc)
    if pgid is None:
        proc.send_signal(signal.SIGKILL)
    else:
        host = edge_host_steps.host_of(ctx)
        await asyncio.to_thread(host.run, f"kill -KILL -- -{pgid}", 30)
    ctx.state["crashed_at"] = time.monotonic()
    code = await proc.wait(60)
    print(f"edge {client} killed (exit {code})")


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
