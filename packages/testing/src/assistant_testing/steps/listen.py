"""Hands-free listening steps: wake word, open mic, follow-up windows (`assistant_edge.listen`).

The edge agent prints what its listening does: WAKE (a wake word, its score), WAKE-NEAR (a
score under the threshold), MIC-OPEN reason=wake|vad|follow_up, MIC-CLOSE reason=vad_end|
no_speech|timeout, FOLLOW-UP (the brain's follow-up armed) and SMART-TURN (Smart Turn's verdict
on a pause: is the turn over?). These steps read those lines after a mark: the start of the
last utterance (`feed_golden_wav` or `speaker_plays`) or of a `listen_mark`.

`speaker_plays` is the real-speaker voice input of the robot tests: a golden WAV played by
the edge host's own speaker (the Mac mini's built-in speaker, afplay) next to the robot, so
the robot's microphone hears it through the air. It never touches the robot's speaker: it
refuses unless the default output is a built-in speaker, sets that speaker's volume only for
the play and restores the volume and mute exactly as they were.
"""

import asyncio
import contextlib
import shlex
import time
import wave
from pathlib import Path
from typing import Any, Literal

from assistant_core.levels import dbfs
from assistant_testing import edge_host
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.steps import edge_host as edge_host_steps
from assistant_testing.steps.brain import brain_state_is
from assistant_testing.steps.link import _client_name, _expect, _get_lines, _Line, _link
from assistant_testing.steps.speech import GOLDEN_DIR, _timings


def _mark(ctx: ScenarioContext, client: str) -> int:
    """Index of the agent's first line after the last utterance or mark."""
    return int(ctx.state.get("listen_mark", {}).get(client, 0))


def _set_mark(ctx: ScenarioContext, client: str) -> int:
    index = len(ctx.processes.get(_client_name(client)).lines)
    ctx.state.setdefault("listen_mark", {})[client] = index
    return index


def _lines_since(ctx: ScenarioContext, client: str, tag: str) -> list[_Line]:
    mark = _mark(ctx, client)
    return [x for x in _get_lines(ctx.processes.get(_client_name(client)), tag) if x.index >= mark]


@step("listen_mark")
async def listen_mark(ctx: ScenarioContext, client: str) -> None:
    """Later listening checks look only at what the agent prints from now on."""
    _set_mark(ctx, client)


@step("edge_listens")
async def edge_listens(ctx: ScenarioContext, client: str, mode: str) -> None:
    """The agent runs listening mode `mode` (its LISTEN line); recorded in the timings."""
    lines = _get_lines(ctx.processes.get(_client_name(client)), "LISTEN")
    assert lines, f"edge {client} printed no LISTEN line"
    fields = lines[-1].fields
    print(lines[-1].text)
    assert fields.get("mode") == mode, f"listening mode {fields.get('mode')!r}, want {mode!r}"
    _timings(ctx)["listen"] = dict(fields)


@step("wake_detected")
async def wake_detected(
    ctx: ScenarioContext,
    client: str,
    model: str = "hey_jarvis",
    word: str = "hey jarvis",
    min_score: float = 0.4,
    within_s: float = 15.0,
) -> None:
    """The edge's wake-word engine heard `model` (WAKE, score at least `min_score`), sent
    `wake{word, score}` with the spoken `word` and opened a mic window for it. The score and
    any near misses go in the timings."""
    agent = _client_name(client)
    mark = _mark(ctx, client)
    line = await _expect(ctx, agent, {"WAKE"}, within_s, what=f"WAKE {model}")
    assert line.index >= mark, f"the wake came before the utterance: {line.text}"
    print(line.text)
    got = line.fields.get("model")
    assert got == model, f"woke on {got!r}, want {model!r}"
    sent = await _expect(ctx, agent, {"SENT"}, 5, fields={"type": "wake"}, what="SENT wake")
    assert sent.index > line.index, f"a wake was sent before this one: {sent.text}"
    payload = sent.payload or {}
    print(sent.text)
    assert payload.get("word") == word, f"sent wake word {payload.get('word')!r}, want {word!r}"
    score = float(payload.get("score") or 0.0)
    assert score >= min_score, f"wake score {score} < {min_score}"
    await _expect(ctx, agent, {"MIC-OPEN"}, 5, fields={"reason": "wake"}, what="MIC-OPEN wake")
    near = [x.text for x in _lines_since(ctx, client, "WAKE-NEAR")]
    ctx.state["wake_word"] = word  # the next transcript must not carry it
    _timings(ctx).setdefault("wake", []).append(
        {"word": word, "score": score, "engine": line.fields.get("engine"), "near_misses": near}
    )


@step("mic_opened")
async def mic_opened(
    ctx: ScenarioContext,
    client: str,
    reason: str,
    barge_in: bool | None = None,
    limit_s: float | None = None,
    within_s: float = 15.0,
) -> None:
    """The edge opened a mic window for `reason` (wake, vad, follow_up, ptt, energy) after the
    mark; `barge_in`: whether it cut the robot's speech; `limit_s`: its time limit."""
    agent = _client_name(client)
    want = {"reason": reason, **({} if barge_in is None else {"barge_in": str(barge_in).lower()})}
    if limit_s is not None:
        want["limit_s"] = f"{limit_s:g}"
    try:
        line = await _expect(
            ctx, agent, {"MIC-OPEN"}, within_s, fields=want, what=f"MIC-OPEN {want}"
        )
    except AssertionError as error:
        raise AssertionError(f"{error}\n{await _speaker_outcome(ctx)}") from None
    assert line.index >= _mark(ctx, client), f"the window opened before the mark: {line.text}"
    print(line.text)


async def _speaker_outcome(ctx: ScenarioContext) -> str:
    """What the edge host's speaker did (a failed check says whether the sound was played)."""
    playing: dict[str, Any] | None = ctx.state.get("speaker")
    if playing is None:
        return "the speaker played nothing"
    try:
        output = await asyncio.wait_for(asyncio.shield(playing["task"]), 30)
    except Exception as error:  # reported; the check fails anyway
        return f"the speaker failed: {error}"
    return f"the speaker played {playing['name']}.wav: {' '.join(output.split())}"


@step("mic_closed")
async def mic_closed(
    ctx: ScenarioContext,
    client: str,
    reason: str = "vad_end",
    opened_by: str | None = None,
    min_s: float = 0.0,
    max_s: float | None = None,
    open_for_s: float | None = None,
    within_s: float = 30.0,
) -> None:
    """The edge's mic window closed for `reason` (vad_end, no_speech, timeout, ...), having
    carried at least `min_s` (and at most `max_s`) of audio; `open_for_s`: it was open that
    long (+-2 s, from when the agent printed MIC-OPEN and MIC-CLOSE)."""
    agent = _client_name(client)
    line = await _expect(ctx, agent, {"MIC-CLOSE"}, within_s, what="MIC-CLOSE")
    print(line.text)
    assert line.fields.get("reason") == reason, f"closed {line.fields.get('reason')!r}: {line.text}"
    if opened_by is not None:
        assert line.fields.get("opened_by") == opened_by, line.text
    seconds = int(line.fields.get("frames", "0")) * 0.02
    assert seconds >= min_s, f"the window carried {seconds:.2f} s of audio (want >= {min_s})"
    if max_s is not None:
        assert seconds <= max_s, f"the window carried {seconds:.2f} s of audio (want <= {max_s})"
    if open_for_s is not None:
        proc = ctx.processes.get(agent)
        opened = [x for x in _get_lines(proc, "MIC-OPEN") if x.index < line.index][-1]
        lasted = proc.line_times[line.index] - proc.line_times[opened.index]
        print(f"the window was open {lasted:.1f} s")
        assert abs(lasted - open_for_s) <= 2, f"open {lasted:.1f} s, want {open_for_s} s"


@step("turn_end_judged")
async def turn_end_judged(
    ctx: ScenarioContext, client: str, complete: bool, within_s: float = 30.0
) -> None:
    """Smart Turn judged a pause after the mark (SMART-TURN): the turn sounded `complete` or
    not. Each verdict is consumed once; the verdicts go in the timings."""
    agent = _client_name(client)
    line = await _expect(ctx, agent, {"SMART-TURN"}, within_s, what="SMART-TURN")
    print(line.text)
    assert line.index >= _mark(ctx, client), f"judged before the mark: {line.text}"
    got = line.fields.get("complete") == "true"
    assert got == complete, f"Smart Turn found the turn complete={got}, want {complete}"
    _timings(ctx).setdefault("smart_turn", []).append(dict(line.fields))


@step("no_turn_for")
async def no_turn_for(ctx: ScenarioContext, client: str, seconds: float) -> None:
    """For `seconds` from now nothing the edge heard since the mark started a turn: no mic
    window opened, no wake was sent. Prints the near misses of the wake word meanwhile."""
    await asyncio.sleep(seconds)
    opened = _lines_since(ctx, client, "MIC-OPEN")
    woke = _lines_since(ctx, client, "WAKE")
    for line in _lines_since(ctx, client, "WAKE-NEAR"):
        print(f"  edge: {line.text}")
    started = "; ".join(x.text for x in woke + opened)
    assert not started, f"a turn started: {started}"
    print(f"no wake and no mic window for {seconds} s")


@step("follow_up_armed")
async def follow_up_armed(ctx: ScenarioContext, client: str, within_s: float = 30.0) -> None:
    """The brain's `mic.follow_up` armed the edge: speech may now start a turn without the
    wake word (FOLLOW-UP). Marks: later checks look from here."""
    line = await _expect(ctx, _client_name(client), {"FOLLOW-UP"}, within_s, what="FOLLOW-UP")
    print(line.text)
    _set_mark(ctx, client)


@step("room_quiet")
async def room_quiet(
    ctx: ScenarioContext, client: str, seconds: float = 2.0, within_s: float = 90.0
) -> None:
    """Wait until the room has no speech for `seconds` (the edge's `/level`: its speech
    detector took no frame for speech), so that someone talking elsewhere in the room cannot
    start or spoil the turn that comes next. A turn that room speech started meanwhile (open
    mic, or a real "hey jarvis") is waited out (its window closed, the brain idle again) and
    set aside: later checks do not take its lines for the next turn's. A precondition, not a
    check of the listening: fails only if the room is never quiet within `within_s`. The waits
    go in the timings."""
    agent = _client_name(client)
    started, tries = time.monotonic(), 0
    while True:
        tries += 1
        await ctx.processes.get(agent).write_line(f"/level {seconds}")
        line = await _expect(ctx, agent, {"LEVEL", "CONSOLE-ERROR"}, seconds + 15, what="LEVEL")
        assert line.tag == "LEVEL", f"the agent did not measure the room: {line.text}"
        assert "speech_ms" in line.fields, "the agent runs no speech detector (push_to_talk?)"
        if await _room_turn_set_aside(ctx, client):
            print("  room speech started a turn: waiting for the brain to be idle again")
            with contextlib.suppress(AssertionError):
                await brain_state_is(ctx, client, "idle", within_s=60)
            continue
        if int(line.fields["speech_ms"]) == 0:
            break
        print(f"  room not quiet: {line.text}")
        if time.monotonic() - started > within_s:
            raise AssertionError(f"the room never had {seconds} s without speech in {within_s} s")
    waited = round(time.monotonic() - started, 1)
    _timings(ctx).setdefault("room_quiet_waits_s", []).append(waited)
    print(f"room quiet for {seconds} s ({line.text}) after {waited} s, {tries} measurement(s)")


async def _room_turn_set_aside(ctx: ScenarioContext, client: str) -> bool:
    """Set aside (consume) the wake and window lines nobody looked at yet, waiting for an open
    window to close; whether there were any."""
    agent = _client_name(client)
    proc = ctx.processes.get(agent)
    consumed = _link(ctx).consumed.setdefault(agent, set())
    stray = [
        x
        for tag in ("WAKE", "MIC-OPEN", "MIC-CLOSE")
        for x in _get_lines(proc, tag)
        if x.index not in consumed
    ]
    for x in sorted(stray, key=lambda x: x.index):
        consumed.add(x.index)
        print(f"  room sound, set aside: {x.text}")
    opened = [x.index for x in _get_lines(proc, "MIC-OPEN")]
    closed = [x.index for x in _get_lines(proc, "MIC-CLOSE")]
    if opened and (not closed or closed[-1] < opened[-1]):
        line = await _expect(ctx, agent, {"MIC-CLOSE"}, 150, what="the room's window to close")
        print(f"  room sound, set aside: {line.text}")
    return bool(stray)


# ---------------------------------------------------------------- a real speaker

_BUILT_IN = "Mac mini Speakers"

_PLAY = """
set -u
cd "$HOME"
out=$(system_profiler SPAudioDataType 2>/dev/null | awk '
  /^        [^ ].*:$/ {{ name=$0; sub(/^ +/, "", name); sub(/:$/, "", name) }}
  /Default Output Device: Yes/ {{ print name }}')
if [ "$out" != {device} ]; then
  echo "SPEAKER-REFUSED default output is '$out', not the built-in speaker"; exit 3
fi
volume=$(osascript -e 'output volume of (get volume settings)')
muted=$(osascript -e 'output muted of (get volume settings)')
restore() {{
  osascript -e "set volume output volume $volume"
  if [ "$muted" = true ]; then osascript -e 'set volume with output muted'
  else osascript -e 'set volume without output muted'; fi
}}
trap restore EXIT
trap 'exit 1' INT TERM HUP
osascript -e 'set volume output volume {level} without output muted'
echo "SPEAKER-START $(perl -MTime::HiRes=time -e 'printf q(%.3f), time')"
afplay {wav}
echo "SPEAKER-DONE"
"""


async def _play(ctx: ScenarioContext, wav: str, level: int) -> str:
    host = edge_host_steps.host_of(ctx)
    if host.ssh is None or not await edge_host_steps.is_mac(host):
        raise AssertionError("speaker_plays needs a macOS edge host (its built-in speaker)")
    script = _PLAY.format(device=shlex.quote(_BUILT_IN), level=int(level), wav=shlex.quote(wav))
    done = await asyncio.to_thread(host.run, script, 120)
    output = host.scrub(done.stdout + done.stderr)
    if done.returncode != 0 or "SPEAKER-DONE" not in output:
        raise AssertionError(f"the edge host's speaker did not play {wav}:\n{output}")
    return output


@step("speaker_plays")
async def speaker_plays(
    ctx: ScenarioContext, client: str, name: str, volume: int = 50, wait: bool = True
) -> None:
    """Play `tests/fixtures/audio/<name>.wav` (from the synced checkout) through the edge
    host's built-in speaker, next to the robot: real sound in the room for the robot's
    microphone (spec "testing rule": voice input). Volume `volume` (0-100) for the play only;
    the speaker's volume and mute are restored after it. `wait: false` returns as it starts
    (barge-in); `speaker_finished` waits for it. Marks: later checks look from here."""
    assert 0 < volume <= 60, f"volume {volume}: at most 60"
    agent = ctx.processes.get(_client_name(client))
    mark = _set_mark(ctx, client)
    ctx.state["fed"] = {"name": name, "index": mark}
    wav = f"{edge_host.EDGE_DIR}/src/{GOLDEN_DIR}/{name}.wav".replace("~/", "")
    task = asyncio.create_task(_play(ctx, wav, volume), name=f"speaker:{name}")
    ctx.state["speaker"] = {"task": task, "name": name, "started": time.monotonic()}
    timings = _timings(ctx)
    timings.setdefault("speaker", []).append({"wav": name, "volume": volume, "device": _BUILT_IN})
    print(f"{_BUILT_IN} plays {name}.wav at volume {volume} (agent line {len(agent.lines)})")
    if wait:
        await speaker_finished(ctx)


@step("speaker_finished")
async def speaker_finished(ctx: ScenarioContext, within_s: float = 60.0) -> None:
    """The edge host's speaker finished the WAV `speaker_plays` started (volume restored)."""
    playing: dict[str, Any] | None = ctx.state.get("speaker")
    assert playing is not None, "nothing is playing; use speaker_plays first"
    playing["output"] = await asyncio.wait_for(playing["task"], within_s)
    took = time.monotonic() - playing["started"]
    print(f"{playing['name']}.wav played in {took:.1f} s; speaker volume restored")


def speech_onset_s(name: str, dbfs_floor: float = -40.0) -> float:
    """Where speech starts in the golden WAV `name`: its first 20 ms above `dbfs_floor`."""
    with wave.open(str(Path(GOLDEN_DIR) / f"{name}.wav")) as wav:
        rate, pcm = wav.getframerate(), wav.readframes(wav.getnframes())
    step = rate // 50 * 2
    for at in range(0, len(pcm) - step, step):
        if dbfs(pcm[at : at + step]) > dbfs_floor:
            return at / 2 / rate
    raise AssertionError(f"{name}.wav: no sound above {dbfs_floor} dBFS")


@step("barge_in_detected_within")
async def barge_in_detected_within(
    ctx: ScenarioContext, ms: float = 300.0, source: Literal["speaker", "feed"] = "speaker"
) -> None:
    """The barge-in (`barge_in_stops_speech`) came at most `ms` after the voice started: from
    the speech onset in the golden WAV, on the edge's wall clock. `source: speaker` (the WAV
    `speaker_plays` played through the air): from the speaker's start (SPEAKER-START, on the
    edge host, the same machine as the edge's BARGE-IN wall=), so the speaker's own start-up
    delay counts against it. `source: feed` (the WAV `feed_golden_wav` fed at the mic input):
    from the agent's FEED wall=. Goes in the timings."""
    barge = ctx.state.get("barge_in")
    assert barge is not None, "no barge-in yet; use barge_in_stops_speech first"
    if source == "feed":
        fed: dict[str, Any] = ctx.state.get("fed") or {}
        assert fed.get("wall"), "nothing was fed; use feed_golden_wav first"
        start, name = float(fed["wall"]), str(fed["name"])
    else:
        playing: dict[str, Any] | None = ctx.state.get("speaker")
        assert playing is not None, "nothing was played; use speaker_plays first"
        if "output" not in playing:
            await speaker_finished(ctx)
        found = next(
            (float(x.split()[1]) for x in playing["output"].splitlines()
             if x.startswith("SPEAKER-START ")), None,
        )  # fmt: skip
        assert found is not None, f"the speaker printed no start time: {playing['output']}"
        start, name = found, str(playing["name"])
    onset = speech_onset_s(name)
    detect_ms = (barge["wall"] - (start + onset)) * 1000
    _timings(ctx)["barge_in_detect_ms"] = round(detect_ms)
    _timings(ctx)["barge_in_detect_source"] = source
    print(f"barge-in detected {detect_ms:.0f} ms after the voice started "
          f"({name}.wav speech at {onset:.2f} s, {source})")  # fmt: skip
    assert 0 < detect_ms <= ms, (
        f"barge-in detected {detect_ms:.0f} ms after the voice (want <= {ms})"
    )


@step("edge_mic_tuned")
async def edge_mic_tuned(ctx: ScenarioContext, client: str, within_s: float = 10.0) -> None:
    """The robot's XVF3800 got Pollen's startup tuning: the body's AEC line says it was written
    and every parameter read back as written (`tuning.applied`); the readback goes in the
    timings."""
    line = await _expect(ctx, _client_name(client), {"AEC"}, within_s, what="AEC")
    tuning = (line.payload or {}).get("tuning") or {}
    print(f"XVF3800 tuning: {tuning}")
    _timings(ctx)["xvf3800_tuning"] = tuning
    assert tuning.get("applied") is True, f"the XVF3800 tuning did not read back: {tuning}"
