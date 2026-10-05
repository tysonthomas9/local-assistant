"""Person tracking steps (S8g): the robot turns toward a voice and follows a face.

Measured on the robot's machine by the state sampler (20 Hz, next to the daemon), as the other
robot steps measure. The head's yaw is taken relative to the body (`head_pose.yaw -
body_yaw`): positive is the robot's left, like a direction of arrival. Angular speeds are the
head's rotation (any axis) over `SPEED_WINDOW_S` between samples, so one sample's jitter is not
taken for a jump. Each step writes what it measured to
`<artifacts>/tracking-<feature>-<what>[-sim].json` (`$ASSISTANT_ARTIFACTS_DIR`, default
`<repo>/artifacts`).

The simulated robot has no camera and no XVF3800 (its daemon answers that it cannot track,
and reads no direction of arrival): the face and the voice direction through the air are
hw-only (`robot_follows_face`, `voice_turned_toward`); everything else runs on both.
"""

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.steps import edge_host as edge_host_steps
from assistant_testing.steps import listen as listen_steps
from assistant_testing.steps.brain import _start_sampler
from assistant_testing.steps.edge import (
    _request,
    check_sampling,
    head_turn_deg,
    move_samples,
)
from assistant_testing.steps.link import SERVER, _client_name, _expect, _get_lines

SPEED_WINDOW_S = 0.2
"""Angular speeds are measured over this much time between samples."""
POLLEN_HEAD_DEG = 65.0
"""Pollen's limit of the head's yaw relative to the body (the legacy app's)."""
POLLEN_BODY_DEG = 160.0
"""Pollen's limit of the body's yaw (the legacy app's)."""


def yaw_rel_deg(sample: dict[str, Any]) -> float:
    """The head's yaw relative to the body, degrees (positive: the robot's left)."""
    yaw = sample["head_pose"]["yaw"] - (sample.get("body_yaw") or 0.0)
    return math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))


def max_speed_dps(samples: list[dict[str, Any]], window_s: float = SPEED_WINDOW_S) -> float:
    """The head's fastest rotation (degrees per second) over `window_s` between samples."""
    peak, j = 0.0, 0
    for i, a in enumerate(samples):
        j = max(j, i + 1)
        while j < len(samples) and samples[j]["t"] - a["t"] < window_s:
            j += 1
        if j >= len(samples):
            break
        b = samples[j]
        peak = max(peak, head_turn_deg(a["head_pose"], b["head_pose"]) / (b["t"] - a["t"]))
    return peak


def _limits(samples: list[dict[str, Any]]) -> tuple[float, float]:
    head = max(abs(yaw_rel_deg(s)) for s in samples)
    body = max(abs(math.degrees(s.get("body_yaw") or 0.0)) for s in samples)
    return head, body


def _record(ctx: ScenarioContext, what: str, data: dict[str, Any]) -> None:
    directory = Path(os.environ.get("ASSISTANT_ARTIFACTS_DIR") or ctx.repo_root / "artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "-sim" if ctx.sim else ""
    path = directory / f"tracking-{ctx.feature_path.stem}-{what}{suffix}.json"
    data = {"feature": ctx.feature_path.stem, "robot": ctx.robot, **data}
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"measured -> {path.name}")


def _samples(sampler: Any) -> list[dict[str, Any]]:
    return [p for line in _get_lines(sampler, "STATE") if (p := line.payload) is not None]


@step("robot_turns_toward")
async def robot_turns_toward(
    ctx: ScenarioContext,
    client: str,
    doa: float,
    min_deg: float = 5.0,
    max_deg: float = 10.0,
    tolerance_deg: float = 2.0,
    max_dps: float = 20.0,
    within_s: float = 60.0,
) -> None:
    """The brain sends `look_at{target: {kind: doa, doa}}` (degrees, positive: the robot's
    left); the body turns the head toward it (VOICE-TURN, waking a robot at rest first): a
    move of ours, so at most `max_deg` of yaw from neutral (+ `tolerance_deg` of servo error)
    and slow. Measured by the sampler from the pose when the turn started to the pose after
    its glide: the yaw changed toward `doa` by at least `min_deg`, never beyond `max_deg`, and
    the head never rotated faster than `max_dps`."""
    name = _client_name(client)
    sampler = await _start_sampler(ctx)
    try:
        target = {"target": {"kind": "doa", "doa": doa}}
        result = await _request(ctx, client, "look_at", target, within_s)
        assert result.get("ok") is True, f"look_at doa {doa} failed: {result}"
        line = await _expect(
            ctx, name, {"VOICE-TURN"}, 10, fields={"turned": "true"}, what="VOICE-TURN"
        )
        print(line.text)
        turn = line.payload or {}
        await asyncio.sleep(float(turn.get("seconds", 1.0)) + 0.8)
    finally:
        await sampler.stop()
    samples = _samples(sampler)
    t_start, t_end = turn["t_start"], turn["t_end"]
    ref, during = move_samples(samples, t_start, t_end + 0.6)
    check_sampling(during, t_end + 0.6 - t_start)
    before, after = yaw_rel_deg(ref), sum(yaw_rel_deg(s) for s in during[-3:]) / 3
    turned = after - before
    peak = max(abs(yaw_rel_deg(s)) for s in during)
    speed = max_speed_dps(during)
    trace = " ".join(f"{yaw_rel_deg(s):.1f}" for s in during[::3])
    print(f"yaw relative to the body every 3rd sample (deg): {trace}")
    print(
        f"turn toward {doa:+g} deg: yaw {before:+.1f} -> {after:+.1f} ({turned:+.1f} deg), "
        f"at most {peak:.1f} deg from neutral, fastest {speed:.1f} deg/s"
    )
    _record(ctx, f"turn{doa:+g}", {
        "doa_deg": doa, "voice_turn": turn, "yaw_before_deg": round(before, 2),
        "yaw_after_deg": round(after, 2), "turned_deg": round(turned, 2),
        "peak_yaw_deg": round(peak, 2), "max_speed_dps": round(speed, 2),
        "samples": len(during),
    })  # fmt: skip
    assert turned * math.copysign(1.0, doa) >= min_deg, (
        f"the head turned {turned:+.1f} deg, want at least {min_deg} toward {doa:+g}"
    )
    assert peak <= max_deg + tolerance_deg, f"the head turned {peak:.1f} deg (limit {max_deg})"
    assert speed <= max_dps, f"the head rotated at {speed:.1f} deg/s (limit {max_dps})"


@step("robot_head_holds")
async def robot_head_holds(
    ctx: ScenarioContext,
    seconds: float = 3.0,
    max_deg: float = 2.0,
    max_mean_deg: float | None = None,
    what: str = "hold",
) -> None:
    """For `seconds` the head holds its pose: sampled next to the daemon, it never turns more
    than `max_deg` from where it was at the start; with `max_mean_deg`, the pose AVERAGED over
    the time (each of yaw, pitch and roll) stays that close to the start, while it may move
    around it by up to `max_deg` (the speech wobble while the robot speaks). Its fastest
    rotation and Pollen's limits are recorded."""
    sampler = await _start_sampler(ctx)
    try:
        await asyncio.sleep(seconds)
    finally:
        await sampler.stop()
    samples = _samples(sampler)
    assert samples, "no state sample"
    check_sampling(samples, samples[-1]["t"] - samples[0]["t"])
    ref = samples[0]
    turns = [head_turn_deg(ref["head_pose"], s["head_pose"]) for s in samples]
    peak = max(turns)
    means = {
        axis: math.degrees(sum(s["head_pose"][axis] - ref["head_pose"][axis] for s in samples))
        / len(samples)
        for axis in ("yaw", "pitch", "roll")
    }
    drift = max(abs(v) for v in means.values())
    head, body = _limits(samples)
    speed = max_speed_dps(samples)
    print(
        f"{len(samples)} samples over {seconds:g} s: at most {peak:.1f} deg from the start, "
        f"mean offset {drift:.1f} deg, fastest {speed:.1f} deg/s, head yaw within {head:.1f}, "
        f"body {body:.1f} deg"
    )
    _record(ctx, what, {
        "seconds": seconds, "peak_deg": round(peak, 2), "mean_offset_deg": round(drift, 2),
        "max_speed_dps": round(speed, 2), "head_yaw_max_deg": round(head, 2),
        "body_yaw_max_deg": round(body, 2), "samples": len(samples),
    })  # fmt: skip
    assert peak <= max_deg, f"the head moved {peak:.1f} deg (limit {max_deg})"
    if max_mean_deg is not None:
        assert drift <= max_mean_deg, (
            f"the head's average pose moved {drift:.1f} deg (limit {max_mean_deg}): {means}"
        )
    assert head <= POLLEN_HEAD_DEG, f"head yaw {head:.1f} deg beyond Pollen's limit"
    assert body <= POLLEN_BODY_DEG, f"body yaw {body:.1f} deg beyond Pollen's limit"


@step("voice_turned_toward")
async def voice_turned_toward(
    ctx: ScenarioContext,
    client: str,
    name: str,
    side: Literal["left", "right"],
    min_deg: float = 5.0,
    max_deg: float = 10.0,
    tolerance_deg: float = 2.0,
    max_dps: float = 20.0,
    volume: int = 50,
    within_s: float = 30.0,
) -> None:
    """hw: `tests/fixtures/audio/<name>.wav` plays through the edge host's speaker, placed on
    the robot's `side`; the mic window it opens makes the body read the XVF3800's direction of
    arrival (DOA, on that side) and turn the head toward it (VOICE-TURN): measured as
    `robot_turns_toward` (sign, at least `min_deg`, at most `max_deg`, at most `max_dps`)."""
    agent = _client_name(client)
    sign = 1.0 if side == "left" else -1.0
    sampler = await _start_sampler(ctx)
    try:
        await listen_steps.speaker_plays(ctx, client, name, volume=volume, wait=False)
        doa_line = await _expect(ctx, agent, {"DOA"}, within_s, what="DOA (the voice direction)")
        print(doa_line.text)
        line = await _expect(
            ctx, agent, {"VOICE-TURN"}, 10, fields={"turned": "true"}, what="VOICE-TURN"
        )
        print(line.text)
        turn = line.payload or {}
        await asyncio.sleep(float(turn.get("seconds", 1.0)) + 0.8)
        await listen_steps.speaker_finished(ctx)
    finally:
        await sampler.stop()
    doa = float(doa_line.fields["deg"])
    assert doa * sign > 0, f"the DOA reads {doa:+.1f} deg: not on the robot's {side}"
    samples = _samples(sampler)
    ref, during = move_samples(samples, turn["t_start"], turn["t_end"] + 0.6)
    before, after = yaw_rel_deg(ref), sum(yaw_rel_deg(s) for s in during[-3:]) / 3
    turned, peak = after - before, max(abs(yaw_rel_deg(s)) for s in during)
    speed = max_speed_dps(during)
    print(f"voice on the {side} ({doa:+.1f} deg): yaw {before:+.1f} -> {after:+.1f}, "
          f"fastest {speed:.1f} deg/s")  # fmt: skip
    _record(ctx, f"voice-{side}", {
        "doa_deg": doa, "voice_turn": turn, "turned_deg": round(turned, 2),
        "peak_yaw_deg": round(peak, 2), "max_speed_dps": round(speed, 2),
    })  # fmt: skip
    assert turned * sign >= min_deg, f"the head turned {turned:+.1f} deg, not to the {side}"
    assert peak <= max_deg + tolerance_deg, f"the head turned {peak:.1f} deg (limit {max_deg})"
    assert speed <= max_dps, f"the head rotated at {speed:.1f} deg/s (limit {max_dps})"


@step("robot_follows_face")
async def robot_follows_face(
    ctx: ScenarioContext,
    client: str,
    min_deg: float = 10.0,
    max_dps: float = 180.0,
    within_s: float = 120.0,
) -> None:
    """hw: a face (a person, or a face shown on a screen) moves to the robot's left, then to
    its right, in front of the camera; Pollen's face tracker follows it: while it sees a face
    the head's yaw (relative to the body) reaches at least `min_deg` to the left and to the
    right, within Pollen's limits (head 65, body 160 degrees), never faster than `max_dps` (no
    jumps). A camera frame is saved at each side (the link server started with
    `save_jpegs`); the face and yaw trace goes to the artifacts."""
    url = edge_host_steps._daemon_url(ctx)
    print(f"move a face to the robot's LEFT, then to its RIGHT, within {within_s:g} s")
    sampler = await _start_sampler(ctx)
    trace: list[dict[str, Any]] = []
    frames: dict[str, str] = {}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within_s
    try:
        while loop.time() < deadline and len(frames) < 2:
            face = (await edge_host_steps._get(f"{url}/api/media/tracking/face", 2)) or {}
            state = await edge_host_steps._get(f"{url}/api/state/full", 2)
            target = face.get("face_target") or {}
            yaw = yaw_rel_deg(state)
            trace.append({"t": round(loop.time(), 3), "detected": bool(target.get("detected")),
                          "x": target.get("x"), "y": target.get("y"),
                          "yaw_rel_deg": round(yaw, 2)})  # fmt: skip
            side = "left" if yaw >= min_deg else "right" if yaw <= -min_deg else None
            if side is not None and target.get("detected") and side not in frames:
                print(f"face followed to the {side}: yaw {yaw:+.1f} deg")
                result = await _request(ctx, client, "snapshot", {"slot": 1, "max_side": 640}, 20)
                assert result.get("ok") is True, f"snapshot failed: {result}"
                jpeg = await _expect(ctx, SERVER, {"JPEG"}, 10, fields={"device": client},
                                     what="the camera frame")  # fmt: skip
                frames[side] = jpeg.fields.get("path", "(not saved: no save_jpegs)")
            await asyncio.sleep(0.1)
    finally:
        await sampler.stop()
    samples = _samples(sampler)
    check_sampling(samples, samples[-1]["t"] - samples[0]["t"] if samples else 0.0)
    head, body = _limits(samples)
    speed = max_speed_dps(samples)
    print(f"followed: frames {frames}; head yaw within {head:.1f}, body {body:.1f} deg, "
          f"fastest {speed:.1f} deg/s")  # fmt: skip
    _record(ctx, "face", {
        "frames": frames, "head_yaw_max_deg": round(head, 2), "body_yaw_max_deg": round(body, 2),
        "max_speed_dps": round(speed, 2), "trace": trace,
    })  # fmt: skip
    assert set(frames) == {"left", "right"}, (
        f"the head did not follow a face to both sides: {frames}"
    )
    assert head <= POLLEN_HEAD_DEG + 2, f"head yaw {head:.1f} deg beyond Pollen's limit"
    assert body <= POLLEN_BODY_DEG + 2, f"body yaw {body:.1f} deg beyond Pollen's limit"
    assert speed <= max_dps, f"the head rotated at {speed:.1f} deg/s: a jump (limit {max_dps})"
