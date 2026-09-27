"""EdgeLink contract steps: round trips through the real message and frame codecs."""

import json
from typing import Any

from pydantic import ValidationError

from assistant_contracts.frames import (
    Frame,
    FrameCodec,
    FrameError,
    FrameKind,
    FrameKindNotNegotiated,
)
from assistant_contracts.messages import (
    MESSAGE_TYPES,
    dump_message,
    message_type_name,
    parse_message,
)
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step

_ENVELOPE = {
    "v": 1,
    "id": "m-e2e",
    "session_id": "s-e2e",
    "turn_id": "t-e2e",
    "ts_mono_ns": 1,
    "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
}


def _covered(ctx: ScenarioContext, key: str) -> set[str]:
    return ctx.state.setdefault(key, set())


@step("roundtrip_message")
async def roundtrip_message(
    ctx: ScenarioContext, type: str, fields: dict[str, Any] | None = None
) -> None:
    """Encode a message as a JSON text frame and decode it back through the union."""
    wire = {**_ENVELOPE, "type": type, **(fields or {})}
    message = parse_message(json.dumps(wire))
    assert message.type == type, f"parsed as {message.type!r}"
    text = dump_message(message)
    again = parse_message(text)
    assert again == message, f"round trip changed the message:\n{message!r}\n{again!r}"
    sent = json.loads(text)
    missing = [k for k in wire if k not in sent]
    assert not missing, f"fields lost on the wire: {missing}"
    _covered(ctx, "message_types").add(type)


@step("expect_message_refused")
async def expect_message_refused(ctx: ScenarioContext, wire: dict[str, Any]) -> None:
    """Decoding this JSON must fail validation."""
    try:
        parse_message(json.dumps(wire))
    except ValidationError:
        return
    raise AssertionError(f"message was accepted: {wire}")


@step("expect_all_message_types_covered")
async def expect_all_message_types_covered(ctx: ScenarioContext) -> None:
    """Every EdgeLink v1 message type was round-tripped in this scenario."""
    all_types = {message_type_name(m) for m in MESSAGE_TYPES}
    missing = sorted(all_types - _covered(ctx, "message_types"))
    assert not missing, f"message types not exercised: {missing}"
    assert len(all_types) == 23


@step("expect_message_type_order")
async def expect_message_type_order(ctx: ScenarioContext, names: list[str]) -> None:
    """The union defines exactly these message types, in this (spec) order."""
    actual = [message_type_name(m) for m in MESSAGE_TYPES]
    assert actual == names, f"message types are {actual}"


def _kind(name: str) -> FrameKind:
    try:
        return FrameKind[name.upper()]
    except KeyError:
        raise ValueError(f"unknown frame kind {name!r}") from None


def _payload(kind: FrameKind, size: int) -> bytes:
    return bytes((i * 7 + int(kind)) % 256 for i in range(size))


@step("roundtrip_frame")
async def roundtrip_frame(
    ctx: ScenarioContext,
    kind: str,
    payload_bytes: int = 640,
    stream: int = 1,
    seq: int = 1,
    capture_ts_us: int = 1_000_000,
    opus_negotiated: bool = False,
) -> None:
    """Encode a binary frame and decode it back; checks the 14-byte header and payload."""
    frame_kind = _kind(kind)
    codec = FrameCodec(opus=opus_negotiated)
    frame = Frame(frame_kind, stream, seq, capture_ts_us, _payload(frame_kind, payload_bytes))
    raw = codec.encode(frame)
    assert len(raw) == 14 + payload_bytes
    assert raw[0] == int(frame_kind)
    assert codec.decode(raw) == frame
    _covered(ctx, "frame_kinds").add(frame_kind.name)


@step("expect_frame_refused")
async def expect_frame_refused(
    ctx: ScenarioContext, kind: str, reason: str = "not_negotiated", payload_bytes: int = 4
) -> None:
    """Encoding and decoding this frame must fail (`not_negotiated` or `malformed`)."""
    frame_kind = _kind(kind)
    frame = Frame(frame_kind, 1, 1, 1, _payload(frame_kind, payload_bytes))
    expected = FrameKindNotNegotiated if reason == "not_negotiated" else FrameError
    codec = FrameCodec(opus=False)
    for action in ("encode", "decode"):
        try:
            if action == "encode":
                codec.encode(frame)
            else:
                header = bytes([int(frame_kind), 1]) + (1).to_bytes(4, "little")
                codec.decode(header + (1).to_bytes(8, "little") + frame.payload)
        except expected:
            continue
        raise AssertionError(f"{action} accepted a {kind} frame ({reason} expected)")


@step("expect_all_frame_kinds_covered")
async def expect_all_frame_kinds_covered(ctx: ScenarioContext) -> None:
    """Every frame kind (0x01-0x05) was round-tripped in this scenario."""
    missing = sorted({k.name for k in FrameKind} - _covered(ctx, "frame_kinds"))
    assert not missing, f"frame kinds not exercised: {missing}"
