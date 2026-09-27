import json
from typing import Any

import pytest
from pydantic import ValidationError

from assistant_contracts.expressions import ExpressName
from assistant_contracts.messages import (
    MESSAGE_TYPES,
    Envelope,
    Express,
    Flush,
    Hello,
    LookAt,
    Welcome,
    dump_message,
    message_type_name,
    parse_message,
)
from assistant_contracts.version import PROTOCOL_VERSION, CloseCode, is_compatible

pytestmark = pytest.mark.unit

EXAMPLES: dict[str, dict[str, Any]] = {
    "hello": {
        "device_id": "lite",
        "sw_version": "0.1.0",
        "body": {
            "kind": "sounddevice",
            "capabilities": {
                "audio_in": {"rate": 16000, "aec": "hw"},
                "audio_out": {"rates": [16000, 24000]},
                "wake": {"engines": ["openwakeword", "sherpa_kws"]},
                "motion": {
                    "expressions": ["happy", "sad"],
                    "look_at": ["user", "doa"],
                    "attention": True,
                },
                "camera": {"w": 1280, "h": 720},
                "doa": True,
            },
        },
        "last_seq": 17,
    },
    "welcome": {
        "session_id": "s-1",
        "wake_words": [{"word": "hey jarvis", "model": "hey_jarvis", "threshold": 0.4}],
        "follow_up_max_s": 10,
        "audio": {"out_rate": 24000},
        "timers": [{"id": "t1", "at_utc": "2026-09-26T10:00:00Z", "sound": "chime"}],
        "mute": False,
    },
    "wake": {"word": "hey jarvis", "score": 0.83, "doa": 12.5},
    "wake.cancel": {"reason": "lost_arbitration"},
    "vad": {"state": "start", "barge_in": True, "stream_id": 2, "played_ms": 640},
    "mic.close": {"reason": "turn complete"},
    "mic.follow_up": {"seconds": 8},
    "text.input": {"text": "what time is it"},
    "speak.begin": {"stream_id": 2, "channel": "speech", "rate": 24000, "turn_id": "t-9"},
    "speak.end": {"stream_id": 2, "channel": "speech", "turn_id": "t-9"},
    "flush": {"stream_id": "all"},
    "duck": {"channel": "media", "gain_db": -12},
    "playback": {"stream_id": 2, "played_ms": 400, "state": "progress"},
    "play_sound": {"sound_id": "chime", "channel": "alert"},
    "attention": {"state": "listening", "assistant": "jarvis"},
    "express": {"name": "happy", "intensity": 0.7},
    "look_at": {"target": {"kind": "world", "x": 0.5, "y": 0.0, "z": 0.2}},
    "snapshot": {"id": "req-5", "slot": 1, "max_side": 1024},
    "result": {"re": "req-5", "ok": True, "data": {"bytes": 20480, "chunks": 3}},
    "privacy": {"muted": True, "hard": False, "until": "2026-09-26T11:00:00Z"},
    "timer.sync": {"jobs": [{"id": "t1", "at_utc": "2026-09-26T10:00:00Z", "sound": "alarm"}]},
    "event": {"kind": "button", "data": {"pressed": True}},
    "error": {"code": "bad_request", "message": "unknown stream"},
}

TYPE_NAMES = [
    "hello", "welcome", "wake", "wake.cancel", "vad", "mic.close", "mic.follow_up",
    "text.input", "speak.begin", "speak.end", "flush", "duck", "playback", "play_sound",
    "attention", "express", "look_at", "snapshot", "result", "privacy", "timer.sync",
    "event", "error",
]  # fmt: skip


def test_all_23_types_defined_in_order() -> None:
    assert [message_type_name(m) for m in MESSAGE_TYPES] == TYPE_NAMES
    assert set(EXAMPLES) == set(TYPE_NAMES)


@pytest.mark.parametrize("model", MESSAGE_TYPES, ids=message_type_name)
def test_every_type_parses_through_the_union(model: type[Envelope]) -> None:
    type_name = message_type_name(model)
    wire: dict[str, Any] = {
        "v": 1,
        "type": type_name,
        "id": "m-1",
        "session_id": "s-1",
        "turn_id": "t-1",
        "ts_mono_ns": 123,
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        **EXAMPLES[type_name],
    }
    message = parse_message(json.dumps(wire))
    assert type(message) is model
    assert message.type == type_name
    assert parse_message(dump_message(message)) == message
    assert parse_message(wire) == message


def test_envelope_defaults() -> None:
    a = Express(name=ExpressName.HAPPY)
    b = Express(name=ExpressName.HAPPY)
    assert a.v == 1
    assert a.id != b.id
    assert a.ts_mono_ns > 0
    assert "session_id" not in json.loads(dump_message(a))


@pytest.mark.parametrize(
    "wire",
    [
        {"v": 2, "type": "wake", "word": "x", "score": 0.5},
        {"v": 1, "type": "nope"},
        {"v": 1, "type": "wake", "word": "x", "score": 1.5},
        {"v": 1, "type": "express", "name": "not_an_intent"},
        {"v": 1, "type": "speak.begin", "stream_id": 256},
        {"v": 1, "type": "welcome"},
        {"v": 1, "type": "look_at", "target": {"kind": "image", "u": 2, "v": 0}},
        {"v": 1, "type": "mic.follow_up", "seconds": 0},
        {"v": 1, "type": "duck", "channel": "media", "gain_db": 3},
    ],
)
def test_invalid_messages_are_refused(wire: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_message(wire)


def test_unknown_fields_are_ignored_for_minor_versions() -> None:
    message = parse_message({"v": 1, "type": "flush", "stream_id": 3, "future_field": True})
    assert message == Flush(id=message.id, ts_mono_ns=message.ts_mono_ns, stream_id=3)


def test_hello_defaults_to_our_protocol_version() -> None:
    hello = Hello.model_validate({"device_id": "pi", "sw_version": "1", "body": {"kind": "null"}})
    assert hello.proto == PROTOCOL_VERSION
    assert is_compatible(hello.proto)
    assert not is_compatible("2.0")
    assert not is_compatible("x")


def test_welcome_requires_a_session() -> None:
    assert Welcome(session_id="s").session_id == "s"


def test_look_at_targets() -> None:
    targets = [{"kind": "user"}, {"kind": "doa", "doa": -30}, {"kind": "image", "u": 0.5, "v": 0.5}]
    for target in targets:
        message = parse_message({"v": 1, "type": "look_at", "target": target})
        assert isinstance(message, LookAt)
        assert message.target.kind == target["kind"]


def test_close_codes() -> None:
    assert CloseCode.VERSION_MISMATCH == 4001
    assert CloseCode.AUTH_REFUSED == 4003


def test_express_enum_has_the_42_pollen_intents() -> None:
    assert len(ExpressName) == 42
    assert {"happy", "yes_understanding", "go_away", "dying"} <= {e.value for e in ExpressName}
