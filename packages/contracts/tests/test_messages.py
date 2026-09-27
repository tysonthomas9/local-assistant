import json
from typing import Any

import pytest
from pydantic import ValidationError

from assistant_contracts.expressions import ExpressName
from assistant_contracts.messages import (
    MESSAGE_TYPES,
    Express,
    Flush,
    Hello,
    Welcome,
    dump_message,
    message_type_name,
    parse_message,
)
from assistant_contracts.version import PROTOCOL_VERSION, CloseCode, is_compatible

pytestmark = pytest.mark.unit

TYPE_NAMES = [
    "hello", "welcome", "wake", "wake.cancel", "vad", "mic.close", "mic.follow_up",
    "text.input", "speak.begin", "speak.end", "flush", "duck", "playback", "play_sound",
    "attention", "express", "look_at", "snapshot", "result", "privacy", "timer.sync",
    "event", "error",
]  # fmt: skip


def test_all_23_types_defined_in_order() -> None:
    assert [message_type_name(m) for m in MESSAGE_TYPES] == TYPE_NAMES


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
    hello = Hello.model_validate(
        {"device_id": "pi", "sw_version": "1", "body": {"kind": "console"}}
    )
    assert hello.proto == PROTOCOL_VERSION
    assert is_compatible(hello.proto)
    assert not is_compatible("2.0")
    assert not is_compatible("x")


def test_welcome_requires_a_session() -> None:
    assert Welcome(session_id="s").session_id == "s"


def test_close_codes() -> None:
    assert CloseCode.VERSION_MISMATCH == 4001
    assert CloseCode.AUTH_REFUSED == 4003


def test_express_enum_has_the_42_pollen_intents() -> None:
    assert len(ExpressName) == 42
    assert {"happy", "yes_understanding", "go_away", "dying"} <= {e.value for e in ExpressName}
