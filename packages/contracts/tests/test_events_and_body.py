from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from assistant_contracts.body import AudioFrame, AudioIO, Body, BodyEvent, PlaybackEvent
from assistant_contracts.capabilities import BodyCapabilities, Capabilities
from assistant_contracts.common import LookAtUser
from assistant_contracts.events import (
    BUS_EVENT_TYPES,
    BodyIntent,
    JobOwner,
    SchedulerDue,
    SpeechRequest,
    due_topic,
    event_type_for_topic,
    skill_topic,
)

pytestmark = pytest.mark.unit


def test_topics_are_unique_and_versioned() -> None:
    topics = [m.topic for m in BUS_EVENT_TYPES]
    assert len(topics) == len(set(topics))
    assert all(m.model_fields["v"].default == 1 for m in BUS_EVENT_TYPES)


def test_topic_lookup() -> None:
    assert due_topic("clock") == "scheduler.due.clock"
    assert event_type_for_topic("scheduler.due.clock") is SchedulerDue
    assert event_type_for_topic("speech.request") is SpeechRequest
    assert event_type_for_topic(skill_topic("radio", "started")) is None


def test_speech_request_needs_exactly_one_of_text_and_prompt() -> None:
    SpeechRequest(assistant_id="jarvis", text="Time is up")
    SpeechRequest(assistant_id="jarvis", prompt="Tell the user the timer is done", mode="llm")
    with pytest.raises(ValidationError):
        SpeechRequest(assistant_id="jarvis")
    with pytest.raises(ValidationError):
        SpeechRequest(assistant_id="jarvis", text="a", prompt="b")
    with pytest.raises(ValidationError):
        SpeechRequest(assistant_id="jarvis", text="a", target="everywhere")


def test_body_intent_fields_match_kind() -> None:
    BodyIntent(device_id="lite", kind="attention", state="thinking")
    BodyIntent(device_id="lite", kind="look_at", target=LookAtUser())
    with pytest.raises(ValidationError):
        BodyIntent(device_id="lite", kind="express")


def test_scheduler_due_round_trip() -> None:
    due = SchedulerDue(
        job_id="j1",
        event="due",
        payload={"message": "tea"},
        owner=JobOwner(assistant_id="jarvis", skill_id="clock"),
        origin_device="lite",
    )
    assert SchedulerDue.model_validate_json(due.model_dump_json()) == due


class _NullAudio:
    aec = "none"

    async def capture(self) -> AsyncIterator[AudioFrame]:
        yield AudioFrame(pcm=b"\x00" * 640, capture_ts_us=0)

    async def play(self, stream_id: int, pcm: bytes, rate: int) -> None:
        return None

    async def flush(self, stream_id: int | None = None) -> int:
        return 0

    async def playback_events(self) -> AsyncIterator[PlaybackEvent]:
        yield PlaybackEvent(stream_id=0, played_ms=0, state="done")


class _NullBody:
    kind = "null"
    motion = None
    camera = None

    def __init__(self) -> None:
        self.audio = _NullAudio()

    async def start(self) -> BodyCapabilities:
        return Capabilities()

    async def stop(self) -> None:
        return None

    async def events(self) -> AsyncIterator[BodyEvent]:
        yield BodyEvent(kind="button")


def test_a_minimal_body_satisfies_the_protocols() -> None:
    body = _NullBody()
    assert isinstance(body, Body)
    assert isinstance(body.audio, AudioIO)
