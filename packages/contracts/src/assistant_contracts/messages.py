"""EdgeLink v1 JSON control messages (proposal section 5.1).

Every message shares the envelope fields of `Envelope`; `type` selects the model. Use
`parse_message` to decode any of them and `dump_message` to encode one.
"""

# Each message narrows the envelope's `type: str` to its Literal (the union discriminator),
# and `welcome` makes `session_id` required. Pydantic supports this; pyright flags it.
# pyright: reportIncompatibleVariableOverride=false

import time
import uuid
from datetime import datetime
from typing import Annotated, ClassVar, Literal

from pydantic import Field, JsonValue, TypeAdapter

from assistant_contracts.capabilities import Capabilities
from assistant_contracts.common import (
    AttentionState,
    Channel,
    ContractModel,
    LookTarget,
    Score,
    Slot,
    StreamId,
)
from assistant_contracts.expressions import ExpressName
from assistant_contracts.version import PROTOCOL_MAJOR, PROTOCOL_VERSION

assert PROTOCOL_MAJOR == 1, "update Envelope.v when the major version changes"

Direction = Literal["edge_to_brain", "brain_to_edge", "both"]


def new_message_id() -> str:
    return f"m-{uuid.uuid4().hex[:12]}"


class Envelope(ContractModel):
    """Fields every EdgeLink JSON message carries."""

    direction: ClassVar[Direction]

    v: Literal[1] = 1
    """Major protocol version (`PROTOCOL_MAJOR`)."""
    type: str
    id: str = Field(default_factory=new_message_id, min_length=1)
    session_id: str | None = None
    turn_id: str | None = None
    ts_mono_ns: int = Field(default_factory=time.monotonic_ns, ge=0)
    traceparent: str | None = None
    """W3C trace context; export is phase 5 but the field travels from day one."""


# ---------------------------------------------------------------- handshake


class BodyInfo(ContractModel):
    kind: str = Field(description='Body driver kind: "reachy" or "console".')
    capabilities: Capabilities = Field(default_factory=Capabilities)


class Hello(Envelope):
    """First message on a connection."""

    direction: ClassVar[Direction] = "edge_to_brain"
    type: Literal["hello"] = "hello"
    device_id: str = Field(min_length=1)
    proto: str = Field(default=PROTOCOL_VERSION, pattern=r"^\d+\.\d+$")
    sw_version: str
    body: BodyInfo
    last_seq: int | None = Field(default=None, ge=0)
    """Last binary frame seq the edge received before a reconnect, if any."""


class WakeWord(ContractModel):
    word: str
    model: str
    threshold: Score


class WelcomeAudio(ContractModel):
    out_rate: int = 24000
    opus: bool = False
    """The brain accepts Opus; together with `hello` capabilities.opus this negotiates 0x05."""
    speak_text: bool = False
    """The brain may send reply text in `speak.begin.text` (the edge advertised `speak_text`)."""


class TimerJob(ContractModel):
    id: str
    at_utc: datetime
    sound: str


class Welcome(Envelope):
    """Reply to `hello`; also re-sent state after a reconnect."""

    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["welcome"] = "welcome"
    session_id: str = Field(min_length=1)
    wake_words: list[WakeWord] = Field(default_factory=list)
    follow_up_max_s: float = Field(default=10.0, ge=0)
    audio: WelcomeAudio = Field(default_factory=WelcomeAudio)
    timers: list[TimerJob] = Field(default_factory=list)
    mute: bool = False


# ---------------------------------------------------------------- wake and mic windows


class Wake(Envelope):
    direction: ClassVar[Direction] = "edge_to_brain"
    type: Literal["wake"] = "wake"
    word: str
    score: Score
    doa: float | None = None


class WakeCancel(Envelope):
    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["wake.cancel"] = "wake.cancel"
    reason: Literal["lost_arbitration", "not_allowed"]


class Vad(Envelope):
    direction: ClassVar[Direction] = "edge_to_brain"
    type: Literal["vad"] = "vad"
    state: Literal["start", "end"]
    barge_in: bool = False
    stream_id: StreamId | None = None
    """For barge-in: the output stream that was playing."""
    played_ms: int | None = Field(default=None, ge=0)


class MicClose(Envelope):
    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["mic.close"] = "mic.close"
    reason: str


class MicFollowUp(Envelope):
    """Reopen the mic without a wake word. The edge caps it at 15 s and refuses while muted."""

    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["mic.follow_up"] = "mic.follow_up"
    seconds: float = Field(gt=0)


class TextInput(Envelope):
    """Typed input (dev and typing clients only)."""

    direction: ClassVar[Direction] = "edge_to_brain"
    type: Literal["text.input"] = "text.input"
    text: str = Field(min_length=1)


# ---------------------------------------------------------------- output audio


class SpeakBegin(Envelope):
    """Binary 0x02 frames with `stream == stream_id` follow until `speak.end`."""

    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["speak.begin"] = "speak.begin"
    stream_id: StreamId
    channel: Channel = "speech"
    rate: int = Field(default=24000, gt=0)
    text: str | None = None
    """The reply text of this stream, if `speak_text` was negotiated (shown or logged)."""


class SpeakEnd(Envelope):
    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["speak.end"] = "speak.end"
    stream_id: StreamId
    channel: Channel = "speech"


class Flush(Envelope):
    """Stop and drop queued audio for one stream, or for all."""

    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["flush"] = "flush"
    stream_id: StreamId | Literal["all"]


class Duck(Envelope):
    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["duck"] = "duck"
    channel: Channel
    gain_db: float = Field(le=0.0, description="0 restores the channel; negative ducks it.")


class Playback(Envelope):
    """The edge playback clock: started, progress every 200 ms, then done or flushed."""

    direction: ClassVar[Direction] = "edge_to_brain"
    type: Literal["playback"] = "playback"
    stream_id: StreamId
    played_ms: int = Field(ge=0)
    state: Literal["started", "progress", "done", "flushed"]


class PlaySound(Envelope):
    """Play a cached clip; the brain pushes missing clips as binary 0x04."""

    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["play_sound"] = "play_sound"
    sound_id: str
    channel: Channel = "alert"


# ---------------------------------------------------------------- body intents


class Attention(Envelope):
    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["attention"] = "attention"
    state: AttentionState
    assistant: str | None = None


class Express(Envelope):
    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["express"] = "express"
    name: ExpressName
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)


class LookAt(Envelope):
    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["look_at"] = "look_at"
    target: LookTarget


class Snapshot(Envelope):
    """Camera request. The edge answers with `result` (re = this id) and 0x03 frames on `slot`."""

    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["snapshot"] = "snapshot"
    slot: Slot
    max_side: int = Field(default=1024, gt=0)


class Result(Envelope):
    direction: ClassVar[Direction] = "edge_to_brain"
    type: Literal["result"] = "result"
    re: str = Field(min_length=1, description="The id of the request this answers.")
    ok: bool
    error: str | None = None
    data: JsonValue = None


# ---------------------------------------------------------------- state, timers, events, errors


class Privacy(Envelope):
    """edge->brain: the authoritative mute state. brain->edge: a request only."""

    direction: ClassVar[Direction] = "both"
    type: Literal["privacy"] = "privacy"
    muted: bool
    hard: bool = False
    until: datetime | None = None


class TimerSync(Envelope):
    """Idempotent mirror of the device's timers, so they ring even while offline."""

    direction: ClassVar[Direction] = "brain_to_edge"
    type: Literal["timer.sync"] = "timer.sync"
    jobs: list[TimerJob] = Field(default_factory=list)


class EdgeEvent(Envelope):
    direction: ClassVar[Direction] = "edge_to_brain"
    type: Literal["event"] = "event"
    kind: Literal["button", "touch", "imu_tap"]
    data: dict[str, JsonValue] = Field(default_factory=dict)


class Error(Envelope):
    direction: ClassVar[Direction] = "both"
    type: Literal["error"] = "error"
    code: str
    message: str


# ---------------------------------------------------------------- union and helpers

MESSAGE_TYPES: tuple[type[Envelope], ...] = (
    Hello,
    Welcome,
    Wake,
    WakeCancel,
    Vad,
    MicClose,
    MicFollowUp,
    TextInput,
    SpeakBegin,
    SpeakEnd,
    Flush,
    Duck,
    Playback,
    PlaySound,
    Attention,
    Express,
    LookAt,
    Snapshot,
    Result,
    Privacy,
    TimerSync,
    EdgeEvent,
    Error,
)
"""All 23 EdgeLink v1 message models, in the order of the proposal's table."""

EdgeLinkMessage = Annotated[
    Hello
    | Welcome
    | Wake
    | WakeCancel
    | Vad
    | MicClose
    | MicFollowUp
    | TextInput
    | SpeakBegin
    | SpeakEnd
    | Flush
    | Duck
    | Playback
    | PlaySound
    | Attention
    | Express
    | LookAt
    | Snapshot
    | Result
    | Privacy
    | TimerSync
    | EdgeEvent
    | Error,
    Field(discriminator="type"),
]

MESSAGE_ADAPTER: TypeAdapter[EdgeLinkMessage] = TypeAdapter(EdgeLinkMessage)


def message_type_name(model: type[Envelope]) -> str:
    """The wire `type` string of a message model, e.g. `WakeCancel` -> "wake.cancel"."""
    default = model.model_fields["type"].default
    assert isinstance(default, str)
    return default


def parse_message(data: str | bytes | dict[str, object]) -> EdgeLinkMessage:
    """Decode one JSON text frame (or an already-decoded dict) into its message model.

    Raises `pydantic.ValidationError` for an unknown `type`, a wrong `v` or bad fields.
    """
    if isinstance(data, dict):
        return MESSAGE_ADAPTER.validate_python(data)
    return MESSAGE_ADAPTER.validate_json(data)


def dump_message(message: Envelope) -> str:
    """Encode a message as a JSON text frame. Fields that are None are left out."""
    return message.model_dump_json(exclude_none=True)


assert len(MESSAGE_TYPES) == 23
assert len({message_type_name(m) for m in MESSAGE_TYPES}) == 23
