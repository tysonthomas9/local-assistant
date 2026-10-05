"""In-process event-bus payloads (proposal section 5.4). Each carries a version `v`.

`topic` is the bus topic a model is published on. Topics with a per-skill suffix
(`scheduler.due.<skill_id>`, `skill.<skill_id>.<name>`) are built with `due_topic` and
`skill_topic`; skill-defined payloads under `skill.*` have no contract model.
"""

from datetime import datetime
from typing import ClassVar, Literal

from pydantic import Field, JsonValue, model_validator

from assistant_contracts.capabilities import Capabilities
from assistant_contracts.common import AttentionState, Channel, ContractModel, LookTarget, Score
from assistant_contracts.expressions import ExpressName


class BusEvent(ContractModel):
    topic: ClassVar[str]
    v: Literal[1] = 1


# ---------------------------------------------------------------- edges and wake


class EdgeConnected(BusEvent):
    topic: ClassVar[str] = "edge.connected"
    device_id: str
    capabilities: Capabilities


class EdgeDisconnected(BusEvent):
    topic: ClassVar[str] = "edge.disconnected"
    device_id: str
    capabilities: Capabilities | None = None


class WakeDetected(BusEvent):
    topic: ClassVar[str] = "wake.detected"
    device_id: str
    word: str
    score: Score


class WakeRouted(BusEvent):
    topic: ClassVar[str] = "wake.routed"
    device_id: str
    assistant_id: str
    session_id: str


# ---------------------------------------------------------------- turns and tools


class TurnStarted(BusEvent):
    topic: ClassVar[str] = "turn.started"
    session_id: str
    turn_id: str
    device_id: str | None = None


class TurnUserText(BusEvent):
    topic: ClassVar[str] = "turn.user_text"
    session_id: str
    turn_id: str
    text: str


class TurnFinished(BusEvent):
    topic: ClassVar[str] = "turn.finished"
    session_id: str
    turn_id: str


class TurnInterrupted(BusEvent):
    topic: ClassVar[str] = "turn.interrupted"
    session_id: str
    turn_id: str
    played_ms: int | None = Field(default=None, ge=0)


class ToolCalled(BusEvent):
    """Observability only."""

    topic: ClassVar[str] = "tool.called"
    skill: str
    tool: str


class ToolResult(BusEvent):
    """Observability only."""

    topic: ClassVar[str] = "tool.result"
    skill: str
    tool: str
    ms: float = Field(ge=0)
    ok: bool


# ---------------------------------------------------------------- speech and audio


class SpeechRequest(BusEvent):
    """Ask the DialogManager to say something; it decides now, queue or drop."""

    topic: ClassVar[str] = "speech.request"
    assistant_id: str
    text: str | None = None
    """Verbatim text (mode "verbatim") or the instruction for the LLM (mode "llm")."""
    prompt: str | None = None
    mode: Literal["verbatim", "llm"] = "verbatim"
    priority: Literal["low", "normal", "high"] = "normal"
    target: str = Field(default="origin", pattern=r"^(origin|device:.+|room:.+)$")
    ttl_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _text_or_prompt(self) -> "SpeechRequest":
        if (self.text is None) == (self.prompt is None):
            raise ValueError("exactly one of text and prompt must be set")
        return self


class SpeechStarted(BusEvent):
    topic: ClassVar[str] = "speech.started"
    device_id: str
    stream_id: int
    assistant_id: str


class SpeechFinished(BusEvent):
    topic: ClassVar[str] = "speech.finished"
    device_id: str
    stream_id: int
    assistant_id: str


class AudioFocusRequest(BusEvent):
    topic: ClassVar[str] = "audio.focus.request"
    device_id: str
    channel: Channel
    owner: str


class AudioFocusGranted(BusEvent):
    topic: ClassVar[str] = "audio.focus.granted"
    device_id: str
    channel: Channel
    owner: str


class AudioFocusLost(BusEvent):
    topic: ClassVar[str] = "audio.focus.lost"
    device_id: str
    channel: Channel
    owner: str


class AudioStreamStarted(BusEvent):
    topic: ClassVar[str] = "audio.stream.started"
    handle: str
    title: str | None = None
    reason: str | None = None


class AudioStreamEnded(BusEvent):
    topic: ClassVar[str] = "audio.stream.ended"
    handle: str
    title: str | None = None
    reason: str | None = None


# ---------------------------------------------------------------- body, privacy, scheduler


class BodyIntent(BusEvent):
    """A body intent for one device; the fields used depend on `kind`."""

    topic: ClassVar[str] = "body.intent"
    device_id: str
    kind: Literal["attention", "express", "look_at"]
    state: AttentionState | None = None
    assistant: str | None = None
    name: ExpressName | None = None
    intensity: float = Field(default=1.0, ge=0.0, le=1.0)
    target: LookTarget | None = None

    @model_validator(mode="after")
    def _fields_match_kind(self) -> "BodyIntent":
        required = {"attention": self.state, "express": self.name, "look_at": self.target}
        if required[self.kind] is None:
            field = {"attention": "state", "express": "name", "look_at": "target"}[self.kind]
            raise ValueError(f"body.intent kind {self.kind!r} needs {field!r}")
        return self


class PrivacyChanged(BusEvent):
    topic: ClassVar[str] = "privacy.changed"
    device_id: str
    muted: bool
    hard: bool = False
    until: datetime | None = None
    """When a timed mute ends."""


class JobOwner(ContractModel):
    assistant_id: str
    skill_id: str


class SchedulerDue(BusEvent):
    """Published on `scheduler.due.<skill_id>` (see `due_topic`)."""

    topic: ClassVar[str] = "scheduler.due"
    job_id: str
    event: str
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    owner: JobOwner
    origin_device: str | None = None


def due_topic(skill_id: str) -> str:
    return f"{SchedulerDue.topic}.{skill_id}"


def skill_topic(skill_id: str, name: str) -> str:
    return f"skill.{skill_id}.{name}"


BUS_EVENT_TYPES: tuple[type[BusEvent], ...] = (
    EdgeConnected,
    EdgeDisconnected,
    WakeDetected,
    WakeRouted,
    TurnStarted,
    TurnUserText,
    TurnFinished,
    TurnInterrupted,
    ToolCalled,
    ToolResult,
    SpeechRequest,
    SpeechStarted,
    SpeechFinished,
    AudioFocusRequest,
    AudioFocusGranted,
    AudioFocusLost,
    AudioStreamStarted,
    AudioStreamEnded,
    BodyIntent,
    PrivacyChanged,
    SchedulerDue,
)

BUS_EVENTS_BY_TOPIC: dict[str, type[BusEvent]] = {m.topic: m for m in BUS_EVENT_TYPES}


def event_type_for_topic(topic: str) -> type[BusEvent] | None:
    """The payload model for a topic, or None for skill-defined `skill.*` topics."""
    if topic.startswith(SchedulerDue.topic + "."):
        return SchedulerDue
    return BUS_EVENTS_BY_TOPIC.get(topic)
