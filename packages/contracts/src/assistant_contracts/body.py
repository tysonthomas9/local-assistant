"""Body capability Protocols (proposal section 5.2), implemented inside the edge agent.

Bodies register through the `assistant.bodies` entry point group, e.g.
`reachy = assistant_robot_reachy:ReachyBody`.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from assistant_contracts.capabilities import BodyCapabilities
from assistant_contracts.common import Aec, AttentionState, LookTarget

BODY_ENTRY_POINT_GROUP = "assistant.bodies"

Attention = AttentionState


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """20 ms of s16le 16 kHz mono microphone audio, echo cancellation already applied."""

    pcm: bytes
    capture_ts_us: int
    rate: int = 16000


@dataclass(frozen=True, slots=True)
class PlaybackEvent:
    """The real playback clock of one output stream."""

    stream_id: int
    played_ms: int
    state: Literal["started", "progress", "done", "flushed"]


@dataclass(frozen=True, slots=True)
class BodyEvent:
    kind: Literal["button", "touch", "imu_tap", "doa"]
    data: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class AudioIO(Protocol):
    aec: Aec

    def capture(self) -> AsyncIterator[AudioFrame]:
        """20 ms s16 16 kHz mono frames, AEC applied."""
        ...

    async def play(self, stream_id: int, pcm: bytes, rate: int) -> None: ...

    async def flush(self, stream_id: int | None = None) -> int:
        """Stop and drop queued audio (one stream or all); returns played_ms."""
        ...

    def playback_events(self) -> AsyncIterator[PlaybackEvent]:
        """The real playback clock."""
        ...


@runtime_checkable
class Motion(Protocol):
    async def attention(self, state: Attention, assistant: str | None) -> None: ...

    async def express(self, name: str, intensity: float = 1.0) -> bool: ...

    async def look_at(self, target: LookTarget) -> bool: ...


@runtime_checkable
class Camera(Protocol):
    async def snapshot(self, max_side: int = 1024) -> bytes:
        """A JPEG whose longer side is at most `max_side`."""
        ...


@runtime_checkable
class Body(Protocol):
    kind: str
    """"reachy" or "console"."""
    audio: AudioIO
    motion: Motion | None
    camera: Camera | None

    async def start(self) -> BodyCapabilities: ...

    async def stop(self) -> None: ...

    def events(self) -> AsyncIterator[BodyEvent]:
        """Button, DOA and IMU tap events."""
        ...


@dataclass(frozen=True, slots=True)
class BodyHealth:
    """A change in whether the body's hardware is reachable (e.g. the robot daemon died)."""

    ok: bool
    detail: str = ""


@runtime_checkable
class ReportsHealth(Protocol):
    """Optional: a body whose hardware can go away and come back while the edge runs.

    The edge reports `ok=False` to the brain as `error{code: "body_unavailable"}` and keeps
    running; the body recovers on its own once its hardware is back (`ok=True`).
    """

    def health(self) -> AsyncIterator[BodyHealth]: ...
