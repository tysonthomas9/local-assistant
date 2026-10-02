"""The `TurnEngine` Protocol: what turns a user's input into the assistant's reply.

Exactly two implementations are planned, and both fit this Protocol unchanged:

- `basic` (`engine.basic.BasicTurnEngine`): our own pipeline. In the skeleton it echoes
  (`echo` mode) or asks the LLM (text mode); it grows into the DIY engine (edge + brain VAD,
  Smart Turn, our speech server's STT and TTS, vLLM) that must beat `realtime`.
- `realtime` (`engine.realtime`, phase 2): HF speech-to-speech over the OpenAI Realtime API,
  the production engine from phase 2. `push_audio` maps to `input_audio_buffer.append`,
  `respond` to `input_audio_buffer.commit` + `response.create` (its server VAD off: the
  DialogManager decides when a turn ends), the reply's `response.audio.delta` /
  `response.text.delta` to `ReplyAudio` / `ReplyText`, and `interrupt(played_ms)` to
  `response.cancel` + `conversation.item.truncate(audio_end_ms=played_ms)`.

The DialogManager owns turn state, attention and the link; an engine only produces the reply.
One `EngineSession` per EdgeSession keeps that session's conversation context.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

from assistant_brain.adapters.priority_gate import RequestClass
from assistant_core.config import AssistantDef

MIC_RATE = 16000
"""Uplink audio (binary 0x01): s16le mono 16 kHz."""


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    device_id: str
    assistant: AssistantDef


@dataclass(frozen=True)
class UserTurn:
    """One turn's input: typed text, or the user's speech (`MIC_RATE` PCM), or both."""

    turn_id: str
    text: str | None = None
    audio: bytes | None = None
    request_class: RequestClass = "voice"
    """`voice` for the user's turns, `proactive` for speech the assistant starts itself."""


@dataclass(frozen=True)
class ReplyText:
    """Reply text, in order. Until S8 adds TTS it reaches the edge as speak text."""

    text: str


@dataclass(frozen=True)
class ReplyAudio:
    """Reply audio: s16le mono PCM at `rate`, sent as binary 0x02 frames."""

    pcm: bytes
    rate: int


@dataclass
class TurnMetrics:
    """Filled in by the engine while it answers; the DialogManager puts it in the turn log."""

    llm_request_id: str | None = None
    llm_queued_ms: float | None = None
    llm_ttft_ms: float | None = None
    llm_total_ms: float | None = None
    extra: dict[str, object] = field(default_factory=dict)


EngineEvent = ReplyText | ReplyAudio


class EngineUnavailable(Exception):
    """A model server the engine needs is down; `spoken` is what to tell the user."""

    def __init__(self, detail: str, spoken: str) -> None:
        super().__init__(detail)
        self.spoken = spoken


class EngineSession(Protocol):
    async def push_audio(self, pcm: bytes) -> None:
        """The user's audio as it arrives (inside mic windows)."""
        ...

    def respond(self, turn: UserTurn, metrics: TurnMetrics) -> AsyncIterator[EngineEvent]:
        """Answer one turn, streaming the reply. Raises `EngineUnavailable` when a model
        server is down. Cancelling the iteration cancels the reply (LLM, TTS, ...)."""
        ...

    async def interrupt(self, played_ms: int | None) -> None:
        """The user barged in: forget what was not heard (`played_ms` of the reply played)."""
        ...

    async def close(self) -> None: ...


class TurnEngine(Protocol):
    @property
    def name(self) -> Literal["echo", "basic", "realtime"]: ...

    async def open_session(self, info: SessionInfo) -> EngineSession: ...

    async def aclose(self) -> None: ...
