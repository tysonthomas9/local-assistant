"""BasicTurnEngine: our own engine (grows into the DIY engine; proposal amendment F).

Two modes in the skeleton:

- `echo`: the reply is the input. Text comes back as text; speech comes back as the same audio
  (16 kHz), so the whole voice path (mic window -> uplink -> reply stream -> playback clock)
  runs without any model.
- text mode (`basic`): the reply comes from the real LLM through `LlmClient` (the priority
  gate, class `voice` for the user's turns and `proactive` for the assistant's own), with the
  assistant's persona and the session's recent history. Speech input needs STT, which arrives
  with task S8; until then it is answered with a spoken notice.
"""

from collections.abc import AsyncIterator
from typing import Any, Literal

from assistant_brain.adapters.llm import ChatResult, LlmClient, LlmUnavailable
from assistant_brain.engine import (
    MIC_RATE,
    EngineEvent,
    EngineUnavailable,
    ReplyAudio,
    ReplyText,
    SessionInfo,
    TurnMetrics,
    UserTurn,
)

LLM_DOWN_REPLY = "Sorry, I can't reach my language model right now. Please try again in a moment."
NO_STT_REPLY = "Sorry, I can't understand speech yet; please type to me."
REPLY_MAX_TOKENS = 400


def persona_prompt(info: SessionInfo) -> str:
    """The system prompt (TODO(phase 4): config/personas/<persona>.md and memory blocks)."""
    name = info.assistant.id.capitalize()
    return (
        f"You are {name}, a friendly little desk robot (a Reachy Mini) and voice assistant. "
        "Your replies are spoken aloud: answer in one to three short sentences of plain text, "
        "with no markdown, lists or emoji."
    )


class EchoSession:
    async def push_audio(self, pcm: bytes) -> None:
        del pcm

    async def respond(self, turn: UserTurn, metrics: TurnMetrics) -> AsyncIterator[EngineEvent]:
        del metrics
        if turn.text:
            yield ReplyText(turn.text)
        if turn.audio:
            yield ReplyAudio(turn.audio, MIC_RATE)

    async def interrupt(self, played_ms: int | None) -> None:
        del played_ms

    async def close(self) -> None:
        pass


class LlmSession:
    def __init__(self, info: SessionInfo, llm: LlmClient) -> None:
        self.info = info
        self.llm = llm
        self.history: list[dict[str, Any]] = []
        self.max_messages = 2 * info.assistant.memory.history_turns

    async def push_audio(self, pcm: bytes) -> None:
        del pcm  # S8: streamed to STT

    async def respond(self, turn: UserTurn, metrics: TurnMetrics) -> AsyncIterator[EngineEvent]:
        if not turn.text:
            yield ReplyText(NO_STT_REPLY)
            return
        user = {"role": "user", "content": turn.text}
        messages = [
            {"role": "system", "content": persona_prompt(self.info)},
            *self.history[-self.max_messages :],
            user,
        ]
        result = ChatResult(f"llm-{turn.turn_id}")
        metrics.llm_request_id = result.request_id
        try:
            async for delta in self.llm.stream_chat(
                messages,
                turn.request_class,
                result=result,
                max_tokens=REPLY_MAX_TOKENS,
                label=f"turn {turn.turn_id}",
            ):
                if metrics.llm_ttft_ms is None:
                    metrics.llm_ttft_ms = result.ttft_ms
                    metrics.llm_queued_ms = result.queued_ms
                yield ReplyText(delta.text)
        except LlmUnavailable as exc:
            raise EngineUnavailable(str(exc), LLM_DOWN_REPLY) from exc
        finally:
            metrics.llm_total_ms = result.total_ms
            metrics.llm_queued_ms = result.queued_ms
        self.history += [user, {"role": "assistant", "content": result.text}]

    async def interrupt(self, played_ms: int | None) -> None:
        """TODO(S8): truncate the last assistant message to what was heard (`played_ms`)."""
        del played_ms

    async def close(self) -> None:
        self.history.clear()


class BasicTurnEngine:
    def __init__(self, mode: Literal["echo", "basic"], llm: LlmClient | None = None) -> None:
        if mode == "basic" and llm is None:
            raise ValueError("the basic engine's text mode needs an LLM client")
        self.name: Literal["echo", "basic"] = mode
        self.llm = llm

    async def open_session(self, info: SessionInfo) -> EchoSession | LlmSession:
        if self.name == "echo":
            return EchoSession()
        assert self.llm is not None
        return LlmSession(info, self.llm)

    async def aclose(self) -> None:
        pass
