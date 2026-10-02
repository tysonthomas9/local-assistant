"""BasicTurnEngine: our own engine (grows into the DIY engine; proposal amendment F).

Two modes in the skeleton:

- `echo`: the reply is the input. Text comes back as text; speech comes back as the same audio
  (16 kHz), so the whole voice path (mic window -> uplink -> reply stream -> playback clock)
  runs without any model.
- `basic`: the reply comes from the real LLM through `LlmClient` (the priority gate, class
  `voice` for the user's turns and `proactive` for the assistant's own), with the turn's
  assistant's persona and the session's recent history. With speech (`[engine].speech`, an
  `SttClient` and a `TtsClient` on the speech server):
  - speech input is transcribed first (`InputTranscript`; nothing heard: no reply);
  - the LLM's streamed text is cut into sentences as it arrives, and each sentence is spoken
    by the TTS in the assistant's voice (`voice.speaker`: Jarvis -> Ryan, Marvin -> Eric)
    while the LLM goes on: `ReplyText(sentence)` then its `ReplyAudio` chunks;
  - a barge-in (`interrupt(played_ms)`) keeps only what was heard in the conversation: whole
    sentences whose audio was played, then the words of the cut sentence in proportion to how
    much of it played. The cut is noted in the turn's `TurnMetrics.truncation`.
  Without speech, replies are text (speak text) and speech input gets a spoken notice.
"""

import asyncio
import contextlib
import re
import time
import unicodedata
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from assistant_brain.adapters.llm import ChatResult, LlmClient, LlmUnavailable
from assistant_brain.adapters.stt import SpeechUnavailable, SttClient
from assistant_brain.adapters.tts import Synthesis, TtsClient
from assistant_brain.engine import (
    MIC_RATE,
    EngineEvent,
    EngineUnavailable,
    InputTranscript,
    ReplyAudio,
    ReplyText,
    SessionInfo,
    TurnMetrics,
    UserTurn,
)
from assistant_core.config import AssistantDef

LLM_DOWN_REPLY = "Sorry, I can't reach my language model right now. Please try again in a moment."
SPEECH_DOWN_REPLY = "Sorry, I can't reach my speech server right now. Please try again in a moment."
NO_STT_REPLY = "Sorry, I can't understand speech yet; please type to me."
REPLY_MAX_TOKENS = 400
MIN_SENTENCE_CHARS = 12
"""A sentence shorter than this is joined to the next one ("Hi. ..."), so a TTS request is
never just one or two words."""
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*(?=\s)|\n+")


def persona_prompt(assistant: AssistantDef) -> str:
    """The system prompt (TODO(phase 4): config/personas/<persona>.md and memory blocks)."""
    name = assistant.id.capitalize()
    return (
        f"You are {name}, a friendly little desk robot (a Reachy Mini) and voice assistant. "
        "Your replies are spoken aloud: answer in one to three short sentences of plain text, "
        "with no markdown, lists or emoji."
    )


class SentenceSplitter:
    """Cuts streamed text into sentences (at . ! ? … followed by a space, or a newline)."""

    def __init__(self, min_chars: int = MIN_SENTENCE_CHARS) -> None:
        self.min_chars = min_chars
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        sentences: list[str] = []
        while True:
            cut = next(
                (m.end() for m in _SENTENCE_END.finditer(self.buffer)
                 if len(self.buffer[: m.end()].strip()) >= self.min_chars),
                None,
            )  # fmt: skip
            if cut is None:
                return sentences
            sentence, self.buffer = self.buffer[:cut].strip(), self.buffer[cut:].lstrip()
            if sentence:
                sentences.append(sentence)

    def flush(self) -> str:
        rest, self.buffer = self.buffer.strip(), ""
        return rest


def speakable(text: str) -> str:
    """The text the TTS gets: no markdown marks or emoji; "" if nothing is left to say."""
    text = re.sub(r"[*_#`~>|]+", " ", text)
    text = "".join(
        ch for ch in text if not unicodedata.category(ch).startswith("S") or ch in "$%&+=<@°€£"
    )
    text = " ".join(text.split())
    return text if any(ch.isalnum() for ch in text) else ""


def heard_text(segments: list["_Segment"], played_ms: int) -> str:
    """What of the spoken sentences was heard after `played_ms` of their audio."""
    heard: list[str] = []
    start = 0
    for segment in segments:
        if start + segment.audio_ms <= played_ms:
            heard.append(segment.text)
            start += segment.audio_ms
            continue
        if segment.audio_ms > 0 and played_ms > start:
            words = segment.text.split()
            count = int(len(words) * (played_ms - start) / segment.audio_ms)
            if count:
                heard.append(" ".join(words[:count]))
        break
    return " ".join(heard)


@dataclass
class _Segment:
    text: str
    audio_ms: int = 0
    """Audio of this sentence handed to the edge so far."""


@dataclass
class _Reply:
    """The reply being spoken, kept for `interrupt`."""

    user: dict[str, Any]
    metrics: TurnMetrics
    result: ChatResult
    segments: list[_Segment] = field(default_factory=list)


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
    def __init__(
        self,
        info: SessionInfo,
        llm: LlmClient,
        stt: SttClient | None = None,
        tts: TtsClient | None = None,
    ) -> None:
        self.info = info
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.history: list[dict[str, Any]] = []
        self.max_messages = 2 * info.assistant.memory.history_turns
        self._reply: _Reply | None = None

    async def push_audio(self, pcm: bytes) -> None:
        del pcm  # the whole utterance is transcribed at the end of the turn (respond)

    async def respond(self, turn: UserTurn, metrics: TurnMetrics) -> AsyncIterator[EngineEvent]:
        assistant = turn.assistant or self.info.assistant
        started = time.monotonic()
        text = turn.text
        if not text and turn.audio:
            if self.stt is None:
                yield ReplyText(NO_STT_REPLY)
                return
            try:
                transcript = await self.stt.transcribe(turn.audio, MIC_RATE)
            except SpeechUnavailable as exc:
                raise EngineUnavailable(str(exc), SPEECH_DOWN_REPLY) from exc
            metrics.stt_ms = transcript.ms
            metrics.extra["stt_request_id"] = transcript.request_id
            text = transcript.text
            yield InputTranscript(text)
        if not text or not any(ch.isalnum() for ch in text):
            return  # nothing was said
        user = {"role": "user", "content": text}
        messages = [
            {"role": "system", "content": persona_prompt(assistant)},
            *self.history[-self.max_messages :],
            user,
        ]
        result = ChatResult(f"llm-{turn.turn_id}")
        metrics.llm_request_id = result.request_id
        reply = _Reply(user, metrics, result)
        self._reply = reply
        deltas = self._llm_text(messages, turn, result, metrics)
        try:
            if self.tts is None:
                async for delta in deltas:
                    yield ReplyText(delta)
            else:
                async for event in self._speak(deltas, assistant, reply, started):
                    yield event
        except LlmUnavailable as exc:
            raise EngineUnavailable(str(exc), LLM_DOWN_REPLY) from exc
        except SpeechUnavailable as exc:
            raise EngineUnavailable(str(exc), SPEECH_DOWN_REPLY) from exc
        finally:
            metrics.llm_total_ms = result.total_ms
            metrics.llm_queued_ms = result.queued_ms
        self._reply = None
        self.history += [user, {"role": "assistant", "content": result.text}]

    async def _llm_text(
        self,
        messages: list[dict[str, Any]],
        turn: UserTurn,
        result: ChatResult,
        metrics: TurnMetrics,
    ) -> AsyncIterator[str]:
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
            yield delta.text

    async def _speak(
        self, deltas: AsyncIterator[str], assistant: AssistantDef, reply: _Reply, started: float
    ) -> AsyncIterator[EngineEvent]:
        """Sentences from the streaming LLM, each spoken as soon as it is complete, while the
        LLM goes on (a producer task feeds a queue)."""
        assert self.tts is not None
        tts, metrics = self.tts, reply.metrics
        voice = assistant.voice.speaker.lower()
        metrics.voice = voice
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        failure: list[Exception] = []

        async def produce() -> None:
            splitter = SentenceSplitter()
            try:
                async for delta in deltas:
                    for sentence in splitter.feed(delta):
                        sentences.put_nowait(sentence)
                if rest := splitter.flush():
                    sentences.put_nowait(rest)
            except Exception as exc:
                failure.append(exc)
            finally:
                sentences.put_nowait(None)

        producer = asyncio.create_task(produce(), name="llm-sentences")
        try:
            while (sentence := await sentences.get()) is not None:
                yield ReplyText(sentence + " ")
                spoken = speakable(sentence)
                if not spoken:
                    continue
                segment = _Segment(sentence)
                reply.segments.append(segment)
                synthesis = Synthesis(voice)
                metrics.tts_requests += 1
                async for pcm in tts.stream(spoken, voice, synthesis):
                    if metrics.first_audio_ms is None:
                        metrics.first_audio_ms = (time.monotonic() - started) * 1000
                        metrics.tts_first_audio_ms = synthesis.first_audio_ms
                    segment.audio_ms = synthesis.audio_ms
                    yield ReplyAudio(pcm, tts.rate)
            if failure:
                raise failure[0]
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    async def interrupt(self, played_ms: int | None) -> None:
        """Keep only what was heard of the cut reply in the conversation."""
        reply, self._reply = self._reply, None
        if reply is None or not reply.segments:
            return
        sent_ms = sum(s.audio_ms for s in reply.segments)
        heard = heard_text(reply.segments, sent_ms if played_ms is None else played_ms)
        reply.metrics.truncation = {
            "played_ms": played_ms,
            "sent_audio_ms": sent_ms,
            "heard_text": heard,
            "spoken_text": " ".join(s.text for s in reply.segments),
            "llm_text": reply.result.text,
        }
        self.history.append(reply.user)
        if heard:
            self.history.append({"role": "assistant", "content": heard})

    async def close(self) -> None:
        self.history.clear()


class BasicTurnEngine:
    def __init__(
        self,
        mode: Literal["echo", "basic"],
        llm: LlmClient | None = None,
        stt: SttClient | None = None,
        tts: TtsClient | None = None,
    ) -> None:
        if mode == "basic" and llm is None:
            raise ValueError("the basic engine's text mode needs an LLM client")
        self.name: Literal["echo", "basic"] = mode
        self.llm = llm
        self.stt = stt
        self.tts = tts

    async def open_session(self, info: SessionInfo) -> EchoSession | LlmSession:
        if self.name == "echo":
            return EchoSession()
        assert self.llm is not None
        return LlmSession(info, self.llm, self.stt, self.tts)

    async def aclose(self) -> None:
        for client in (self.stt, self.tts):
            if client is not None:
                await client.aclose()
