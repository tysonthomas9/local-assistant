"""The turn log: one record per turn, kept in memory and served by the admin endpoint.

A record holds the input (for a voice turn, its STT transcript), the reply, every turn state
with its time (ms since the turn started), the LLM timings (queued, time to first token,
total), the speech timings (`speech`: STT, the TTS's first audio, the reply's first audio,
the voice), what was heard of a reply cut by a barge-in (`truncated`: `played_ms`,
`heard_text`, ...) and the outcome: `finished`, `interrupted`, `error` or `abandoned` (the
edge went away).
"""

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Literal

from assistant_brain.console import emit

TURN_LOG_SIZE = 500
TurnState = Literal["listening", "thinking", "speaking", "idle"]
Outcome = Literal["finished", "interrupted", "error", "abandoned"]


@dataclass
class TurnRecord:
    turn_id: str
    session_id: str
    device_id: str
    assistant: str
    engine: str
    kind: Literal["text", "voice", "proactive"]
    input_text: str | None = None
    input_audio_ms: int = 0
    reply_text: str = ""
    reply_audio_ms: int = 0
    states: list[dict[str, object]] = field(default_factory=list)
    llm: dict[str, object] = field(default_factory=dict)
    speech: dict[str, object] = field(default_factory=dict)
    truncated: dict[str, object] | None = None
    outcome: Outcome | None = None
    error: str | None = None
    started_mono: float = field(default_factory=time.monotonic)
    total_ms: float | None = None

    def state(self, state: TurnState) -> None:
        at = round((time.monotonic() - self.started_mono) * 1000, 1)
        self.states.append({"state": state, "t_ms": at})
        emit("TURN-STATE", device=self.device_id, turn=self.turn_id, state=state, t_ms=at)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("started_mono")
        return data


class TurnLog:
    def __init__(self) -> None:
        self.records: deque[TurnRecord] = deque(maxlen=TURN_LOG_SIZE)

    def start(self, record: TurnRecord) -> TurnRecord:
        self.records.append(record)
        emit(
            "TURN-START",
            {"input": record.input_text, "audio_ms": record.input_audio_ms},
            device=record.device_id,
            turn=record.turn_id,
            kind=record.kind,
            engine=record.engine,
        )
        return record

    def end(self, record: TurnRecord, outcome: Outcome, error: str | None = None) -> None:
        if record.outcome is not None:
            return
        record.outcome = outcome
        record.error = error
        record.total_ms = round((time.monotonic() - record.started_mono) * 1000, 1)
        emit(
            "TURN-END",
            record.as_dict(),
            device=record.device_id,
            turn=record.turn_id,
            outcome=outcome,
        )

    def as_list(self) -> list[dict[str, object]]:
        return [r.as_dict() for r in self.records]
