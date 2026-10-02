"""The LLM priority gate (proposal amendment G): who may call the LLM server, and when.

- At most `max_concurrency` requests are in flight, across all sessions.
- `reserved_voice_slots` of them are never handed to a non-voice request: a proactive or
  background request is admitted only while more than that many slots are free.
- Waiters are served by class, voice > proactive > background, first come first served within
  a class.

Pure logic (no I/O), so its ordering rules are unit-tested; the feature `llm/priority_gate`
drives it with real requests. Every decision is recorded in `log` for the admin endpoint.
"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

RequestClass = Literal["voice", "proactive", "background"]
CLASSES: tuple[RequestClass, ...] = ("voice", "proactive", "background")
"""Highest priority first."""

GateEventKind = Literal["queued", "admitted", "released", "cancelled"]
LOG_SIZE = 2000


@dataclass(frozen=True, slots=True)
class GateEvent:
    kind: GateEventKind
    request_id: str
    cls: RequestClass
    in_flight: int
    """Requests in flight right after this event."""
    waiting: int
    t: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _Waiter:
    request_id: str
    cls: RequestClass
    admitted: asyncio.Future[None] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


class PriorityGate:
    def __init__(
        self,
        max_concurrency: int,
        reserved_voice_slots: int = 0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if not 0 <= reserved_voice_slots < max_concurrency:
            raise ValueError("reserved_voice_slots must be in [0, max_concurrency)")
        self.max_concurrency = max_concurrency
        self.reserved_voice_slots = reserved_voice_slots
        self.clock = clock
        self.in_flight: dict[str, RequestClass] = {}
        self.max_in_flight_seen = 0
        self._waiting: dict[RequestClass, deque[_Waiter]] = {c: deque() for c in CLASSES}
        self.log: deque[GateEvent] = deque(maxlen=LOG_SIZE)

    # ------------------------------------------------------------ state

    @property
    def waiting(self) -> int:
        return sum(len(q) for q in self._waiting.values())

    def snapshot(self) -> dict[str, object]:
        by_class = {c: sum(1 for v in self.in_flight.values() if v == c) for c in CLASSES}
        return {
            "max_concurrency": self.max_concurrency,
            "reserved_voice_slots": self.reserved_voice_slots,
            "in_flight": len(self.in_flight),
            "in_flight_by_class": by_class,
            "waiting": {c: [w.request_id for w in self._waiting[c]] for c in CLASSES},
            "max_in_flight_seen": self.max_in_flight_seen,
        }

    def _record(self, kind: GateEventKind, request_id: str, cls: RequestClass) -> None:
        self.log.append(
            GateEvent(kind, request_id, cls, len(self.in_flight), self.waiting, self.clock())
        )

    # ------------------------------------------------------------ admission

    def can_admit(self, cls: RequestClass) -> bool:
        """Whether a request of `cls` could take a slot right now (ignoring waiters)."""
        limit = self.max_concurrency
        if cls != "voice":
            limit -= self.reserved_voice_slots
        return len(self.in_flight) < limit

    def _admit(self, request_id: str, cls: RequestClass) -> None:
        self.in_flight[request_id] = cls
        self.max_in_flight_seen = max(self.max_in_flight_seen, len(self.in_flight))
        self._record("admitted", request_id, cls)

    def _dispatch(self) -> None:
        """Hand free slots to waiters: highest class first, FIFO within a class."""
        for cls in CLASSES:
            queue = self._waiting[cls]
            while queue and self.can_admit(cls):
                waiter = queue.popleft()
                if waiter.admitted.done():  # cancelled while waiting
                    continue
                self._admit(waiter.request_id, cls)
                waiter.admitted.set_result(None)
            if queue:
                return  # a higher class still waits: nothing below it may pass

    async def acquire(self, request_id: str, cls: RequestClass) -> None:
        """Wait for a slot. Cancelling the wait gives the place up (no slot is held)."""
        if request_id in self.in_flight:
            raise ValueError(f"request {request_id!r} already holds a slot")
        higher_or_same_waiting = any(self._waiting[c] for c in CLASSES[: CLASSES.index(cls) + 1])
        if not higher_or_same_waiting and self.can_admit(cls):
            self._admit(request_id, cls)
            return
        waiter = _Waiter(request_id, cls)
        self._waiting[cls].append(waiter)
        self._record("queued", request_id, cls)
        try:
            await waiter.admitted
        except asyncio.CancelledError:
            if request_id in self.in_flight:  # admitted just as we were cancelled
                self.release(request_id)
            else:
                with contextlib.suppress(ValueError):
                    self._waiting[cls].remove(waiter)
                self._record("cancelled", request_id, cls)
                self._dispatch()
            raise

    def release(self, request_id: str) -> None:
        cls = self.in_flight.pop(request_id, None)
        if cls is None:
            return
        self._record("released", request_id, cls)
        self._dispatch()

    @contextlib.asynccontextmanager
    async def slot(self, request_id: str, cls: RequestClass) -> AsyncIterator[None]:
        await self.acquire(request_id, cls)
        try:
            yield
        finally:
            self.release(request_id)
