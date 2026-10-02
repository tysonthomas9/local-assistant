"""A paced PCM player with a playback clock, shared by every edge body.

The brain streams speech faster than real time. A body must not hand all of it to its audio
device at once: a barge-in `flush` would then have seconds of queued audio to drop, and the
`played_ms` it reports would be a guess. `PacedPlayer` keeps only `lead_ms` of audio queued
in the device (`PcmSink.write`) and feeds it 20 ms at a time, so:

- `flush` stops the sound within about `lead_ms` plus the device's own buffer;
- the clock is exact up to that lead: `played_ms = written_ms - (play_until - now)`, where
  `play_until` is when the device runs out of what it was given (it restarts at `now` after a
  gap, so silence between chunks is not counted).

Streams play one after another in the order they started (`speak.begin`). An empty `play`
chunk marks the end of a stream (`speak.end`); `done` is reported once its last sample has
had time to play. Events: `started` (first chunk written), `progress` (every
`progress_ms` of played audio, 200 ms by default), `done`, `flushed`.

Pure asyncio, no audio library: the sink converts the s16le mono PCM to what its device takes.
"""

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

PlaybackState = Literal["started", "progress", "done", "flushed"]

CHUNK_MS = 20
LEAD_MS = 120
PROGRESS_MS = 200


class PcmSink(Protocol):
    """Where a `PacedPlayer` writes audio (an audio device, or nothing)."""

    def write(self, pcm: bytes, rate: int) -> None:
        """Queue s16le mono PCM at `rate` for playback; must not block for long."""
        ...

    def clear(self) -> None:
        """Drop whatever the device still has queued (barge-in)."""
        ...


@dataclass(frozen=True, slots=True)
class ClockEvent:
    stream_id: int
    played_ms: int
    state: PlaybackState


@dataclass
class _Stream:
    stream_id: int
    rate: int
    pending: bytearray = field(default_factory=bytearray)
    ended: bool = False
    written_ms: float = 0.0
    started: bool = False
    next_progress_ms: int = PROGRESS_MS


class PacedPlayer:
    def __init__(
        self,
        sink: PcmSink,
        *,
        lead_ms: int = LEAD_MS,
        chunk_ms: int = CHUNK_MS,
        progress_ms: int = PROGRESS_MS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sink = sink
        self.lead_ms = lead_ms
        self.chunk_ms = chunk_ms
        self.progress_ms = progress_ms
        self.clock = clock
        self._streams: deque[_Stream] = deque()
        self._by_id: dict[int, _Stream] = {}
        self._play_until = 0.0
        self._events: asyncio.Queue[ClockEvent] = asyncio.Queue()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------ the AudioIO side

    async def play(self, stream_id: int, pcm: bytes, rate: int) -> None:
        """Queue PCM of `stream_id`; an empty chunk ends the stream."""
        stream = self._by_id.get(stream_id)
        if stream is None:
            if not pcm:
                return
            stream = _Stream(stream_id, rate, next_progress_ms=self.progress_ms)
            self._by_id[stream_id] = stream
            self._streams.append(stream)
        if pcm:
            stream.pending += pcm
        else:
            stream.ended = True
        self._ensure_running()
        self._wake.set()

    async def flush(self, stream_id: int | None = None) -> int:
        """Stop and drop `stream_id` (or everything); returns the played ms of the current one.

        Reports `flushed` for every dropped stream.
        """
        now = self.clock()
        current = self._streams[0] if self._streams else None
        played = self._played_ms(current, now) if current is not None else 0
        targets = [s for s in self._streams if stream_id is None or s.stream_id == stream_id]
        if current is not None and current in targets:
            self.sink.clear()
            self._play_until = now
        for stream in targets:
            self._streams.remove(stream)
            self._by_id.pop(stream.stream_id, None)
            ms = played if stream is current else 0
            self._events.put_nowait(ClockEvent(stream.stream_id, ms, "flushed"))
        self._wake.set()
        return played if current is not None and current in targets else 0

    async def events(self) -> AsyncIterator[ClockEvent]:
        while True:
            yield await self._events.get()

    @property
    def active(self) -> int | None:
        """The stream now playing (or about to), if any."""
        return self._streams[0].stream_id if self._streams else None

    def played_ms(self) -> int:
        return self._played_ms(self._streams[0], self.clock()) if self._streams else 0

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ------------------------------------------------------------ pacing

    def _played_ms(self, stream: _Stream, now: float) -> int:
        queued_ms = max(0.0, self._play_until - now) * 1000 if stream.started else 0.0
        return max(0, int(stream.written_ms - queued_ms))

    def _ensure_running(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._pump(), name="paced-player")

    async def _pump(self) -> None:
        tick = self.chunk_ms / 1000
        while True:
            if not self._streams:
                self._wake.clear()
                await self._wake.wait()
                continue
            stream = self._streams[0]
            now = self.clock()
            ahead_ms = max(0.0, self._play_until - now) * 1000
            chunk_bytes = stream.rate * 2 * self.chunk_ms // 1000
            if stream.pending and ahead_ms < self.lead_ms:
                chunk = bytes(stream.pending[:chunk_bytes])
                del stream.pending[:chunk_bytes]
                self.sink.write(chunk, stream.rate)
                ms = len(chunk) / 2 / stream.rate * 1000
                self._play_until = max(self._play_until, now) + ms / 1000
                stream.written_ms += ms
                if not stream.started:
                    stream.started = True
                    self._events.put_nowait(ClockEvent(stream.stream_id, 0, "started"))
                continue
            played = self._played_ms(stream, now)
            if stream.started and played >= stream.next_progress_ms:
                self._events.put_nowait(ClockEvent(stream.stream_id, played, "progress"))
                stream.next_progress_ms = (played // self.progress_ms + 1) * self.progress_ms
            if stream.ended and not stream.pending and now >= self._play_until:
                self._streams.popleft()
                self._by_id.pop(stream.stream_id, None)
                self._events.put_nowait(
                    ClockEvent(stream.stream_id, int(stream.written_ms), "done")
                )
                continue
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(tick / 2):
                    await self._wake.wait()
