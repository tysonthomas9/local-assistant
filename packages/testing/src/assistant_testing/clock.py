"""A manually driven clock with the same shape as `assistant_core.clock.Clock`."""

import asyncio
import heapq
import itertools
from datetime import UTC, datetime, timedelta


class FakeClock:
    """Time only moves when a test calls `advance`; sleepers wake in deadline order."""

    def __init__(self, start: datetime | None = None, start_mono_ns: int = 0) -> None:
        self._wall0 = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._mono0 = start_mono_ns
        self._elapsed_ns = 0
        self._order = itertools.count()
        self._sleepers: list[tuple[int, int, asyncio.Future[None]]] = []

    def monotonic_ns(self) -> int:
        return self._mono0 + self._elapsed_ns

    def now(self) -> datetime:
        return self._wall0 + timedelta(microseconds=self._elapsed_ns // 1000)

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        deadline = self._elapsed_ns + int(seconds * 1e9)
        heapq.heappush(self._sleepers, (deadline, next(self._order), future))
        await future

    async def advance(self, seconds: float) -> None:
        """Move time forward, waking every sleeper whose deadline has passed."""
        target = self._elapsed_ns + int(seconds * 1e9)
        while self._sleepers and self._sleepers[0][0] <= target:
            deadline, _, future = heapq.heappop(self._sleepers)
            self._elapsed_ns = deadline
            if not future.done():
                future.set_result(None)
            await asyncio.sleep(0)
        self._elapsed_ns = target
        await asyncio.sleep(0)
