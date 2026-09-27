"""Injectable clock. Production code takes a `Clock` instead of calling time directly."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def monotonic_ns(self) -> int:
        """Monotonic nanoseconds (for `ts_mono_ns`, timeouts and latency)."""
        ...

    def now(self) -> datetime:
        """Timezone-aware UTC wall time (for scheduling)."""
        ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real clock."""

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
