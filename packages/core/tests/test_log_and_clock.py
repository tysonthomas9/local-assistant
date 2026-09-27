import io
import json
from datetime import UTC

import pytest
import structlog

from assistant_core.clock import Clock, SystemClock
from assistant_core.log import configure_logging, get_logger
from assistant_testing import FakeClock

pytestmark = pytest.mark.unit


def test_json_logs_carry_service_and_bound_context() -> None:
    stream = io.StringIO()
    configure_logging(service="brain", level="info", fmt="json", stream=stream)
    structlog.contextvars.bind_contextvars(session_id="s-1")
    get_logger().debug("hidden")
    get_logger().info("turn.started", turn_id="t-1")
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(lines) == 1
    assert lines[0]["event"] == "turn.started"
    assert lines[0]["service"] == "brain"
    assert lines[0]["session_id"] == "s-1"
    assert lines[0]["level"] == "info"
    structlog.contextvars.clear_contextvars()


def test_clocks_share_one_shape() -> None:
    assert isinstance(SystemClock(), Clock)
    assert isinstance(FakeClock(), Clock)
    assert SystemClock().now().tzinfo is UTC


async def test_fake_clock_wakes_sleepers_in_order() -> None:
    import asyncio

    clock = FakeClock()
    woke: list[str] = []

    async def sleeper(name: str, seconds: float) -> None:
        await clock.sleep(seconds)
        woke.append(name)

    tasks = [asyncio.create_task(sleeper("b", 2)), asyncio.create_task(sleeper("a", 1))]
    await asyncio.sleep(0)
    start = clock.now()
    await clock.advance(1.5)
    assert woke == ["a"]
    await clock.advance(1)
    assert woke == ["a", "b"]
    assert clock.monotonic_ns() == 2_500_000_000
    assert (clock.now() - start).total_seconds() == 2.5
    await asyncio.gather(*tasks)
