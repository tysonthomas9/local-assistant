import io
import json
from datetime import UTC

import pytest
import structlog

from assistant_core.clock import Clock, SystemClock
from assistant_core.log import configure_logging, get_logger

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


def test_system_clock_is_a_clock() -> None:
    assert isinstance(SystemClock(), Clock)
    assert SystemClock().now().tzinfo is UTC
    assert SystemClock().monotonic_ns() > 0
