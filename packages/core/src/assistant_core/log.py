"""structlog setup: JSON lines (for journald) or a console renderer for development.

Bind `service`, `device_id`, `assistant`, `session_id`, `turn_id` and `trace_id` with
`structlog.contextvars.bind_contextvars` so they appear on every line.
"""

import logging
import sys
from typing import Literal, TextIO

import structlog

LogFormat = Literal["json", "console"]


def configure_logging(
    *,
    service: str,
    level: str = "info",
    fmt: LogFormat = "json",
    stream: TextIO | None = None,
) -> None:
    """Configure structlog (and route stdlib logging through the same level)."""
    numeric = logging.getLevelNamesMapping()[level.upper()]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=stream or sys.stderr),
        cache_logger_on_first_use=False,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service)
    logging.basicConfig(level=numeric, stream=stream or sys.stderr, force=True)


def get_logger(name: str | None = None) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger
