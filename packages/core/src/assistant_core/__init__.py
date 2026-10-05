"""Shared runtime plumbing: layered config, structlog setup, injectable clock."""

from assistant_core.clock import Clock, SystemClock
from assistant_core.config import (
    AssistantConfig,
    AssistantDef,
    ConfigError,
    LlmConfig,
    load_assistant,
    load_config,
    parse_cli_overrides,
)
from assistant_core.log import configure_logging, get_logger

__all__ = [
    "AssistantConfig",
    "AssistantDef",
    "Clock",
    "ConfigError",
    "LlmConfig",
    "SystemClock",
    "configure_logging",
    "get_logger",
    "load_assistant",
    "load_config",
    "parse_cli_overrides",
]
