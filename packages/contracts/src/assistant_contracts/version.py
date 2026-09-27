"""EdgeLink protocol version and WebSocket close codes."""

from enum import IntEnum
from typing import Final

PROTOCOL_VERSION: Final = "1.0"
"""Major.minor. Any message JSON Schema change needs a bump (the contract test enforces it)."""

PROTOCOL_MAJOR: Final = int(PROTOCOL_VERSION.split(".", 1)[0])
"""The `v` field every JSON message carries. A peer with another major is refused."""


class CloseCode(IntEnum):
    """Application WebSocket close codes used by EdgeLink (4000-4999 is the private range)."""

    VERSION_MISMATCH = 4001
    """The peer speaks another major protocol version."""

    AUTH_REFUSED = 4003
    """Missing, unknown or revoked device token."""


def is_compatible(proto: str) -> bool:
    """True if a peer's `hello.proto` ("major.minor") has our major version."""
    major, _, _ = proto.partition(".")
    return major.isdigit() and int(major) == PROTOCOL_MAJOR
