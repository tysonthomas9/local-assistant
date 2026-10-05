"""EdgeLink v1 contracts: JSON messages, binary frames, capabilities, Body Protocols, bus events.

This package depends on pydantic only and imports no other assistant package.
"""

from assistant_contracts.capabilities import BodyCapabilities, Capabilities
from assistant_contracts.expressions import ExpressName
from assistant_contracts.frames import (
    Frame,
    FrameCodec,
    FrameError,
    FrameKind,
    FrameKindNotNegotiated,
    decode_frame,
    encode_frame,
)
from assistant_contracts.messages import (
    MESSAGE_TYPES,
    EdgeLinkMessage,
    Envelope,
    dump_message,
    parse_message,
)
from assistant_contracts.version import PROTOCOL_MAJOR, PROTOCOL_VERSION, CloseCode

__all__ = [
    "MESSAGE_TYPES",
    "PROTOCOL_MAJOR",
    "PROTOCOL_VERSION",
    "BodyCapabilities",
    "Capabilities",
    "CloseCode",
    "EdgeLinkMessage",
    "Envelope",
    "ExpressName",
    "Frame",
    "FrameCodec",
    "FrameError",
    "FrameKind",
    "FrameKindNotNegotiated",
    "decode_frame",
    "dump_message",
    "encode_frame",
    "parse_message",
]
