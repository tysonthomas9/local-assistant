"""Which peer may send which message type and frame kind (proposal section 5.1)."""

from typing import Final, Literal

from assistant_contracts.frames import FrameKind
from assistant_contracts.messages import MESSAGE_TYPES, message_type_name

Side = Literal["edge", "brain"]

_DIRECTION: Final = {message_type_name(m): m.direction for m in MESSAGE_TYPES}

FRAME_SENDER: Final[dict[FrameKind, frozenset[Side]]] = {
    FrameKind.MIC_PCM: frozenset({"edge"}),
    FrameKind.OUT_PCM: frozenset({"brain"}),
    FrameKind.JPEG_CHUNK: frozenset({"edge"}),
    FrameKind.SOUND_CLIP: frozenset({"brain"}),
    FrameKind.OPUS: frozenset({"edge", "brain"}),
}


def message_allowed_from(sender: Side, type_name: str) -> bool:
    """True if `sender` may send messages of wire type `type_name`."""
    direction = _DIRECTION.get(type_name)
    if direction is None:
        return False
    return (
        direction == "both" or direction == f"{sender}_to_{'brain' if sender == 'edge' else 'edge'}"
    )


def frame_allowed_from(sender: Side, kind: FrameKind) -> bool:
    return sender in FRAME_SENDER[kind]
