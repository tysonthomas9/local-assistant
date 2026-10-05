"""EdgeLink v1 binary frames: a 14-byte little-endian header followed by the payload.

    kind u8 | stream u8 | seq u32 | capture_ts_us u64 | payload

Kinds: 0x01 mic PCM (edge->brain, s16le 16 kHz mono, 20 ms = 640 B), 0x02 output PCM
(brain->edge, rate from `speak.begin`, stream = speak stream_id), 0x03 JPEG snapshot chunk
(edge->brain, stream = slot), 0x04 WAV sound clip for the edge cache (brain->edge) and
0x05 Opus, which is reserved: the codec refuses it unless `opus` was negotiated.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from assistant_contracts.capabilities import Capabilities

HEADER: Final = struct.Struct("<BBIQ")
HEADER_SIZE: Final = HEADER.size
assert HEADER_SIZE == 14

MAX_SEQ: Final = 2**32 - 1
MAX_TS_US: Final = 2**64 - 1

MIC_FRAME_BYTES: Final = 640
"""20 ms of s16le 16 kHz mono."""


class FrameKind(IntEnum):
    MIC_PCM = 0x01
    OUT_PCM = 0x02
    JPEG_CHUNK = 0x03
    SOUND_CLIP = 0x04
    OPUS = 0x05


_PCM_KINDS: Final = frozenset({FrameKind.MIC_PCM, FrameKind.OUT_PCM})


class FrameError(ValueError):
    """A binary frame is malformed or not allowed."""


class FrameKindNotNegotiated(FrameError):
    """The frame kind exists but this connection did not negotiate it (0x05 Opus)."""


@dataclass(frozen=True, slots=True)
class Frame:
    kind: FrameKind
    stream: int
    seq: int
    capture_ts_us: int
    payload: bytes


def opus_negotiated(edge: Capabilities, brain_accepts_opus: bool) -> bool:
    """Opus is on only if the edge advertises it and the brain accepted it in `welcome`."""
    return edge.opus and brain_accepts_opus


def _check_kind(kind: int, *, opus: bool) -> FrameKind:
    try:
        frame_kind = FrameKind(kind)
    except ValueError:
        raise FrameError(f"unknown frame kind 0x{kind:02x}") from None
    if frame_kind is FrameKind.OPUS and not opus:
        raise FrameKindNotNegotiated("frame kind 0x05 (Opus) was not negotiated")
    return frame_kind


def _check_payload(kind: FrameKind, payload: bytes) -> None:
    if kind in _PCM_KINDS and len(payload) % 2:
        raise FrameError(f"{kind.name} payload must be whole s16 samples, got {len(payload)} B")


def encode_frame(frame: Frame, *, opus: bool = False) -> bytes:
    """Encode a frame. `opus=True` only when 0x05 was negotiated on this connection."""
    kind = _check_kind(int(frame.kind), opus=opus)
    if not 0 <= frame.stream <= 255:
        raise FrameError(f"stream {frame.stream} does not fit in a u8")
    if not 0 <= frame.seq <= MAX_SEQ:
        raise FrameError(f"seq {frame.seq} does not fit in a u32")
    if not 0 <= frame.capture_ts_us <= MAX_TS_US:
        raise FrameError(f"capture_ts_us {frame.capture_ts_us} does not fit in a u64")
    _check_payload(kind, frame.payload)
    return HEADER.pack(kind, frame.stream, frame.seq, frame.capture_ts_us) + frame.payload


def decode_frame(data: bytes | bytearray | memoryview, *, opus: bool = False) -> Frame:
    """Decode a binary WebSocket message. Raises `FrameError` if it is malformed."""
    if len(data) < HEADER_SIZE:
        raise FrameError(f"frame is {len(data)} B, shorter than the {HEADER_SIZE} B header")
    raw_kind, stream, seq, ts_us = HEADER.unpack_from(data)
    kind = _check_kind(raw_kind, opus=opus)
    payload = bytes(data[HEADER_SIZE:])
    _check_payload(kind, payload)
    return Frame(kind=kind, stream=stream, seq=seq, capture_ts_us=ts_us, payload=payload)


class FrameCodec:
    """Per-connection codec that remembers whether Opus was negotiated."""

    def __init__(self, *, opus: bool = False) -> None:
        self.opus: bool = opus

    def encode(self, frame: Frame) -> bytes:
        return encode_frame(frame, opus=self.opus)

    def decode(self, data: bytes | bytearray | memoryview) -> Frame:
        return decode_frame(data, opus=self.opus)
