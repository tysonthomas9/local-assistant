import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from assistant_contracts.capabilities import Capabilities
from assistant_contracts.frames import (
    HEADER_SIZE,
    MAX_SEQ,
    MAX_TS_US,
    Frame,
    FrameCodec,
    FrameError,
    FrameKind,
    FrameKindNotNegotiated,
    decode_frame,
    encode_frame,
    opus_negotiated,
)

pytestmark = pytest.mark.unit

V1_KINDS = [FrameKind.MIC_PCM, FrameKind.OUT_PCM, FrameKind.JPEG_CHUNK, FrameKind.SOUND_CLIP]


def test_header_is_14_bytes_little_endian() -> None:
    frame = Frame(FrameKind.MIC_PCM, stream=7, seq=0x01020304, capture_ts_us=0x1122, payload=b"")
    raw = encode_frame(frame)
    assert HEADER_SIZE == 14
    assert raw == bytes([0x01, 7, 0x04, 0x03, 0x02, 0x01]) + (0x1122).to_bytes(8, "little")


@given(
    kind=st.sampled_from(V1_KINDS),
    stream=st.integers(0, 255),
    seq=st.integers(0, MAX_SEQ),
    ts=st.integers(0, MAX_TS_US),
    samples=st.binary(max_size=4096),
)
def test_round_trip_property(
    kind: FrameKind, stream: int, seq: int, ts: int, samples: bytes
) -> None:
    payload = samples if kind in (FrameKind.JPEG_CHUNK, FrameKind.SOUND_CLIP) else samples * 2
    frame = Frame(kind, stream=stream, seq=seq, capture_ts_us=ts, payload=payload)
    raw = encode_frame(frame)
    assert len(raw) == HEADER_SIZE + len(payload)
    assert decode_frame(raw) == frame
    assert decode_frame(memoryview(raw)) == frame


def test_codec_follows_capability_negotiation() -> None:
    frame = Frame(FrameKind.OPUS, stream=1, seq=1, capture_ts_us=1, payload=b"\x01")
    off = FrameCodec(opus=opus_negotiated(Capabilities(opus=True), brain_accepts_opus=False))
    with pytest.raises(FrameKindNotNegotiated):
        off.encode(frame)
    on = FrameCodec(opus=opus_negotiated(Capabilities(opus=True), brain_accepts_opus=True))
    assert on.decode(on.encode(frame)) == frame
    assert not opus_negotiated(Capabilities(), brain_accepts_opus=True)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\x01\x00\x00", "shorter than"),
        (struct.pack("<BBIQ", 0x06, 0, 0, 0), "unknown frame kind 0x06"),
        (struct.pack("<BBIQ", 0x00, 0, 0, 0), "unknown frame kind 0x00"),
        (struct.pack("<BBIQ", 0x01, 0, 0, 0) + b"\x00", "whole s16 samples"),
    ],
)
def test_decode_rejects_malformed(raw: bytes, message: str) -> None:
    with pytest.raises(FrameError, match=message):
        decode_frame(raw)


@pytest.mark.parametrize(
    "frame",
    [
        Frame(FrameKind.MIC_PCM, stream=256, seq=0, capture_ts_us=0, payload=b""),
        Frame(FrameKind.MIC_PCM, stream=0, seq=MAX_SEQ + 1, capture_ts_us=0, payload=b""),
        Frame(FrameKind.MIC_PCM, stream=0, seq=0, capture_ts_us=-1, payload=b""),
        Frame(FrameKind.OUT_PCM, stream=0, seq=0, capture_ts_us=0, payload=b"\x00"),
    ],
)
def test_encode_rejects_out_of_range(frame: Frame) -> None:
    with pytest.raises(FrameError):
        encode_frame(frame)
