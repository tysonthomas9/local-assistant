"""The paced player's gentle barge-in stop (`PacedPlayer.stop_after`), on its real clock."""

import asyncio
import math
from array import array

import pytest

from assistant_core.playback import ClockEvent, PacedPlayer, fade_out

pytestmark = pytest.mark.unit

RATE = 16000


class _Recorder:
    """A sink that keeps what it was given (no device)."""

    def __init__(self) -> None:
        self.pcm = bytearray()
        self.cleared = 0

    def write(self, pcm: bytes, rate: int) -> None:
        del rate
        self.pcm += pcm

    def clear(self) -> None:
        self.cleared += 1


def _tone(ms: int) -> bytes:
    n = RATE * ms // 1000
    return array("h", (int(8000 * math.sin(i / 8)) for i in range(n))).tobytes()


async def _next(player: PacedPlayer, state: str) -> ClockEvent:
    async with asyncio.timeout(5):
        async for event in player.events():
            if event.state == state:
                return event
    raise AssertionError("unreachable")


def test_fade_out_ends_in_silence() -> None:
    pcm = array("h", [10000] * 1000).tobytes()
    faded = array("h")
    faded.frombytes(fade_out(pcm, 200))
    assert faded[799] == 10000
    assert faded[800] < 10000
    assert abs(faded[-1]) < 100


async def test_stop_after_plays_on_then_fades_and_reports_flushed() -> None:
    sink = _Recorder()
    player = PacedPlayer(sink)
    await player.play(1, _tone(3000), RATE)
    await player.play(2, _tone(1000), RATE)  # queued behind it: dropped at once
    await _next(player, "started")
    await asyncio.sleep(0.5)
    loop = asyncio.get_running_loop()
    asked = loop.time()
    stream, end_ms = await player.stop_after(350, 80)
    assert stream == 1
    await player.play(1, _tone(1000), RATE)  # late audio of the stopping stream: not played
    dropped = await _next(player, "flushed")
    assert dropped.stream_id == 2
    stopped = await _next(player, "flushed")
    took_ms = (loop.time() - asked) * 1000
    assert stopped.stream_id == 1
    assert stopped.played_ms == end_ms
    assert 300 <= took_ms <= 450, f"stopped {took_ms:.0f} ms after the barge-in"
    assert sink.cleared == 0  # nothing cut short: the faded tail ends the sound
    tail = array("h")
    tail.frombytes(bytes(sink.pcm[-2 * RATE * 10 // 1000 :]))  # the last 10 ms
    assert max(abs(x) for x in tail) < 1500, "the last 10 ms are not faded"
    await player.close()


async def test_stop_after_a_stream_not_started_drops_it_at_once() -> None:
    player = PacedPlayer(_Recorder())
    assert await player.stop_after(350, 80) == (None, 0)
    await player.close()
