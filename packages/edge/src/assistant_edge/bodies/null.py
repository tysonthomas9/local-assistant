"""NullBody: no microphone, no speaker, no motion, no camera.

For running an edge agent where there is no audio hardware (a headless box, a CI runner).
Speech it is sent still runs on the real playback clock (`PacedPlayer`), into no device, so
the brain sees the same `playback` events as from a real speaker.
"""

import asyncio
from collections.abc import AsyncIterator

from assistant_contracts.body import AudioFrame, BodyEvent, PlaybackEvent
from assistant_contracts.capabilities import BodyCapabilities, Capabilities
from assistant_contracts.common import Aec
from assistant_core.playback import PacedPlayer


class _NoDevice:
    """A `PcmSink` that plays into nothing (at the real-time pace of the player)."""

    def write(self, pcm: bytes, rate: int) -> None:
        del pcm, rate

    def clear(self) -> None:
        pass


class NullAudio:
    """No capture; playback on the real clock into no device."""

    aec: Aec = "none"

    def __init__(self) -> None:
        self.player = PacedPlayer(_NoDevice())

    async def capture(self) -> AsyncIterator[AudioFrame]:
        await asyncio.Event().wait()  # no microphone: never yields
        return
        yield  # pragma: no cover

    async def play(self, stream_id: int, pcm: bytes, rate: int) -> None:
        await self.player.play(stream_id, pcm, rate)

    async def flush(self, stream_id: int | None = None) -> int:
        return await self.player.flush(stream_id)

    async def playback_events(self) -> AsyncIterator[PlaybackEvent]:
        async for event in self.player.events():
            yield PlaybackEvent(event.stream_id, event.played_ms, event.state)

    @property
    def playing(self) -> int | None:
        return self.player.active

    async def close(self) -> None:
        await self.player.close()


class NullBody:
    kind = "null"

    def __init__(self) -> None:
        self.audio = NullAudio()
        self.motion = None
        self.camera = None

    async def start(self) -> BodyCapabilities:
        return Capabilities()

    async def stop(self) -> None:
        await self.audio.close()

    async def events(self) -> AsyncIterator[BodyEvent]:
        await asyncio.Event().wait()
        return
        yield  # pragma: no cover
