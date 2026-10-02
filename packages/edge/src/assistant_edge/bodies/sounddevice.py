"""SoundDeviceBody: the default microphone and speaker of this machine (PortAudio).

For an edge on a laptop or desktop. No motion and no camera. There is no echo cancellation
(`aec: none`), so use headphones or push-to-talk. `sounddevice` is imported on `start()`, so the
package imports without PortAudio installed.
"""

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from assistant_contracts.body import AudioFrame, BodyEvent, PlaybackEvent
from assistant_contracts.capabilities import BodyCapabilities, Capabilities
from assistant_contracts.common import Aec
from assistant_core.playback import PacedPlayer

MIC_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms


class _Speaker:
    """A `PcmSink` on the default output device; reopened when the stream rate changes."""

    def __init__(self, sd: Any) -> None:
        self.sd = sd
        self.rate = 0
        self.stream: Any = None
        self.buffer = bytearray()
        self.lock = threading.Lock()

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        del time_info, status
        need = frames * 2
        with self.lock:
            chunk = bytes(self.buffer[:need])
            del self.buffer[:need]
        outdata[: len(chunk)] = chunk
        outdata[len(chunk) :] = b"\x00" * (need - len(chunk))

    def write(self, pcm: bytes, rate: int) -> None:
        if rate != self.rate:
            self.close()
            self.stream = self.sd.RawOutputStream(
                samplerate=rate, channels=1, dtype="int16", callback=self._callback
            )
            self.stream.start()
            self.rate = rate
        with self.lock:
            self.buffer += pcm

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class SoundDeviceAudio:
    aec: Aec = "none"

    def __init__(self, sd: Any) -> None:
        self.sd = sd
        self.speaker = _Speaker(sd)
        self.player = PacedPlayer(self.speaker)
        self._mic: Any = None

    async def capture(self) -> AsyncIterator[AudioFrame]:
        loop = asyncio.get_running_loop()
        frames: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=100)

        def callback(indata: Any, count: int, time_info: Any, status: Any) -> None:
            del count, time_info, status
            frame = AudioFrame(bytes(indata), time.monotonic_ns() // 1000)
            loop.call_soon_threadsafe(_put_latest, frames, frame)

        self._mic = self.sd.RawInputStream(
            samplerate=MIC_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=callback,
        )
        self._mic.start()
        try:
            while True:
                yield await frames.get()
        finally:
            self._mic.close()
            self._mic = None

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
        self.speaker.close()


def _put_latest(queue: asyncio.Queue[AudioFrame], frame: AudioFrame) -> None:
    """Queue a frame, dropping the oldest when the consumer is behind (live audio)."""
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(frame)


class SoundDeviceBody:
    kind = "sounddevice"

    def __init__(self) -> None:
        import sounddevice

        self.audio = SoundDeviceAudio(sounddevice)
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
