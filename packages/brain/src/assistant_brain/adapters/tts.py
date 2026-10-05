"""The TTS adapter: the speech server's streaming `POST {base_url}/v1/audio/speech`.

`stream(text, voice)` yields the reply audio as it is synthesized: s16le mono PCM at `rate`
(the server's `X-Sample-Rate`, 24 kHz), in whole samples. Closing the iteration closes the
HTTP stream, which stops the synthesis on the server. A server that cannot be reached or
answers an error raises `SpeechUnavailable`.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from assistant_brain.adapters.stt import SpeechUnavailable
from assistant_core.config import TtsConfig

CONNECT_TIMEOUT_S = 3.0
READ_TIMEOUT_S = 30.0
DEFAULT_RATE = 24000


@dataclass
class Synthesis:
    """Filled in while one sentence streams."""

    voice: str
    first_audio_ms: float | None = None
    """From the request to its first audio."""
    audio_ms: int = 0
    request_id: str | None = None


class TtsClient:
    def __init__(self, config: TtsConfig) -> None:
        self.config = config
        self.url = config.base_url.rstrip("/") + "/v1/audio/speech"
        self.rate = DEFAULT_RATE
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream(
        self, text: str, voice: str, synthesis: Synthesis | None = None
    ) -> AsyncIterator[bytes]:
        synthesis = synthesis if synthesis is not None else Synthesis(voice)
        started = time.monotonic()
        body = {"input": text, "voice": voice, "response_format": "pcm"}
        try:
            async with self._http.stream("POST", self.url, json=body) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode(errors="replace")[:300]
                    raise SpeechUnavailable(f"{self.url} answered {response.status_code}: {detail}")
                self.rate = int(response.headers.get("x-sample-rate", DEFAULT_RATE))
                synthesis.request_id = response.headers.get("x-request-id")
                carry, total = b"", 0
                async for data in response.aiter_bytes():
                    data = carry + data
                    whole = len(data) - len(data) % 2
                    carry = data[whole:]
                    if not whole:
                        continue
                    if synthesis.first_audio_ms is None:
                        synthesis.first_audio_ms = (time.monotonic() - started) * 1000
                    total += whole
                    synthesis.audio_ms = total * 1000 // (2 * self.rate)
                    yield data[:whole]
        except httpx.HTTPError as exc:
            raise SpeechUnavailable(f"{self.url}: {type(exc).__name__}: {exc}") from exc
