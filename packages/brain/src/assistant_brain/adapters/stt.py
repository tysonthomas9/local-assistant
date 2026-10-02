"""The STT adapter: the speech server's `POST {base_url}/v1/audio/transcriptions`.

The user's audio (s16le mono, 16 kHz from the edge's mic windows) goes up as raw PCM
(`audio/pcm`, `?rate=`); the transcript comes back as JSON. A server that cannot be reached
or answers an error raises `SpeechUnavailable`.
"""

import time
from dataclasses import dataclass

import httpx

from assistant_core.config import SttConfig

CONNECT_TIMEOUT_S = 3.0
READ_TIMEOUT_S = 30.0


class SpeechUnavailable(Exception):
    """The speech server (STT or TTS) could not be reached, or it answered with an error."""


@dataclass(frozen=True)
class Transcript:
    text: str
    ms: float
    """Round trip of the request (the server's own time is `server_ms`)."""
    server_ms: float | None
    audio_ms: int
    request_id: str | None


class SttClient:
    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self.url = config.base_url.rstrip("/") + "/v1/audio/transcriptions"
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def transcribe(self, pcm: bytes, rate: int) -> Transcript:
        started = time.monotonic()
        try:
            response = await self._http.post(
                self.url,
                params={"rate": rate},
                content=pcm,
                headers={"Content-Type": "audio/pcm"},
            )
        except httpx.HTTPError as exc:
            raise SpeechUnavailable(f"{self.url}: {type(exc).__name__}: {exc}") from exc
        if response.status_code >= 400:
            detail = response.text[:300]
            raise SpeechUnavailable(f"{self.url} answered {response.status_code}: {detail}")
        try:
            body = response.json()
        except ValueError as exc:
            raise SpeechUnavailable(f"{self.url}: bad JSON: {exc}") from exc
        return Transcript(
            text=str(body.get("text") or "").strip(),
            ms=(time.monotonic() - started) * 1000,
            server_ms=body.get("ms"),
            audio_ms=len(pcm) * 1000 // (2 * rate),
            request_id=body.get("id"),
        )
