"""In-process bridge between our tools and the conversation app's live LocalStream.

Tools only receive `deps` (robot, movement manager, instance path), but muting, wake-word gating,
book reading and persona switching need the live conversation stream. run_app.py calls
`install()` before the app starts; it wraps three LocalStream methods (upstream code unchanged):

  __init__            -> remember the stream (stream())
  _dispatch_activity  -> publish activity reasons to subscribers (user_speech_started,
                         response_created, assistant_transcript_done, ...)
  clear_audio_queue   -> publish "interrupted" (barge-in, conversation.say/interrupt)

It also wraps the SDK's MediaManager.push_audio_sample to keep a playback clock:
`audio_seconds_left()` estimates how much already-sent speech is still waiting to be played
(pushed audio is played in real time; a barge-in flush resets it). The book reader uses it to
wait until a passage has finished playing before its pause and the next passage; the wake gate
uses it to start the follow-up window when a reply has finished playing.

This module is imported normally (not re-executed on profile reloads like tool files), so its
state survives persona switches. Subscribers are called on the app's event loop and must be quick.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Coroutine

logger = logging.getLogger("reachy_bridge")

_stream: Any = None
_subscribers: list[Callable[[str], None]] = []
_sub_lock = threading.Lock()
_installed = False
_play_lock = threading.Lock()
_play_end = 0.0          # monotonic time when pushed audio will have finished playing
OUTPUT_RATE = 16000      # the robot's audio board; refreshed from the media manager when available


def install() -> None:
    """Patch LocalStream once. Safe to call twice."""
    global _installed
    if _installed:
        return
    from reachy_mini_conversation_app import console

    cls = console.LocalStream
    orig_init, orig_dispatch, orig_clear = cls.__init__, cls._dispatch_activity, cls.clear_audio_queue

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        global _stream
        orig_init(self, *args, **kwargs)
        _stream = self

    def _dispatch_activity(self: Any, reason: str) -> None:
        orig_dispatch(self, reason)
        publish(reason)

    def clear_audio_queue(self: Any) -> None:
        global _play_end
        orig_clear(self)
        with _play_lock:
            _play_end = time.monotonic()
        publish("interrupted")

    cls.__init__, cls._dispatch_activity, cls.clear_audio_queue = __init__, _dispatch_activity, clear_audio_queue

    _install_slot_wait()

    try:
        from reachy_mini.media import media_manager

        orig_push = media_manager.MediaManager.push_audio_sample

        def push_audio_sample(self: Any, data: Any) -> None:
            orig_push(self, data)
            try:  # bookkeeping only: must never break the app's play loop
                rate = self.get_output_audio_samplerate() or OUTPUT_RATE
                shape = getattr(data, "shape", None) or (len(data),)
                # samples = the longer axis (upstream accepts (n,), (n, ch) and (ch, n))
                samples = shape[0] if len(shape) == 1 else max(shape[0], shape[1])
                add_played_audio(samples / float(rate))
            except Exception:
                logger.debug("playback clock: couldn't account for %r", type(data), exc_info=True)

        media_manager.MediaManager.push_audio_sample = push_audio_sample
    except Exception:
        logger.warning("playback clock unavailable", exc_info=True)
    _installed = True


SLOT_WAIT_S = 3.0


def _pool_url() -> str | None:
    """http://host:port/v1/pool of a local speech server, from the app's realtime URL."""
    import os
    from urllib.parse import urlparse
    try:
        from reachy_mini_conversation_app.config import HF_REALTIME_WS_URL_ENV
        url = os.environ.get(HF_REALTIME_WS_URL_ENV, "")
    except Exception:
        return None
    u = urlparse(url)
    if u.hostname not in ("127.0.0.1", "localhost", "::1") or not u.port:
        return None
    return f"http://{u.hostname}:{u.port}/v1/pool"


def _install_slot_wait() -> None:
    """Before (re)connecting, wait until the speech server has freed a session slot.

    After a session closes the server needs ~50 ms (longer if a reply was being generated) to release its
    pipeline; a reconnect that arrives earlier is rejected and the app only retries after 1-1.5 s.
    """
    try:
        from reachy_mini_conversation_app import huggingface_realtime as hr
    except Exception:
        return
    orig_start_up = hr.HuggingFaceRealtimeHandler.start_up

    async def start_up(self: Any) -> None:
        url = _pool_url()
        if url:
            import httpx
            deadline = time.monotonic() + SLOT_WAIT_S
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    while time.monotonic() < deadline:
                        pool = (await client.get(url)).json()
                        if pool.get("in_use", 0) < pool.get("size", 1):
                            break
                        await asyncio.sleep(0.05)
            except Exception as e:
                logger.debug("slot wait skipped: %r", e)
        return await orig_start_up(self)
    hr.HuggingFaceRealtimeHandler.start_up = start_up


def add_played_audio(seconds: float) -> None:
    """Account for `seconds` of audio handed to the speaker (called by the push_audio_sample wrapper)."""
    global _play_end
    with _play_lock:
        _play_end = max(_play_end, time.monotonic()) + seconds


def audio_seconds_left() -> float:
    """Seconds of already-pushed speech still to be played (0 when the robot is quiet)."""
    with _play_lock:
        return max(0.0, _play_end - time.monotonic())


def stream() -> Any:
    """The live LocalStream, or None if the app hasn't started (e.g. in tests)."""
    return _stream


def publish(reason: str) -> None:
    with _sub_lock:
        subs = list(_subscribers)
    for cb in subs:
        try:
            cb(reason)
        except Exception:
            logger.exception("bridge subscriber failed on %r", reason)


def subscribe(callback: Callable[[str], None]) -> Callable[[], None]:
    """Call `callback(reason)` on every activity event; returns an unsubscribe function."""
    with _sub_lock:
        _subscribers.append(callback)

    def unsubscribe() -> None:
        with _sub_lock:
            if callback in _subscribers:
                _subscribers.remove(callback)
    return unsubscribe


def wait_for(reasons: set[str], timeout: float) -> str | None:
    """Block (from a background thread) until one of `reasons` is published; returns it or None."""
    hit: list[str] = []
    done = threading.Event()

    def cb(reason: str) -> None:
        if reason in reasons and not hit:
            hit.append(reason)
            done.set()
    unsubscribe = subscribe(cb)
    try:
        done.wait(timeout)
        return hit[0] if hit else None
    finally:
        unsubscribe()


def run_in_app_loop(factory: Callable[[], Coroutine[Any, Any, Any]], timeout: float = 10) -> Any:
    """Run a coroutine on the app's event loop from another thread and return its result."""
    s = _stream
    loop = getattr(s, "_asyncio_loop", None)
    if loop is None:
        raise RuntimeError("conversation app is not running")
    return asyncio.run_coroutine_threadsafe(factory(), loop).result(timeout)


def mic_muted() -> bool | None:
    return None if _stream is None else bool(_stream._mic_muted)


def set_mic_muted(muted: bool) -> bool:
    """Mute/unmute the microphone (frames are dropped before being sent). False if no stream."""
    if _stream is None:
        return False
    _stream._mic_muted = bool(muted)
    publish("mic_muted" if muted else "mic_unmuted")
    return True
