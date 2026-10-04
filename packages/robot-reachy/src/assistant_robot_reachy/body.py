"""ReachyBody: the Reachy Mini (Lite) through its daemon and the reachy-mini SDK.

The daemon (`python -m assistant_robot_reachy.daemon`) must run on this machine with its
media on: the SDK's LOCAL media backend reads the camera from the daemon's IPC socket and opens
the robot's USB audio card directly (macOS: CoreAudio `osxaudiosrc`/`osxaudiosink`; Linux:
PulseAudio), 16 kHz stereo float. The microphone is the XVF3800's processed channel 0, with
its hardware echo canceller (`aec: hw`, see `xvf3800`); at start the board gets Pollen's
conversation-app tuning (`xvf3800.STARTUP_TUNING`, read back into the AEC line as `tuning`).
Nothing here moves the robot unless the daemon answers with its backend ready; every move
goes through `MotionArbiter` (Pollen's MovementManager, `moves`).

Its `[body.reachy]` table: `connection` (the SDK's connection mode, `localhost_only`),
`expressions` (a TOML file whose `[moves]` table adds or replaces expression -> Pollen move
names over the built-in `EMOTION_MOVES`) and `idle_sleep_s` (120: how long the robot stays
awake at neutral after a turn before it goes to sleep with its motors off).

The robot wakes for the first turn (or expression) and stays awake during a conversation:
the movement manager holds its pose, plays the attention poses and expressions in order, and
the SDK's speech wobble moves the head while it speaks; `idle` is neutral, head up. After
`idle_sleep_s` without activity (attention or an expression) it goes to rest (IDLE-SLEEP).
Muting, stopping the agent, an error in a move and the watchdog still put it to rest at once.

While the edge's link to the brain is up (`link_changed`), the body touches the motor
watchdog's heartbeat file (`assistant_robot_reachy.watchdog`): the watchdog next to the daemon
puts the robot to rest with its motors off when the heartbeat stops (this agent died, froze or
stopped, or the link dropped). Stopping rests the robot, then removes the file.

If the daemon goes away, the body reports `BodyHealth(ok=False)` (the edge tells the brain
`error{code: body_unavailable}`), refuses motion, and reconnects on its own when the daemon is
back.

TODO(phase 3):
- A precise playback clock from the sink's real position instead of the paced estimate.
- Face tracking (`MovementManager.set_head_tracking`; off by default, as in Pollen's app).
"""

import asyncio
import contextlib
import json
import sys
import threading
import time
import tomllib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

from assistant_contracts.body import AudioFrame, BodyEvent, BodyHealth, PlaybackEvent
from assistant_contracts.capabilities import (
    AudioInCaps,
    BodyCapabilities,
    CameraCaps,
    Capabilities,
    MotionCaps,
)
from assistant_contracts.common import Aec, AttentionState, LookTarget
from assistant_core.playback import PacedPlayer
from assistant_robot_reachy import watchdog
from assistant_robot_reachy.arbiter import (
    DAEMON_URL,
    MotionArbiter,
    RobotUnavailable,
    connect_mini,
    robot_ready,
)
from assistant_robot_reachy.xvf3800 import aec_status, apply_startup_tuning

DEVICE_RATE = 16000
FRAME_SAMPLES = 320
HEALTH_EVERY_S = 1.0
FRESH_BEAT_S = 1.5
"""The movement manager commands the robot only while the link heartbeat is this fresh (the
watchdog rests the robot at `watchdog.STALE_S`)."""


def emit(tag: str, payload: Any = None, **fields: object) -> None:
    parts = [tag, *(f"{k}={v}" for k, v in fields.items())]
    if payload is not None:
        parts.append(json.dumps(payload, default=str))
    sys.stdout.write(" ".join(parts) + "\n")
    sys.stdout.flush()


class _Speaker:
    """`PcmSink` on the robot speaker: s16 mono at any rate -> float32 at 16 kHz."""

    def __init__(self, body: "ReachyBody") -> None:
        self.body = body

    def write(self, pcm: bytes, rate: int) -> None:
        mini = self.body.mini
        if mini is None:
            return
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if rate != DEVICE_RATE and len(x):
            n = len(x) * DEVICE_RATE // rate
            x = np.interp(np.arange(n) * (rate / DEVICE_RATE), np.arange(len(x)), x)
            x = x.astype(np.float32)
        mini.media.push_audio_sample(x)

    def clear(self) -> None:
        mini = self.body.mini
        if mini is not None and mini.media.audio is not None:
            mini.media.audio.clear_player()


class ReachyAudio:
    aec: Aec = "hw"

    def __init__(self, body: "ReachyBody") -> None:
        self.body = body
        self.player = PacedPlayer(_Speaker(body))
        self._thread: threading.Thread | None = None
        self._running = False

    async def capture(self) -> AsyncIterator[AudioFrame]:
        loop = asyncio.get_running_loop()
        frames: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=100)

        def put(frame: AudioFrame) -> None:
            if frames.full():
                frames.get_nowait()
            frames.put_nowait(frame)

        def pump() -> None:
            pending = np.zeros(0, dtype=np.int16)
            while self._running:
                mini = self.body.mini
                sample = None
                if mini is not None:
                    with contextlib.suppress(Exception):
                        sample = mini.media.get_audio_sample()
                if sample is None:
                    time.sleep(0.005)
                    continue
                mono = sample[:, 0] if sample.ndim == 2 else sample
                pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
                pending = np.concatenate([pending, pcm])
                while len(pending) >= FRAME_SAMPLES:
                    chunk, pending = pending[:FRAME_SAMPLES], pending[FRAME_SAMPLES:]
                    frame = AudioFrame(chunk.astype("<i2").tobytes(), time.monotonic_ns() // 1000)
                    loop.call_soon_threadsafe(put, frame)

        self._running = True
        self._thread = threading.Thread(target=pump, name="reachy-mic", daemon=True)
        self._thread.start()
        try:
            while True:
                yield await frames.get()
        finally:
            self._running = False

    async def play(self, stream_id: int, pcm: bytes, rate: int) -> None:
        await self.player.play(stream_id, pcm, rate)

    async def flush(self, stream_id: int | None = None) -> int:
        return await self.player.flush(stream_id)

    async def stop_after(self, ms: int, fade_ms: int) -> tuple[int | None, int]:
        return await self.player.stop_after(ms, fade_ms)

    async def playback_events(self) -> AsyncIterator[PlaybackEvent]:
        async for event in self.player.events():
            yield PlaybackEvent(event.stream_id, event.played_ms, event.state)

    async def close(self) -> None:
        self._running = False
        await self.player.close()


class ReachyMotion:
    """Attention states become small, slow head poses (`MotionArbiter.attend`), played in
    order by a worker so the link is never blocked by a move; each prints a MOTION line with
    the pose and when it was reached (the robot machine's monotonic clock). After `idle` the
    robot stays awake; `idle_sleep_s` later, with no new attention or expression meanwhile,
    it goes to rest (`rest`: asleep, motors off, if it was at rest before the turn)."""

    def __init__(self, body: "ReachyBody", idle_sleep_s: float = 120.0) -> None:
        self.body = body
        self.idle_sleep_s = idle_sleep_s
        self._states: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._idle_timer: asyncio.Task[None] | None = None
        self._enqueue_lock = asyncio.Lock()
        self._expressing = 0
        self._attention: str | None = None
        """The brain's latest attention state: the idle timeout runs only after `idle`."""

    def _arbiter(self) -> MotionArbiter:
        arbiter = self.body.arbiter
        if arbiter is None or not self.body.healthy:
            raise RobotUnavailable("the robot is not connected")
        return arbiter

    async def attention(self, state: AttentionState, assistant: str | None) -> None:
        del assistant
        if self.body.arbiter is None or not self.body.healthy:
            emit("MOTION-ERROR", {"detail": "the robot is not connected"}, attention=state)
            return
        self._cancel_idle_timer()
        self._attention = state
        self._queue(state)

    def _queue(self, state: str) -> None:
        self._states.put_nowait(state)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._attend_loop(), name="reachy-attention")

    def _cancel_idle_timer(self) -> bool:
        """Stop the idle timeout; whether one was running."""
        timer, self._idle_timer = self._idle_timer, None
        if timer is None or timer.done():
            return False
        timer.cancel()
        return True

    def _idle_now(self) -> bool:
        """No turn running (the last attention state was `idle`, or none came), no state
        waiting and no expression pending: the idle timeout may run."""
        idle = self._attention in (None, "idle")
        return idle and self._states.empty() and not self._expressing

    def _start_idle_timer(self) -> None:
        self._cancel_idle_timer()
        self._idle_timer = asyncio.create_task(self._idle_sleep(), name="reachy-idle-sleep")

    async def _idle_sleep(self) -> None:
        started = time.monotonic()
        await asyncio.sleep(self.idle_sleep_s)
        self._idle_timer = None
        emit("IDLE-SLEEP", after_s=f"{self.idle_sleep_s:g}",
             waited_s=f"{time.monotonic() - started:.2f}")  # fmt: skip
        self._queue("rest")

    async def _attend_loop(self) -> None:
        while True:
            state = await self._states.get()
            try:
                arbiter = self._arbiter()
                match state:
                    case "listening":
                        arbiter.set_listening(True)
                    case "speaking":
                        arbiter.set_speaking(True)
                    case _:
                        arbiter.breathe()
                done = await asyncio.to_thread(arbiter.attend, state)
            except Exception as exc:
                emit("MOTION-ERROR", {"detail": f"{type(exc).__name__}: {exc}"}, attention=state)
                continue
            emit("MOTION", done or {}, attention=state, moved=str(done is not None).lower())
            if self._idle_now() and done is not None:
                self._start_idle_timer()  # awake at neutral until the idle timeout

    async def rest(self) -> None:
        """Stop attending: drop pending states and put an attending robot back to rest."""
        self._cancel_idle_timer()
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        while not self._states.empty():
            self._states.get_nowait()
        arbiter = self.body.arbiter
        if arbiter is not None:
            # Unconditionally: `attend` waits (its lock) for a move still running in its thread
            # after the worker was cancelled, then rests the robot if that move woke it.
            try:
                done = await asyncio.to_thread(arbiter.attend, "rest")
            except Exception as exc:
                emit("MOTION-ERROR", {"detail": f"{type(exc).__name__}: {exc}"}, rest="true")
            else:
                emit("MOTION", done or {}, rest="true", moved=str(done is not None).lower())

    async def express(self, name: str, intensity: float = 1.0) -> bool:
        """Queue the expression's move on the movement manager in the order the requests came
        (several may be pending: they play one after the other) and wait for its end."""
        arbiter = self._arbiter()
        started = time.monotonic()
        self._cancel_idle_timer()  # activity: the idle timeout starts again after it
        self._expressing += 1
        record: dict[str, Any] = {}
        ok = False
        try:
            async with self._enqueue_lock:  # FIFO: the order the requests came
                queued = await asyncio.to_thread(arbiter.enqueue_emotion, name, intensity)
            if queued is not None:
                record = await asyncio.to_thread(arbiter.finish_emotion, queued)
                ok = True
        finally:
            self._expressing -= 1
            if arbiter.attending and self._idle_now():
                self._start_idle_timer()
        emit("MOTION", record, express=name, intensity=intensity,
             ok=str(ok).lower(), took_s=f"{time.monotonic() - started:.2f}")  # fmt: skip
        return ok

    async def look_at(self, target: LookTarget) -> bool:
        """TODO(phase 3): look_at_world / look_at_image and face tracking."""
        del target
        return False


class _JpegEncoder:
    """BGR frame -> JPEG with its longer side at most `max_side` (GStreamer jpegenc)."""

    def encode(self, frame: Any, max_side: int) -> bytes:
        from gi.repository import Gst  # pyright: ignore[reportAttributeAccessIssue]

        height, width = frame.shape[:2]
        scale = min(1.0, max_side / max(width, height))
        out_w, out_h = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
        pipeline = Gst.parse_launch(
            f"appsrc name=src caps=video/x-raw,format=BGR,width={width},height={height},"
            "framerate=0/1 ! videoconvert ! videoscale ! "
            f"video/x-raw,width={out_w},height={out_h} ! jpegenc quality=85 ! "
            "appsink name=sink sync=false"
        )
        src, sink = pipeline.get_by_name("src"), pipeline.get_by_name("sink")
        pipeline.set_state(Gst.State.PLAYING)
        try:
            src.emit("push-buffer", Gst.Buffer.new_wrapped(frame.tobytes()))
            src.emit("end-of-stream")
            sample = sink.emit("try-pull-sample", 5 * Gst.SECOND)
            if sample is None:
                raise RuntimeError("the JPEG encoder produced no image")
            buffer = sample.get_buffer()
            ok, info = buffer.map(Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError("could not read the encoded JPEG")
            try:
                return bytes(info.data)
            finally:
                buffer.unmap(info)
        finally:
            pipeline.set_state(Gst.State.NULL)


class ReachyCamera:
    def __init__(self, body: "ReachyBody") -> None:
        self.body = body
        self.encoder = _JpegEncoder()

    def _grab(self, max_side: int) -> bytes:
        mini = self.body.mini
        if mini is None or not self.body.healthy:
            raise RobotUnavailable("the robot is not connected")
        frame = None
        deadline = time.monotonic() + 3
        while frame is None and time.monotonic() < deadline:
            frame = mini.media.get_frame()
        if frame is None:
            raise RuntimeError("the camera returned no frame")
        return self.encoder.encode(frame, max_side)

    async def snapshot(self, max_side: int = 1024) -> bytes:
        return await asyncio.to_thread(self._grab, max_side)

    def size(self) -> tuple[int, int] | None:
        mini = self.body.mini
        camera = mini.media.camera if mini is not None else None
        return tuple(camera.resolution) if camera is not None else None


def expression_moves(path: str | None) -> dict[str, str]:
    """The `[moves]` table of the `expressions` file (expression -> Pollen move name)."""
    if path is None:
        return {}
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(f"[body.reachy] expressions: no file {path}")
    moves = tomllib.loads(file.read_text()).get("moves", {})
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in moves.items()):
        raise ValueError(f"{path}: [moves] maps expression names to move names (strings)")
    return dict(moves)


class ReachyBody:
    kind = "reachy"

    def __init__(
        self,
        daemon_url: str = DAEMON_URL,
        connection: str = "localhost_only",
        expressions: str | None = None,
        idle_sleep_s: float = 120.0,
    ) -> None:
        if not 0 < idle_sleep_s <= 3600:
            raise ValueError(f"[body.reachy] idle_sleep_s {idle_sleep_s}: 0 < seconds <= 3600")
        self.daemon_url = daemon_url
        self.connection = connection
        self.moves = expression_moves(expressions)
        self.mini: Any = None
        self.arbiter: MotionArbiter | None = None
        self.healthy = False
        self.audio = ReachyAudio(self)
        self.motion = ReachyMotion(self, float(idle_sleep_s))
        self.camera = ReachyCamera(self)
        self._health: asyncio.Queue[BodyHealth] = asyncio.Queue()
        self._watch: asyncio.Task[None] | None = None
        self._beat: asyncio.Task[None] | None = None
        self.link_up = False
        self._last_beat = 0.0

    # ------------------------------------------------------------ lifecycle

    def _connect(self) -> None:
        mini = connect_mini(
            connection_mode=self.connection,
            media_backend="local",
            automatic_body_yaw=False,
            log_level="WARNING",
        )
        mini.media.start_recording()
        mini.media.start_playing()
        self.mini = mini
        self.arbiter = MotionArbiter(mini, daemon_url=self.daemon_url, alive=self._alive)
        self.arbiter.emotions.update(self.moves)

    def _alive(self) -> bool:
        """The link is up and its heartbeat fresh (read by the movement manager's thread)."""
        return self.link_up and time.monotonic() - self._last_beat <= FRESH_BEAT_S

    def _beat_once(self) -> None:
        try:
            watchdog.touch()
        except OSError as exc:
            emit("HEARTBEAT-ERROR", {"detail": f"{type(exc).__name__}: {exc}"})
        else:
            self._last_beat = time.monotonic()

    def _disconnect(self) -> None:
        if self.arbiter is not None:
            self.arbiter.manager.stop()
        mini, self.mini, self.arbiter = self.mini, None, None
        if mini is None:
            return
        with contextlib.suppress(Exception):
            mini.media.close()
        with contextlib.suppress(Exception):
            mini.client.disconnect()

    async def start(self) -> BodyCapabilities:
        """Connect to the running daemon; fails (nothing moves) if the robot is not there."""
        ok, detail = await asyncio.to_thread(robot_ready, self.daemon_url)
        if not ok:
            raise RobotUnavailable(f"no robot: {detail}")
        await asyncio.to_thread(self._connect)
        self.healthy = True
        tuning = await asyncio.to_thread(apply_startup_tuning)
        aec = await asyncio.to_thread(aec_status)
        emit("AEC", {**aec, "tuning": tuning})
        size = self.camera.size()
        self._watch = asyncio.create_task(self._watch_daemon(), name="reachy-health")
        self._beat = asyncio.create_task(self._heartbeat(), name="reachy-heartbeat")
        return Capabilities(
            audio_in=AudioInCaps(rate=16000, aec="hw"),
            motion=MotionCaps(
                expressions=sorted({*self.arbiter.emotions, "random"}) if self.arbiter else [],
                attention=True,
            ),
            camera=CameraCaps(w=size[0], h=size[1]) if size else None,
            doa=False,
        )

    def link_changed(self, up: bool) -> None:
        """The edge's link to the brain came up or went down: the heartbeat goes on or stops
        (a link back within the watchdog's `STALE_S` leaves the robot as it is)."""
        self.link_up = up
        if up:  # at once: a link that drops right away still arms the watchdog
            self._beat_once()

    async def _heartbeat(self) -> None:
        while True:
            if self.link_up:
                self._beat_once()
            await asyncio.sleep(watchdog.HEARTBEAT_EVERY_S)

    async def stop(self) -> None:
        if self._watch is not None:
            self._watch.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch
        await self.motion.rest()  # the heartbeat goes on meanwhile: one rest, this one
        if self._beat is not None:
            self._beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._beat
        with contextlib.suppress(OSError):
            watchdog.clear()
        await self.audio.close()
        await asyncio.to_thread(self._disconnect)

    async def events(self) -> AsyncIterator[BodyEvent]:
        """TODO(phase 3): DOA from the XVF3800 (`get_DoA`) and IMU taps (wireless model)."""
        await asyncio.Event().wait()
        return
        yield  # pragma: no cover

    async def health(self) -> AsyncIterator[BodyHealth]:
        while True:
            yield await self._health.get()

    async def _watch_daemon(self) -> None:
        while True:
            await asyncio.sleep(HEALTH_EVERY_S)
            ok, detail = await asyncio.to_thread(robot_ready, self.daemon_url)
            if self.healthy and not ok:
                self.healthy = False
                await self.audio.flush(None)
                self._health.put_nowait(BodyHealth(False, detail))
                await asyncio.to_thread(self._disconnect)
            elif not self.healthy and ok:
                try:
                    await asyncio.to_thread(self._connect)
                except Exception as exc:
                    emit("BODY-RETRY", {"detail": f"{type(exc).__name__}: {exc}"})
                    continue
                self.healthy = True
                self._health.put_nowait(BodyHealth(True, "robot reconnected"))
