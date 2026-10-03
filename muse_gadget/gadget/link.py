"""The gadget's connection to Muse, wrapped for the robot.

``RobotLinkSession`` adds two things to the SDK's ``LinkSession``: the
``POST /chat/subscribe`` NDJSON event stream on the same encrypted session,
which carries Muse's replies (as in the ESP32 firmware,
``esp32/components/muse/muse_chat_session.cpp``), and answers to the
``client.invoke`` events on that stream (``gadget/client_invoke.py``).

``RobotService`` is the SDK's ``Service`` with the restricted command list,
the neutral display name and ``RobotLinkSession``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from musegadget import __version__ as sdk_version
from musegadget import link_client
from musegadget.link_client import DeviceDescription, LinkSession, Outcome
from musegadget.noise import Header
from musegadget.service import DEFAULT_NOISE_HOST, Service

from gadget import client_invoke, restrict

log = logging.getLogger(__name__)


SUBSCRIBE_PATH = "/chat/subscribe"
MAX_EVENT_LINE = 256 * 1024


class Subscription:
    """One ``POST /chat/subscribe`` stream: NDJSON chat events, one per line."""

    def __init__(self, session: "RobotLinkSession", stream_id: int) -> None:
        self._session = session
        self.stream_id = stream_id
        self.opened: asyncio.Future = asyncio.get_running_loop().create_future()
        self._events: asyncio.Queue = asyncio.Queue()
        self._buf = bytearray()
        self._ended = False

    # Called by the session's read loop for every frame on this stream.
    def on_frame(self, frame) -> None:
        if self._ended:
            return
        if frame.kind == "reset":
            self._end(ConnectionError("subscription reset"))
            return
        if frame.kind == "response":
            status = frame.value.status
            if not self.opened.done():
                self.opened.set_result(status)
            if status >= 400:
                self._end(None)
                return
            data, ended = frame.value.body, frame.value.end_body
        else:
            data, ended = frame.value.data, frame.value.end_body
        self._feed(data)
        if ended:
            self._end(None)

    def _feed(self, data: bytes) -> None:
        self._buf += data
        while True:
            cut = self._buf.find(b"\n")
            if cut < 0:
                if len(self._buf) > MAX_EVENT_LINE:
                    self._buf.clear()   # an oversized line: drop it, as the firmware does
                return
            line = bytes(self._buf[:cut]).strip()
            del self._buf[:cut + 1]
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                if event.get("event") == client_invoke.EVENT:
                    self._session.on_client_invoke(event)
                self._events.put_nowait(event)

    def _end(self, error: Exception | None) -> None:
        self._ended = True
        if not self.opened.done():
            if error is not None:
                self.opened.set_exception(error)
            else:
                self.opened.set_result(0)
        self._events.put_nowait(None)

    async def next(self, timeout: float) -> dict | None:
        """The next event, or None after ``timeout`` seconds. Raises ConnectionError once it ends."""
        try:
            event = await asyncio.wait_for(self._events.get(), max(timeout, 0))
        except asyncio.TimeoutError:
            return None
        if event is None:
            self._events.put_nowait(None)
            raise ConnectionError("subscription ended")
        return event

    async def close(self) -> None:
        self._session._requests.pop(self.stream_id, None)
        if not self._ended:
            self._ended = True
            try:
                await self._session._send_frames(self._session._transport.encrypt_reset(self.stream_id))
            except Exception:
                pass


class RobotLinkSession(LinkSession):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seen_invokes = client_invoke.SeenIds()
        self._client_invoke = client_invoke.enabled()

    async def _invoke(self, message: dict) -> None:
        # Upstream link.invoke handling. Ids are shared with client.invoke, so the
        # same request never runs twice, whichever path delivered it first.
        invoke_id = message.get("id")
        if isinstance(invoke_id, str) and not self._seen_invokes.add(invoke_id):
            log.info("link.invoke duplicate ignored id=%s", client_invoke.safe_name(invoke_id))
            return
        await super()._invoke(message)

    def on_client_invoke(self, event: dict) -> None:
        """Answer a ``client.invoke`` from the subscription stream (see ``gadget/client_invoke.py``)."""
        if not self._client_invoke:
            return
        invoke = client_invoke.parse(event)
        if invoke is None:
            log.info("client.invoke ignored: malformed")
            return
        if not self._seen_invokes.add(invoke.invoke_id):
            log.info("client.invoke duplicate ignored id=%s", invoke.invoke_id)
            return
        log.info("client.invoke command=%s id=%s", invoke.command, invoke.invoke_id)
        task = asyncio.ensure_future(self._answer_client_invoke(invoke))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _answer_client_invoke(self, invoke: client_invoke.Invoke) -> None:
        result = client_invoke.refusal(invoke)
        if result is None:
            async with self._invokes:
                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, self._run_command, invoke.command, invoke.params, invoke.timeout_ms,
                    )
                except Exception as exc:
                    log.warning("client.invoke command=%s failed: %s", invoke.command, type(exc).__name__)
                    result = client_invoke.upstream_executor.error("command failed")
        await self.send({"method": "link.result", "id": invoke.invoke_id, **result})
        log.info("client.invoke answered command=%s id=%s ok=%s",
                 invoke.command, invoke.invoke_id, bool(result.get("ok")))

    async def subscribe(self, session_id: str | None = None) -> Subscription:
        """Open ``POST /chat/subscribe``; returns once the VM answered with a status."""
        body = json.dumps({"session_id": session_id} if session_id else {}).encode()
        headers = [
            Header("Content-Type", "application/json"),
            Header("accept", "application/x-ndjson"),
            Header("x-request-id", str(uuid.uuid4())),
            Header("x-app-id", link_client.APP_ID),
        ]
        encrypted = self._transport.encrypt_http_request("POST", SUBSCRIBE_PATH, body, headers=headers)
        sub = Subscription(self, encrypted.stream_id)
        self._requests[encrypted.stream_id] = sub
        try:
            await self._send_frames(encrypted.frames)
            status = await asyncio.wait_for(asyncio.shield(sub.opened), link_client.REQUEST_TIMEOUT_S)
        except BaseException:
            await sub.close()
            raise
        if not 200 <= status < 300:
            await sub.close()
            raise SubscribeRefused(status)
        return sub


class SubscribeRefused(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


@dataclass
class RobotService(Service):
    display_name: str = field(default_factory=restrict.display_name)

    @property
    def link(self) -> RobotLinkSession | None:
        """The registered session, or None while the link is down."""
        session = self._current
        if session is None or session.registered_at is None:
            return None
        return session

    async def _session(self, vm: dict, pairing: dict) -> tuple[Outcome, float]:
        device = DeviceDescription(
            node_id=self.identity.node_id,
            display_name=self.display_name,
            version=sdk_version,
            commands=restrict.command_specs(),
        )
        session = RobotLinkSession(
            noise_host=pairing.get("noise_host") or DEFAULT_NOISE_HOST,
            vm_id=vm["vm_id"] or vm["vm_name"],
            vm_auth_token=vm["vm_auth_token"],
            device=device,
            run_command=self.executor.run,
        )
        log.info("connecting to the Muse")
        started = time.monotonic()
        self._current = session
        try:
            outcome = await session.run(self._stop)
        except Exception as exc:
            log.warning("session failed: %s", type(exc).__name__)
            outcome = Outcome.CLOSED
        finally:
            self._current = None
        lasted = time.monotonic() - (session.registered_at or time.monotonic())
        log.info("session ended: %s after %.0fs", outcome.value, time.monotonic() - started)
        return outcome, lasted
