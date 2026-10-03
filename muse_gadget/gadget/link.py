"""The gadget's connection to Muse, wrapped for the robot.

``RobotLinkSession`` adds one thing to the SDK's ``LinkSession``: a generic
request on the same encrypted session (the bridge needs ``GET /chat/history``
to read Muse's reply, as the ESP32 firmware does).

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

from gadget import restrict

log = logging.getLogger(__name__)


class RobotLinkSession(LinkSession):
    async def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        """Send one request on this session; return ``(status, decoded JSON or text)``."""
        data = json.dumps(body).encode() if body is not None else b""
        headers = [
            Header("x-request-id", str(uuid.uuid4())),
            Header("x-app-id", link_client.APP_ID),
        ]
        if body is not None:
            headers.insert(0, Header("Content-Type", "application/json"))
        encrypted = self._transport.encrypt_http_request(method, path, data, headers=headers)
        pending = link_client._Request(asyncio.get_running_loop().create_future())
        self._requests[encrypted.stream_id] = pending
        try:
            await self._send_frames(encrypted.frames)
            status, raw = await asyncio.wait_for(pending.done, link_client.REQUEST_TIMEOUT_S)
        finally:
            self._requests.pop(encrypted.stream_id, None)
        try:
            decoded = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            decoded = raw.decode("utf-8", errors="replace")[:2000]
        return status, decoded


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
