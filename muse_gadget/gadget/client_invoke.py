"""Answer Muse's ``client.invoke`` events, so the robot's tools work by voice.

For turns the gadget sends itself, Muse asks for device commands with a
``client.invoke`` event on the ``/chat/subscribe`` stream instead of a
``link.invoke`` on ``/link-control``. This is undocumented and may change.
The answer is the documented ``link.result`` on ``/link-control``, with
``id`` set to the event's ``invoke_id``; nothing else is sent.

Only ``restrict.ALLOWED_COMMANDS`` run. Anything else gets an error result.
Events without a usable ``command_id`` and ``invoke_id`` are ignored, and each
``invoke_id`` is answered once. Logs carry names, ids and ok/error only.

``MUSE_CLIENT_INVOKE=0`` turns this off.
"""

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

from musegadget import executor as upstream_executor

from gadget import restrict

ENV = "MUSE_CLIENT_INVOKE"
EVENT = "client.invoke"
MAX_SEEN = 1024
MAX_PARAMS_JSON = 64 * 1024
_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,200}")


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"


def safe_name(value) -> str:
    """``value`` if it is a short plain name, else "invalid" (for logs)."""
    return value if isinstance(value, str) and _NAME.fullmatch(value) else "invalid"


@dataclass(frozen=True)
class Invoke:
    invoke_id: str
    command: str
    params: dict | None       # None: params_json was missing a JSON object
    timeout_ms: int | None


def parse(event: dict) -> Invoke | None:
    """The request in a ``client.invoke`` event, or None if the event is malformed."""
    if not isinstance(event, dict) or event.get("event") != EVENT:
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    invoke_id, command = payload.get("invoke_id"), payload.get("command_id")
    if safe_name(invoke_id) == "invalid" or safe_name(command) == "invalid":
        return None
    # Omitted params_json means no parameters. An explicit null, "" or non-string is refused,
    # so a broken request can't turn into e.g. a random dance.
    raw = payload.get("params_json", "{}")
    params = None
    if isinstance(raw, str) and raw and len(raw) <= MAX_PARAMS_JSON:
        try:
            decoded = json.loads(raw)
        except ValueError:
            decoded = None
        params = decoded if isinstance(decoded, dict) else None
    timeout = payload.get("timeout_ms")
    timeout_ms = timeout if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout > 0 else None
    return Invoke(invoke_id, command, params, timeout_ms)


def refusal(invoke: Invoke) -> dict | None:
    """An error result if ``invoke`` must not run, else None."""
    if invoke.command not in restrict.ALLOWED_COMMANDS or invoke.command in restrict.BLOCKED_COMMANDS:
        return upstream_executor.error(f"unsupported command: {invoke.command}")
    if invoke.params is None:
        return upstream_executor.error("params_json must be a JSON object")
    return None


class SeenIds:
    """The last ``MAX_SEEN`` invoke ids, so each is answered once."""

    def __init__(self, limit: int = MAX_SEEN) -> None:
        self._ids: OrderedDict[str, None] = OrderedDict()
        self._limit = limit

    def add(self, invoke_id: str) -> bool:
        """True if ``invoke_id`` is new (and now remembered)."""
        if invoke_id in self._ids:
            return False
        self._ids[invoke_id] = None
        while len(self._ids) > self._limit:
            self._ids.popitem(last=False)
        return True
