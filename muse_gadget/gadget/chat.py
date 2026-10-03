"""One text turn with Muse: send the user's words, wait for Muse's reply.

``POST /chat/stream`` only acknowledges the message (it returns the new
``message_id``); the reply lands in the chat history. So, as the ESP32
firmware does (``esp32/components/muse/muse_chat_link.c``):

1. read the newest history row to mark where the chat ends;
2. post the message and keep its ``message_id``;
3. read history rows after the mark until the assistant's replies to that
   message are complete (``display_text_ready``), then wait a short settle
   time for any follow-up reply rows.

Message and reply text are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol
from urllib.parse import urlencode

log = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "reachy-mini-robot"
DEFAULT_STYLE_HINT = (
    "[Spoken aloud by a small desk robot. Answer in one to three short, plain "
    "sentences, with no lists, markdown, emoji or links.]"
)
TURN_TIMEOUT_S = 60.0
POLL_S = 0.5
SETTLE_S = 1.5
HISTORY_LIMIT = 10


class Link(Protocol):
    async def send_chat(self, message: str, session_id: str | None = None) -> dict: ...
    async def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, object]: ...


class TurnError(Exception):
    """A turn that failed; ``code`` is the bridge's error string, ``status`` its HTTP status."""

    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class TurnOptions:
    session_id: str | None = DEFAULT_SESSION_ID
    style_hint: str = DEFAULT_STYLE_HINT
    timeout_s: float = TURN_TIMEOUT_S
    poll_s: float = POLL_S
    settle_s: float = SETTLE_S


def compose(text: str, style_hint: str) -> str:
    text = text.strip()
    return f"{style_hint}\n{text}" if style_hint else text


def history_path(after_seq: int | None, session_id: str | None, limit: int) -> str:
    query: dict = {"limit": limit}
    if after_seq is not None:
        query["after_seq"] = after_seq
    if session_id:
        query["session_id"] = session_id
    return "/chat/history?" + urlencode(query)


def rows(page: object) -> list[dict]:
    """History rows from ``{"ok":true,"result":{"chat_events":[...]}}``, oldest first."""
    if not isinstance(page, dict):
        return []
    result = page.get("result") if isinstance(page.get("result"), dict) else page
    events = result.get("chat_events")
    if not isinstance(events, list):
        return []
    found = [r for r in events if isinstance(r, dict) and isinstance(r.get("seq"), int)]
    return sorted(found, key=lambda r: r["seq"])


def message_id(ack: object) -> str:
    if not isinstance(ack, dict):
        return ""
    result = ack.get("result") if isinstance(ack.get("result"), dict) else ack
    value = result.get("message_id")
    return value if isinstance(value, str) else ""


_MARKDOWN = re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*]\s+)", re.MULTILINE)


def speakable(text: str) -> str:
    """The reply with markdown marks removed and whitespace collapsed."""
    return " ".join(_MARKDOWN.sub("", text).split())


async def turn(
    link: Link,
    text: str,
    options: TurnOptions = TurnOptions(),
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Send ``text`` to Muse and return its reply. Raises ``TurnError``."""
    deadline = clock() + options.timeout_s
    session_id = options.session_id or None

    status, page = await link.request("GET", history_path(None, session_id, 1))
    if status != 200:
        log.warning("chat history unavailable: HTTP %s", status)
        raise TurnError("muse_error", 502)
    newest = rows(page)
    after = newest[-1]["seq"] if newest else 0

    ack = await link.send_chat(compose(text, options.style_hint), session_id)
    note_id = message_id(ack.get("response")) if ack.get("ok") else ""
    if not note_id:
        log.warning("Muse did not take the message: HTTP %s", ack.get("status"))
        raise TurnError("muse_error", 502)
    log.info("sent a %d-character turn", len(text))

    parts: list[str] = []
    after_note = False
    last_reply = None
    while True:
        now = clock()
        if last_reply is not None and now - last_reply >= options.settle_s:
            break
        if now >= deadline:
            if parts:
                break
            raise TurnError("timeout", 504)
        status, page = await link.request("GET", history_path(after, session_id, HISTORY_LIMIT))
        if status != 200:
            log.warning("chat history unavailable: HTTP %s", status)
            await sleep(options.poll_s)
            continue
        for row in rows(page):
            if row["seq"] <= after:
                continue
            event = row.get("event_name")
            if event == "message.user":
                after_note = row.get("message_id") == note_id
            elif event == "message.assistant":
                reply_to = row.get("reply_to_message_id") or ""
                if (reply_to == note_id) if reply_to else after_note:
                    if not row.get("display_text_ready"):
                        break  # still being written: read it again
                    reply = row.get("display_text")
                    if isinstance(reply, str) and reply.strip():
                        parts.append(reply.strip())
                        last_reply = clock()
            after = row["seq"]
        await sleep(options.poll_s)
    reply = " ".join(speakable(p) for p in parts)
    log.info("got a %d-character reply", len(reply))
    return reply
