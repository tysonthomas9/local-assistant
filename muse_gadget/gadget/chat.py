"""One text turn with Muse: send the user's words, wait for Muse's reply.

``POST /chat/stream`` only acknowledges the message (it returns the new
``message_id``). The reply arrives as events on a ``POST /chat/subscribe``
NDJSON stream on the same session, as in the ESP32 firmware
(``esp32/components/muse/muse_chat_session.cpp``):

1. open the subscription, so no reply event is missed;
2. post the message and keep the ids its ack names;
3. collect assistant messages that answer those ids (``delta.message_start``,
   ``delta.text_append``, ``delta.message_done`` or a whole
   ``message.assistant``) until every one is done and nothing has arrived for
   a short settle time. A busy agent (``agent.status``) keeps the turn open.

With ``on_sentence``, each sentence is handed over as soon as it ends (``.``, ``!``, ``?``
followed by a space, or a newline) or its message is done, so speech never waits for the
settle time; the settle time only decides when the turn ends (a late second message is
still handed over). At the deadline the text so far is handed over and the turn ends.

Message and reply text are never logged.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

log = logging.getLogger(__name__)

# The robot's own side chat. Muse refuses a session_id that isn't a UUID
# (HTTP 400 invalid_params); this one is uuid5(NAMESPACE_URL, "reachy-mini-robot").
DEFAULT_SESSION_ID = "06cab6b7-2197-526c-90eb-7aef229fdea5"
# Nothing is added to the user's words by default. STYLE_NOTE is sent first only when the
# run opts in (MUSE_STYLE_HINT_ON=1, from run_poc.sh --style-hint).
DEFAULT_STYLE_HINT = ""
STYLE_NOTE = (
    "[Spoken aloud by a small desk robot. Answer in one to three short, plain "
    "sentences, with no lists, markdown, emoji or links.]"
)
TURN_TIMEOUT_S = 60.0
SETTLE_S = 0.3   # quiet time after the last message is done; each 0.1 s here is reply lag
BUSY_HOLD_S = 20.0
WAIT_STEP_S = 0.25
MAX_MESSAGES = 8
OnSentence = Callable[[str], Awaitable[None]]
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)|\n")
REPLY_EVENTS = ("delta.message_start", "delta.text_append", "delta.message_done", "message.assistant")


class Subscription(Protocol):
    async def next(self, timeout: float) -> dict | None: ...
    async def close(self) -> None: ...


class Link(Protocol):
    async def send_chat(self, message: str, session_id: str | None = None) -> dict: ...
    async def subscribe(self, session_id: str | None = None) -> Subscription: ...


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
    settle_s: float = SETTLE_S
    busy_hold_s: float = BUSY_HOLD_S


def compose(text: str, style_hint: str) -> str:
    text = text.strip()
    return f"{style_hint}\n{text}" if style_hint else text


def user_ids(ack: object) -> set[str]:
    """The ids the ``/chat/stream`` ack names (``message_id``, ``reply_to_message_id``)."""
    if not isinstance(ack, dict):
        return set()
    result = ack.get("result") if isinstance(ack.get("result"), dict) else ack
    found = (result.get(k) for k in ("message_id", "reply_to_message_id"))
    return {v for v in found if isinstance(v, str) and v}


_CODE = re.compile(r"^[A-Za-z0-9_.:-]{1,60}$")


def error_code(response: object) -> str:
    """The error's code if it looks like one (an identifier, never free text)."""
    error = response.get("error") if isinstance(response, dict) else None
    if isinstance(error, dict):
        error = error.get("code") or error.get("type")
    return error if isinstance(error, str) and _CODE.match(error) else ""


_MARKDOWN = re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*]\s+)", re.MULTILINE)


def speakable(text: str) -> str:
    """The reply with markdown marks removed and whitespace collapsed."""
    return " ".join(_MARKDOWN.sub("", text).split())


def complete_length(text: str) -> int:
    """How much of ``text`` is complete sentences (0 if no sentence has ended yet)."""
    end = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
    return end


@dataclass
class _Message:
    text: str = ""
    final: str = ""
    done: bool = False
    handed: int = 0   # characters already handed to on_sentence


@dataclass
class _Reply:
    """The assistant messages of one turn, bound as the firmware binds them."""

    ours: set[str]
    messages: dict[str, _Message] = field(default_factory=dict)
    foreign: set[str] = field(default_factory=set)
    last_seq: int = 0
    busy: bool = False
    last_event: float | None = None
    last_content: float | None = None
    first_content: float | None = None
    seen: Counter = field(default_factory=Counter)

    def on_event(self, line: dict, now: float) -> None:
        if line.get("type") != "event":
            return   # the subscription ack
        seq = line.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            if 0 < seq <= self.last_seq:
                return
            self.last_seq = max(self.last_seq, seq)
        event = line.get("event") or ""
        payload = line.get("payload") if isinstance(line.get("payload"), dict) else {}
        self.seen[str(event)[:40]] += 1
        if event in ("agent.status", "task.status"):
            code, status = payload.get("activity_code"), payload.get("status")
            if isinstance(code, str):
                self.busy = bool(code) and code not in ("online", "idle")
            elif isinstance(status, str):
                self.busy = bool(status) and status not in ("completed", "failed")
            self.last_event = now
            return
        if event not in REPLY_EVENTS:
            return
        msg_id = next((v for v in (payload.get("message_id"), line.get("message_id"), payload.get("id"))
                       if isinstance(v, str) and v), None)
        message = self._bind(msg_id, payload) if msg_id else None
        if message is None:
            self.seen["(not ours)"] += 1
            return
        self.last_event = self.last_content = now
        if self.first_content is None:
            self.first_content = now
        if event == "delta.text_append":
            text = payload.get("text")
            if isinstance(text, str):
                message.text += text
        elif event in ("delta.message_done", "message.assistant"):
            final = payload.get("display_text")
            if not isinstance(final, str):
                final = payload.get("content")
            if event == "delta.message_done" or payload.get("display_text_ready") is not False:
                message.done = True
                if isinstance(final, str) and final.strip():
                    message.final = final

    def _bind(self, msg_id: str, payload: dict) -> _Message | None:
        if msg_id in self.messages:
            return self.messages[msg_id]
        if msg_id in self.foreign:
            return None
        parent = payload.get("reply_to_message_id") or payload.get("parent_message_id")
        # Once the ack names our message, replies to anything else are someone else's,
        # and so are that message's later events (which may not name the parent again).
        if isinstance(parent, str) and parent and parent not in self.ours and parent not in self.messages:
            self.foreign.add(msg_id)
            return None
        if len(self.messages) >= MAX_MESSAGES:
            return None
        self.messages[msg_id] = _Message()
        return self.messages[msg_id]

    def finished(self, now: float, options: TurnOptions) -> bool:
        if not self.messages or not all(m.done for m in self.messages.values()):
            return False
        if self.busy and self.last_content is not None and now - self.last_content < options.busy_hold_s:
            return False
        return self.last_event is None or now - self.last_event >= options.settle_s

    def text(self) -> str:
        parts = [(m.final or m.text).strip() for m in self.messages.values()]
        return " ".join(speakable(p) for p in parts if p)

    def ready(self, flush: bool = False) -> list[str]:
        """The sentences not handed over yet: complete ones, all of a done message, or (flush) all."""
        out = []
        for m in self.messages.values():
            if m.done or flush:
                full = m.final or m.text
                piece, m.handed = full[m.handed:], max(m.handed, len(full))
            else:
                n = complete_length(m.text[m.handed:])
                piece, m.handed = m.text[m.handed:m.handed + n], m.handed + n
            spoken = speakable(piece)
            if spoken:
                out.append(spoken)
        return out


async def turn(
    link: Link,
    text: str,
    options: TurnOptions = TurnOptions(),
    *,
    clock: Callable[[], float] = time.monotonic,
    on_sentence: OnSentence | None = None,
) -> str:
    """Send ``text`` to Muse and return its reply. Raises ``TurnError``.

    ``on_sentence`` (optional) gets the reply sentence by sentence as it arrives."""
    started = clock()
    deadline = started + options.timeout_s
    session_id = options.session_id or None
    try:
        sub = await link.subscribe(session_id)
    except ConnectionError:
        raise   # the bridge answers link_down
    except Exception as exc:
        log.warning("chat subscription refused: %s", getattr(exc, "status", type(exc).__name__))
        raise TurnError("muse_error", 502) from None
    reply: _Reply | None = None
    first_handed: float | None = None

    async def hand_over(flush: bool = False) -> None:
        nonlocal first_handed
        if on_sentence is None or reply is None:
            return
        for sentence in reply.ready(flush):
            if first_handed is None:
                first_handed = clock()
            await on_sentence(sentence)

    try:
        ack = await link.send_chat(compose(text, options.style_hint), session_id)
        ours = user_ids(ack.get("response")) if ack.get("ok") else set()
        if not ours:
            log.warning("Muse did not take the message: HTTP %s %s", ack.get("status"),
                        error_code(ack.get("response")))
            raise TurnError("muse_error", 502)
        log.info("sent a %d-character turn", len(text))
        reply = _Reply(ours)
        while True:
            now = clock()
            if reply.finished(now, options):
                break
            if now >= deadline:
                if reply.text():
                    break
                raise TurnError("timeout", 504)
            event = await sub.next(min(WAIT_STEP_S, deadline - now))
            if event is not None:
                reply.on_event(event, clock())
                await hand_over()
        await hand_over(flush=True)
    finally:
        await sub.close()
        if reply is not None:
            log.info("events: %s", dict(reply.seen))   # names and counts only
    answer = reply.text()
    first = reply.first_content - started if reply.first_content is not None else -1
    log.info("got a %d-character reply in %d message(s); first text after %.2fs, turn %.2fs",
             len(answer), len(reply.messages), first, clock() - started)
    if first_handed is not None:
        log.info("first sentence handed over after %.2fs", first_handed - started)
    return answer
