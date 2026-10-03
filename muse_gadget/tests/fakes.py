"""A fake Muse link: answers /chat/stream and streams /chat/subscribe events like the VM."""

from __future__ import annotations

import asyncio


class FakeSubscription:
    """Events waiting on the stream. With a fake ``clock``, waiting advances it instead of sleeping."""

    def __init__(self, clock=None) -> None:
        self.events: list[dict] = [{"type": "ack", "ok": True}]
        self.clock = clock
        self.closed = False

    async def next(self, timeout: float) -> dict | None:
        if self.events:
            return self.events.pop(0)
        if self.clock is not None:
            self.clock.now += max(timeout, 0)
        else:
            await asyncio.sleep(max(timeout, 0))
        return None

    async def close(self) -> None:
        self.closed = True


class FakeLink:
    """``replies`` maps user text to Muse's reply messages.

    ``mode="delta"`` streams each reply as message_start, two text_append pieces
    and message_done; ``mode="full"`` sends a whole ``message.assistant``, first
    not ready and then ready. Each turn also streams events that aren't ours: a
    reply to someone else's message, the user-message echo and a replayed seq.
    """

    def __init__(self, replies: dict | None = None, mode: str = "delta", clock=None) -> None:
        self.replies = replies or {}
        self.mode = mode
        self.clock = clock
        self.sent: list[tuple[str, str | None]] = []
        self.subs: list[FakeSubscription] = []
        self.sub_sessions: list[str | None] = []
        self._seq = 100
        self._next_id = 0

    def _id(self) -> str:
        self._next_id += 1
        return f"m-{self._next_id}"

    def _event(self, name: str, **payload) -> dict:
        self._seq += 1
        return {"type": "event", "seq": self._seq, "event": name, "payload": payload}

    async def subscribe(self, session_id: str | None = None) -> FakeSubscription:
        self.sub_sessions.append(session_id)
        sub = FakeSubscription(self.clock)
        self.subs.append(sub)
        return sub

    async def send_chat(self, message: str, session_id: str | None = None) -> dict:
        self.sent.append((message, session_id))
        note = self._id()
        sub = self.subs[-1]
        other = self._id()
        sub.events += [
            self._event("message.user", message_id=note, display_text=message),
            self._event("delta.message_start", message_id=other, reply_to_message_id="m-elsewhere"),
            self._event("delta.message_done", message_id=other, display_text="not for the robot"),
            self._event("agent.status", activity_code="thinking"),
        ]
        sub.events.append(dict(sub.events[-1]))   # a replayed seq is dropped
        for reply in self.replies.get(message.splitlines()[-1], []):
            mid = self._id()
            if self.mode == "delta":
                half = len(reply) // 2
                sub.events += [
                    self._event("delta.message_start", message_id=mid, reply_to_message_id=note),
                    self._event("delta.text_append", message_id=mid, text=reply[:half]),
                    self._event("delta.text_append", message_id=mid, text=reply[half:]),
                    self._event("delta.message_done", message_id=mid),
                ]
            else:
                sub.events += [
                    self._event("message.assistant", message_id=mid, reply_to_message_id=note,
                                display_text=reply[:3], display_text_ready=False),
                    self._event("message.assistant", message_id=mid, reply_to_message_id=note,
                                display_text=reply, display_text_ready=True),
                ]
        sub.events.append(self._event("agent.status", activity_code="idle"))
        return {"ok": True, "status": 200, "response": {"ok": True, "result": {"message_id": note}}}
