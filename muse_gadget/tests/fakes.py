"""A fake Muse link: answers /chat/history and /chat/stream like the VM."""

from __future__ import annotations


class FakeLink:
    """Chat history in memory. ``replies`` maps user text to the assistant rows it gets."""

    def __init__(self, replies: dict | None = None, ready_after_polls: int = 0) -> None:
        self.events: list[dict] = [
            {"seq": 7, "event_name": "message.assistant", "message_id": "m-old",
             "reply_to_message_id": "m-older", "display_text": "old reply", "display_text_ready": True},
        ]
        self.replies = replies or {}
        self.ready_after_polls = ready_after_polls
        self.sent: list[tuple[str, str | None]] = []
        self.paths: list[str] = []
        self._next = 100

    def _add(self, **row) -> dict:
        self._next += 1
        row["seq"] = self._next
        self.events.append(row)
        return row

    async def send_chat(self, message: str, session_id: str | None = None) -> dict:
        self.sent.append((message, session_id))
        note = self._add(event_name="message.user", message_id=f"m-{self._next + 1}",
                         display_text=message, display_text_ready=True)
        text = message.splitlines()[-1]
        for reply in self.replies.get(text, []):
            self._add(event_name="message.assistant", message_id=f"m-{self._next + 1}",
                      reply_to_message_id=note["message_id"], display_text=reply,
                      display_text_ready=self.ready_after_polls == 0)
        return {"ok": True, "status": 200, "response": {"ok": True, "result": {"message_id": note["message_id"]}}}

    async def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, object]:
        assert method == "GET" and path.startswith("/chat/history?")
        self.paths.append(path)
        query = dict(p.split("=", 1) for p in path.split("?", 1)[1].split("&"))
        limit = int(query["limit"])
        if "after_seq" in query:
            if self.ready_after_polls:
                self.ready_after_polls -= 1
                if not self.ready_after_polls:
                    for row in self.events:
                        row["display_text_ready"] = True
            after = int(query["after_seq"])
            page = [r for r in self.events if r["seq"] > after][:limit]
        else:
            page = self.events[-limit:]
        return 200, {"ok": True, "result": {"chat_events": [dict(r) for r in page]}}
