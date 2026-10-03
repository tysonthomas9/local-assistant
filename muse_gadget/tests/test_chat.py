import asyncio

import pytest

from gadget import chat
from fakes import FakeLink

FAST = chat.TurnOptions(poll_s=0, settle_s=0)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.25)


def run_turn(link, text, options=FAST, clock=None):
    clock = clock or Clock()
    return asyncio.run(chat.turn(link, text, options, clock=clock, sleep=clock.sleep))


def test_turn_returns_the_reply_to_our_message_only():
    link = FakeLink({"hello": ["Hi there!"]})
    assert run_turn(link, "hello") == "Hi there!"
    message, session_id = link.sent[0]
    assert session_id == chat.DEFAULT_SESSION_ID
    assert message.startswith(chat.DEFAULT_STYLE_HINT) and message.endswith("\nhello")
    assert "session_id=reachy-mini-robot" in link.paths[0]


def test_turn_joins_several_reply_rows_and_strips_markdown():
    link = FakeLink({"weather?": ["**Sunny** today.", "# Enjoy `it`"]})
    assert run_turn(link, "weather?") == "Sunny today. Enjoy it"


def test_turn_waits_for_display_text_ready():
    link = FakeLink({"slow": ["Done."]}, ready_after_polls=3)
    assert run_turn(link, "slow") == "Done."


def test_turn_times_out_without_reply():
    link = FakeLink({})
    with pytest.raises(chat.TurnError) as err:
        run_turn(link, "anyone?", chat.TurnOptions(poll_s=1, settle_s=0, timeout_s=5))
    assert (err.value.code, err.value.status) == ("timeout", 504)


def test_turn_main_chat_and_no_hint():
    link = FakeLink({"hey": ["Yo."]})
    assert run_turn(link, "hey", chat.TurnOptions(session_id=None, style_hint="", poll_s=0, settle_s=0)) == "Yo."
    assert link.sent == [("hey", None)]
    assert all("session_id" not in p for p in link.paths)


def test_refused_message_is_muse_error():
    class Refusing(FakeLink):
        async def send_chat(self, message, session_id=None):
            return {"ok": False, "status": 403, "response": None}

    with pytest.raises(chat.TurnError) as err:
        run_turn(Refusing(), "hi")
    assert (err.value.code, err.value.status) == ("muse_error", 502)
