import asyncio

import pytest

from gadget import chat
from fakes import FakeLink, FakeSubscription

FAST = chat.TurnOptions(settle_s=0)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def run_turn(link, text, options=FAST, clock=None):
    clock = clock or link.clock or Clock()
    link.clock = clock
    return asyncio.run(chat.turn(link, text, options, clock=clock))


def test_turn_returns_the_reply_to_our_message_only():
    link = FakeLink({"hello": ["Hi there!"]})
    assert run_turn(link, "hello") == "Hi there!"
    message, session_id = link.sent[0]
    assert session_id == chat.DEFAULT_SESSION_ID
    assert message.startswith(chat.DEFAULT_STYLE_HINT) and message.endswith("\nhello")
    assert link.sub_sessions == [chat.DEFAULT_SESSION_ID]   # side chat events come only if asked for
    assert link.subs[0].closed


def test_default_session_id_is_a_uuid():
    import uuid
    assert str(uuid.UUID(chat.DEFAULT_SESSION_ID)) == chat.DEFAULT_SESSION_ID


def test_turn_joins_several_messages_and_strips_markdown():
    link = FakeLink({"weather?": ["**Sunny** today.", "# Enjoy `it`"]})
    assert run_turn(link, "weather?") == "Sunny today. Enjoy it"


def test_whole_message_waits_for_display_text_ready():
    link = FakeLink({"slow": ["Done now."]}, mode="full")
    assert run_turn(link, "slow") == "Done now."


def test_turn_waits_for_the_settle_time():
    clock = Clock()
    link = FakeLink({"hi": ["Hello."]}, clock=clock)
    assert run_turn(link, "hi", chat.TurnOptions(settle_s=1.5), clock) == "Hello."
    assert clock.now >= 1.5


def test_busy_agent_keeps_the_turn_open():
    class Busy(FakeLink):
        async def send_chat(self, message, session_id=None):
            ack = await super().send_chat(message, session_id)
            self.subs[-1].events.pop()   # no "idle": still busy after the reply
            return ack

    clock = Clock()
    link = Busy({"hi": ["Working on it."]}, clock=clock)
    assert run_turn(link, "hi", chat.TurnOptions(settle_s=0, busy_hold_s=5), clock) == "Working on it."
    assert clock.now >= 5


def test_turn_times_out_without_reply():
    link = FakeLink({})
    with pytest.raises(chat.TurnError) as err:
        run_turn(link, "anyone?", chat.TurnOptions(settle_s=0, timeout_s=5))
    assert (err.value.code, err.value.status) == ("timeout", 504)
    assert link.subs[0].closed


def test_turn_main_chat_and_no_hint():
    link = FakeLink({"hey": ["Yo."]})
    assert run_turn(link, "hey", chat.TurnOptions(session_id=None, style_hint="", settle_s=0)) == "Yo."
    assert link.sent == [("hey", None)] and link.sub_sessions == [None]


def test_refused_message_is_muse_error():
    class Refusing(FakeLink):
        async def send_chat(self, message, session_id=None):
            return {"ok": False, "status": 403, "response": None}

    link = Refusing()
    with pytest.raises(chat.TurnError) as err:
        run_turn(link, "hi")
    assert (err.value.code, err.value.status) == ("muse_error", 502)
    assert link.subs[0].closed


def test_refused_subscription_is_muse_error():
    class NoSubscribe(FakeLink):
        async def subscribe(self, session_id=None):
            raise PermissionError(403)

    with pytest.raises(chat.TurnError) as err:
        run_turn(NoSubscribe(), "hi")
    assert (err.value.code, err.value.status) == ("muse_error", 502)


def test_reply_text_never_logged(caplog):
    caplog.set_level("DEBUG")
    run_turn(FakeLink({"secret question": ["secret answer"]}), "secret question")
    assert "secret" not in caplog.text and "not for the robot" not in caplog.text
    assert "events:" in caplog.text


def test_fake_subscription_sleeps_without_clock():
    sub = FakeSubscription()
    sub.events.clear()
    assert asyncio.run(sub.next(0.01)) is None
