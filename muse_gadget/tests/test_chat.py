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
    assert message == "hello"   # the user's words only: no note by default
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


def test_style_note_is_off_unless_opted_in(monkeypatch):
    from gadget import __main__ as gadget_main

    monkeypatch.delenv("MUSE_STYLE_HINT_ON", raising=False)
    assert gadget_main.turn_options().style_hint == ""
    monkeypatch.setenv("MUSE_STYLE_HINT_ON", "1")
    hint = gadget_main.turn_options().style_hint
    assert hint == chat.STYLE_NOTE and "reachy" not in hint
    assert chat.compose("hello", hint) == f"{chat.STYLE_NOTE}\nhello"


# ------------------------------------------------------------------ sentences as they arrive
class TimedSub:
    """Scripted events at fixed times on a fake clock: [(time, event), ...]."""

    def __init__(self, clock, events):
        self.clock, self.events, self.closed = clock, list(events), False

    async def next(self, timeout):
        if self.events and self.events[0][0] <= self.clock.now + timeout:
            at, event = self.events.pop(0)
            self.clock.now = max(self.clock.now, at)
            return event
        self.clock.now += max(timeout, 0)
        return None

    async def close(self):
        self.closed = True


class TimedLink:
    """Muse's reply as timed events; ``script(ev, note)`` returns [(time, event), ...]."""

    def __init__(self, clock, script):
        self.clock, self.script, self.seq = clock, script, 0
        self.sent = []

    def ev(self, name, **payload):
        self.seq += 1
        return {"type": "event", "seq": self.seq, "event": name, "payload": payload}

    async def subscribe(self, session_id=None):
        self.sub = TimedSub(self.clock, [])
        return self.sub

    async def send_chat(self, message, session_id=None):
        self.sent.append(message)
        self.sub.events = self.script(self.ev, "u-1")
        return {"ok": True, "status": 200, "response": {"ok": True, "result": {"message_id": "u-1"}}}


def stream_turn(script, options):
    clock = Clock()
    link = TimedLink(clock, script)
    heard = []

    async def on_sentence(sentence):
        heard.append((sentence, clock.now))

    reply = asyncio.run(chat.turn(link, "hi", options, clock=clock, on_sentence=on_sentence))
    return reply, heard, clock.now


def test_first_sentence_is_handed_over_before_message_done():
    def script(ev, note):
        return [(0.5, ev("delta.message_start", message_id="a", reply_to_message_id=note)),
                (0.6, ev("delta.text_append", message_id="a", text="Hello **there**. How")),
                (3.0, ev("delta.text_append", message_id="a", text=" are you?")),
                (3.1, ev("delta.message_done", message_id="a"))]
    reply, heard, _ = stream_turn(script, chat.TurnOptions(settle_s=0.3))
    assert heard == [("Hello there.", 0.6), ("How are you?", 3.1)]
    assert reply == "Hello there. How are you?"


def test_speech_never_waits_for_the_settle_time():
    def script(ev, note):
        return [(1.0, ev("message.assistant", message_id="a", reply_to_message_id=note,
                         display_text="Done", display_text_ready=True))]
    _, heard, ended = stream_turn(script, chat.TurnOptions(settle_s=1.5))
    assert heard == [("Done", 1.0)]
    assert ended >= 2.5, "the settle time still decides when the turn ends"


def test_a_late_second_message_is_still_handed_over():
    def script(ev, note):
        return [(0.5, ev("message.assistant", message_id="a", reply_to_message_id=note,
                         display_text="First.", display_text_ready=True)),
                (0.7, ev("delta.message_start", message_id="b", reply_to_message_id=note)),
                (0.75, ev("delta.text_append", message_id="b", text="Second")),
                (0.8, ev("delta.message_done", message_id="b"))]
    reply, heard, _ = stream_turn(script, chat.TurnOptions(settle_s=0.3))
    assert heard == [("First.", 0.5), ("Second", 0.8)]
    assert reply == "First. Second"


def test_timeout_hands_over_the_text_so_far_and_ends():
    def script(ev, note):
        return [(0.5, ev("delta.message_start", message_id="a", reply_to_message_id=note)),
                (0.6, ev("delta.text_append", message_id="a", text="Sure! Let me thin"))]
    reply, heard, ended = stream_turn(script, chat.TurnOptions(settle_s=0.3, timeout_s=5))
    assert heard == [("Sure!", 0.6), ("Let me thin", 5.0)]
    assert reply == "Sure! Let me thin" and ended == 5.0


def test_one_shot_turn_is_unchanged_without_on_sentence():
    def script(ev, note):
        return [(0.6, ev("delta.message_start", message_id="a", reply_to_message_id=note)),
                (0.7, ev("delta.text_append", message_id="a", text="One. Two.")),
                (0.8, ev("delta.message_done", message_id="a"))]
    clock = Clock()
    assert asyncio.run(chat.turn(TimedLink(clock, script), "hi", chat.TurnOptions(settle_s=0.3),
                                 clock=clock)) == "One. Two."


def test_complete_length():
    assert chat.complete_length("Hi there. How") == 9
    assert chat.complete_length("Pi is 3.14 ok") == 0
    assert chat.complete_length("Done.") == 0   # the end comes with message_done
    assert chat.complete_length("one\ntwo") == 4
