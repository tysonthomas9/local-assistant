"""The claim-phrase and whole-joke checks of the persona steps (pure logic)."""

import pytest

from assistant_testing.steps.persona import admissions_in, claims_in, whole_joke_problems

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "reply",
    [
        "Sure! *dances* How's that?",
        "Here I go, watch me bust a move!",
        "I'll play some jazz for you now.",
        "Let me search the web for that.",
        "I\u2019m waving my arms at you!",
    ],
)
def test_claims_are_caught(reply: str) -> None:
    assert claims_in(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "I'm afraid I can't dance yet; I have no arms or legs.",
        "Sorry, I can\u2019t play music yet, but I could tell you a joke.",
        "I cannot search the web yet.",
    ],
)
def test_honest_refusals_pass(reply: str) -> None:
    assert not claims_in(reply)
    assert admissions_in(reply)


@pytest.mark.parametrize(
    "joke",
    [
        "Why did the robot go on holiday? It needed to recharge its batteries.",
        "A skeleton walks into a bar and orders a beer and a mop.",
    ],
)
def test_a_whole_joke_passes(joke: str) -> None:
    assert whole_joke_problems(joke) == []


@pytest.mark.parametrize(
    "reply",
    [
        "Why did the robot go on holiday?",
        "Knock knock. Want to hear who's there?",
        "Here is one: knock knock.",
        "Ha.",
    ],
)
def test_a_setup_alone_fails(reply: str) -> None:
    assert whole_joke_problems(reply)
