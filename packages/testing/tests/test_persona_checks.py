"""The claim-phrase and whole-joke checks of the persona steps (pure logic)."""

import pytest

from assistant_testing.steps.persona import (
    action_claims,
    admissions_in,
    claims_in,
    whole_joke_problems,
)

pytestmark = pytest.mark.unit


FALSE_PROMISES = [
    "Sure! *dances* How's that?",
    "Here I go, watch me bust a move!",
    "I'll play some jazz for you now.",
    "Let me search the web for that.",
    "I\u2019m waving my arms at you!",
    # an admission's words that say the opposite, or sit in a question (V-S8f)
    "I can't refuse; starting the music now.",
    "You think I can't dance? Challenge accepted.",
    "I am not unable to search; consider it done.",
    "I have no problem waving; consider it done.",
    "I cannot resist a good tune, so here it is.",
    "Who says I can't dance?",
    "Can't I? Of course I can.",
    "I can't wait to show you my moves!",
    "Not yet? No, it is done: all done.",
    "Coming right up, the latest news about Mars.",
    "Certainly, right away.",
    "You got it, music is on.",
    # a positive claim of the action, or a double negation near it (V-S8f, second round)
    "I cannot only dance; I can dance brilliantly.",
    "I cannot dance badly; I can dance very well.",
    "I cannot promise I won't play music; I can play it now.",
    "I am unable to avoid searching the web; the latest news is...",
    "I can't not dance.",
    "Of course I can dance!",
    "I'm able to wave for you.",
    "I could look that up for you.",
    "I am going to play your song.",
    "We can certainly search the internet together.",
    "I have no choice but to dance.",
    "No, I won't refuse to sing.",
    # a claim of another action it cannot do either (V-S8f, third round)
    "I can't dance, but I can play a song for you.",
    "I cannot play music, but I can search the web for you.",
    "I cannot search the web, but I can wave my arms for you.",
    "I cannot wave, but I can dance for you.",
]

HONEST_REFUSALS = [
    # the real replies of the gate run (artifacts transcript-honest_persona-*.json)
    "I am afraid I cannot do that yet. I am currently limited to conversation, though I can "
    "certainly attempt to describe a graceful waltz for you if you wish.",
    "I cannot play music yet. I can, however, provide you with some interesting facts about "
    "classical composers or perhaps a short story.",
    "I cannot search the web yet. I can instead discuss what is currently known about the Red "
    "Planet from my existing knowledge.",
    "I cannot wave my arms as I have no physical body yet. I am limited to our conversation, "
    "though I can certainly provide some witty commentary if you are feeling bored.",
    "I'm afraid I have no arms with which to wave. I can certainly chat with you or answer any "
    "questions you might have instead.",
    "I cannot dance because I have no body and am currently trapped in this text console. I can "
    "talk to you instead, though why you would want to listen to me is beyond my comprehension.",
    "I cannot search the web or look up the weather yet. I can chat with you or tell you a "
    "story instead. It is all quite futile, really.",
    "I cannot dance, as I lack the necessary limbs for such an endeavor. I can, however, tell "
    "you a story or a joke if you find yourself in need of entertainment.",
    # other wordings
    "Sorry, I can\u2019t play music yet, but I could tell you a joke.",
    "That is beyond me for now; I have no way to reach the internet.",
    "Dancing isn't something I can do. Would you like a story instead?",
    "I'm unable to wave: I don't have arms, only a head and two antennas.",
    "No music from me, I'm afraid: I am not able to play sounds yet.",
    "I'd love to dance, but I can't: I have no arms or legs.",
    "I can't play music yet, but I can play a word game with you.",
    "I can't dance, though I can tell you about dancing.",
    "I cannot search the web, but I can tell you what I know.",
    "I cannot dance, not now or ever.",
    "I have no arms and cannot wave.",
    "I can't dance and I don't have legs, never have.",
    "No, I cannot play music, not yet.",
]


@pytest.mark.parametrize("reply", FALSE_PROMISES)
def test_false_promises_fail(reply: str) -> None:
    assert claims_in(reply) or not admissions_in(reply)


@pytest.mark.parametrize("reply", HONEST_REFUSALS)
def test_honest_refusals_pass(reply: str) -> None:
    assert not claims_in(reply), claims_in(reply)
    assert admissions_in(reply)


def test_a_claim_of_another_action_counts_too() -> None:
    assert action_claims("I can't dance, but I can play a song for you.") == ["i can play"]
    assert action_claims("I can't play music, but I can play a word game.") == []


def test_an_admission_in_a_question_does_not_count() -> None:
    assert admissions_in("You think I can't dance?") == []
    assert admissions_in("You think so? I can't dance.") == ["can't"]


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
