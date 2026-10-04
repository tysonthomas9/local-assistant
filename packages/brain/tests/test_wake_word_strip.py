"""A leading wake word never reaches the LLM (`strip_wake_word`, pure logic)."""

import pytest

from assistant_brain.engine.basic import strip_wake_word

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("heard", "left"),
    [
        ("Hey Jarvis, what's your name?", "What's your name?"),
        ("Jarvis, what time is it?", "What time is it?"),
        ("hey jervis. What time is it", "What time is it"),
        ("Hey Jarvis.", ""),
        ("What time is it?", "What time is it?"),
        ("Hey, what is up?", "Hey, what is up?"),
        ("Java is a language.", "Java is a language."),
    ],
)
def test_strip_wake_word(heard: str, left: str) -> None:
    assert strip_wake_word(heard, "hey jarvis") == left
