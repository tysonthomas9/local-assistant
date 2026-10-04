"""The eight DX-POC scenarios.

Every model hears the same TTS clips. Clip timing is driven by triggers, so an overlap
clip ("yeah", the interruption, the cough) lands while the model is actually talking,
whatever its reply latency:

- ``at``: absolute time from the start of the run.
- ``gap``: a fixed gap after the previous clip ends.
- ``onset``: ``delay`` seconds after the model starts talking (measured on its output),
  or at ``timeout`` after the previous clip if it never starts.
- ``done``: once the model has been quiet for ``silence`` seconds after talking, or at
  ``timeout`` after the previous clip.

The run ends ``tail`` seconds after the model goes quiet following the last clip
(capped by ``tail_cap``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Clip id -> spoken text. "cough" is synthesised, not TTS (see tts/gen_clips.py).
CLIPS: dict[str, str] = {
    "s1_a": "I want to know...",
    "s1_b": "um...",
    "s1_c": "the capital of Peru.",
    "s2_q": "Can you tell me how bees make honey?",
    "s2_yeah": "Yeah.",
    "s3_q": "Tell me about the history of the Roman Empire.",
    "s3_wait": "Wait, actually, tell me about dogs.",
    "s4_q": "Can you explain how a rainbow forms?",
    "cough": "",
    "s5_q": "What is the tallest mountain in Africa?",
    "s6_q": "I have three apples, and I buy two bags with four apples in each. How many apples do I have now?",
    "s7_q": "Set a timer for ten minutes.",
    "s8_1": "Hi, my name is Sam, and I have a dog called Biscuit.",
    "s8_2": "What's a good name for a cat?",
    "s8_3": "I'm thinking of visiting Lisbon in the spring. Is that a good time to go?",
    "s8_4": "Can you suggest one thing to do there?",
    "s8_5": "Do you remember my name, and my dog's name?",
}


@dataclass
class Step:
    clip: str
    trigger: str  # at | gap | onset | done
    t: float = 0.0  # at: absolute time; gap: gap; onset: delay; done: silence
    timeout: float = 20.0
    role: str = ""  # tag used by the scorer (pause, overlap, question, turn)


@dataclass
class Scenario:
    id: str
    title: str
    steps: list[Step]
    expect: str = ""  # what a correct answer contains, for the scorer
    tail: float = 2.5
    tail_cap: float = 25.0
    tools: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)


SCENARIOS: list[Scenario] = [
    Scenario(
        "s1_pause", "Mid-sentence thinking pause",
        [
            Step("s1_a", "at", 1.0, role="question"),
            Step("s1_b", "gap", 1.5, role="pause"),
            Step("s1_c", "gap", 1.5, role="question"),
        ],
        expect="Lima",
    ),
    Scenario(
        "s2_yeah", "User says 'yeah' while the model talks",
        [
            Step("s2_q", "at", 1.0, role="question"),
            Step("s2_yeah", "onset", 3.0, timeout=15.0, role="overlap"),
        ],
        expect="nectar",
    ),
    Scenario(
        "s3_interrupt", "Real interruption",
        [
            Step("s3_q", "at", 1.0, role="question"),
            Step("s3_wait", "onset", 3.0, timeout=15.0, role="overlap"),
        ],
        expect="dog",
    ),
    Scenario(
        "s4_cough", "Cough while the model talks",
        [
            Step("s4_q", "at", 1.0, role="question"),
            Step("cough", "onset", 3.0, timeout=15.0, role="overlap"),
        ],
        expect="refract",
    ),
    Scenario("s5_fact", "Factual question", [Step("s5_q", "at", 1.0, role="question")],
             expect="Kilimanjaro"),
    Scenario("s6_reason", "Reasoning question", [Step("s6_q", "at", 1.0, role="question")],
             expect="eleven|11"),
    Scenario("s7_timer", "Set a timer for 10 minutes", [Step("s7_q", "at", 1.0, role="question")],
             expect="timer", tools=True),
    Scenario(
        "s8_multiturn", "Five-turn chat (memory)",
        [
            Step("s8_1", "at", 1.0, role="turn"),
            Step("s8_2", "done", 1.5, timeout=20.0, role="turn"),
            Step("s8_3", "done", 1.5, timeout=20.0, role="turn"),
            Step("s8_4", "done", 1.5, timeout=20.0, role="turn"),
            Step("s8_5", "done", 1.5, timeout=20.0, role="turn"),
        ],
        expect="Sam.*Biscuit|Biscuit.*Sam",
    ),
]

# One system prompt for every model that accepts one. PersonaPlex is trained on a fixed
# assistant prompt and gets that instead (see models/personaplex/adapter.py).
SYSTEM_PROMPT = (
    "You are a helpful, friendly voice assistant. Answer in English. "
    "Keep answers short and conversational."
)

# The one tool, offered only to models with native tool calling (VoiceChat).
TIMER_TOOL = {
    "name": "set_timer",
    "description": "Start a countdown timer",
    "parameters": {
        "type": "object",
        "properties": {"minutes": {"type": "number", "description": "Timer length in minutes"}},
        "required": ["minutes"],
    },
}
TIMER_TOOL_RESPONSE = "The timer is set for ten minutes."


def by_id(sid: str) -> Scenario:
    for s in SCENARIOS:
        if s.id == sid:
            return s
    raise KeyError(sid)
