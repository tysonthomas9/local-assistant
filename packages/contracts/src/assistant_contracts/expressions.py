"""The `express` vocabulary shared by the brain and every body.

Seeded with the 42 emotion intent names of Pollen's conversation app
(`EMOTION_INTENTS` in tools/play_emotion.py, Apache-2.0). Only the names live here; each body maps
them to its own moves (for Reachy: config/bodies/reachy.toml, phase 3).
"""

from enum import StrEnum


class ExpressName(StrEnum):
    """Semantic expression intents the brain may send in `express`."""

    RANDOM = "random"
    HAPPY = "happy"
    EXCITED = "excited"
    LOVING = "loving"
    GRATEFUL = "grateful"
    SUCCESS = "success"
    THINKING = "thinking"
    ATTENTIVE = "attentive"
    CONFUSED = "confused"
    UNCERTAIN = "uncertain"
    SAD = "sad"
    DOWNCAST = "downcast"
    LONELY = "lonely"
    ANGRY = "angry"
    IRRITATED = "irritated"
    DISPLEASED = "displeased"
    DISGUSTED = "disgusted"
    SCARED = "scared"
    ANXIOUS = "anxious"
    SURPRISED = "surprised"
    AMAZED = "amazed"
    CALMING = "calming"
    RELIEF = "relief"
    IMPATIENT = "impatient"
    EMBARRASSED = "embarrassed"
    BORED = "bored"
    TIRED = "tired"
    SLEEPY = "sleepy"
    YES = "yes"
    YES_UNDERSTANDING = "yes_understanding"
    NO = "no"
    NO_SAD = "no_sad"
    NO_EXCITED = "no_excited"
    NO_FIRM = "no_firm"
    WELCOMING = "welcoming"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    GO_AWAY = "go_away"
    HELPFUL = "helpful"
    DANCE = "dance"
    ELECTRIC = "electric"
    DYING = "dying"
