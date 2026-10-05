"""The system prompt follows the persona file, the edge's body and the registered abilities
(`assistant_brain.persona`, pure logic)."""

import hashlib
from pathlib import Path

import pytest

from assistant_brain.persona import (
    Ability,
    BodyInfo,
    Personas,
    body_section,
    capabilities_section,
    read_persona,
    system_prompt,
)
from assistant_contracts.capabilities import Capabilities, MotionCaps
from assistant_core.config import ConfigError

pytestmark = pytest.mark.unit

PERSONAS = Path(__file__).resolve().parents[3] / "config" / "personas"
REACHY = BodyInfo(
    "reachy",
    Capabilities(
        motion=MotionCaps(expressions=["happy"], look_at=["user", "world", "doa"], attention=True)
    ),
)


@pytest.mark.parametrize("persona_id", ["jarvis", "marvin"])
def test_the_shipped_personas_load_with_their_marker(persona_id: str) -> None:
    persona = Personas(PERSONAS, [persona_id]).get(persona_id)
    sha = hashlib.sha256((PERSONAS / f"{persona_id}.md").read_bytes()).hexdigest()
    assert persona.marker == f"{persona_id}.md#{sha[:12]}"
    assert persona.text.startswith(f"You are {persona_id.capitalize()}")
    assert "+++" not in persona.text


def test_a_missing_persona_fails_at_start(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Personas(tmp_path, ["nobody"])


def test_unclosed_front_matter_is_refused(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("+++\nschema_version = 1\nYou are X.\n")
    with pytest.raises(ConfigError, match="not closed"):
        read_persona(tmp_path, "x")


def test_reachy_has_a_head_and_antennas_and_no_arms() -> None:
    text = body_section(REACHY)
    assert "two antennas" in text
    assert "no arms" in text
    assert "follows their face" in text
    assert "cannot dance" in text


def test_tracking_off_is_not_claimed() -> None:
    body = BodyInfo("reachy", Capabilities(motion=MotionCaps(look_at=["doa"], attention=True)))
    assert "face" not in body_section(body).replace("in your face", "")


def test_console_and_unknown_bodies_cannot_move() -> None:
    assert "cannot move" in body_section(BodyInfo("console", Capabilities()))
    assert "cannot move" in body_section(BodyInfo("hovercraft", Capabilities()))
    assert "cannot move" in body_section(None)


def test_chat_only_names_what_it_cannot_do() -> None:
    text = capabilities_section(())
    assert "no tools or skills yet" in text
    for missing in ("play music", "search the web", "set timers"):
        assert missing in text


def test_a_registered_ability_is_offered_and_no_longer_denied() -> None:
    text = capabilities_section([Ability("timers", "set timers and alarms")])
    assert "set timers and alarms" in text
    assert "set timers, alarms or reminders" not in text
    assert "no tools or skills yet" not in text
    assert "play music" in text


def test_the_prompt_has_persona_body_capabilities_and_style() -> None:
    persona = Personas(PERSONAS, ["jarvis"]).get("jarvis")
    prompt = system_prompt("Jarvis", persona, REACHY)
    for part in (persona.text, "no arms", "You cannot (yet)", "the setup and the punchline"):
        assert part in prompt
