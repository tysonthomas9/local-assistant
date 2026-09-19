"""Switch the robot's persona (character + voice), e.g. "be a Victorian butler", "go back to normal".

Personas are the generated local_<name>[_web] profiles (see local_backend/make_personas.py); the
current offline/web mode is kept. The switch restarts the conversation session (history is lost,
lists and reminders are kept), so the tool returns immediately and a background thread applies it
after the robot has spoken its confirmation. Applying it inside the tool would restart the backend
while this tool's result is still pending, and the confirmation would never be spoken.
"""

import logging
import re
import sys
import threading
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_bridge  # noqa: E402

from reachy_mini_conversation_app import config  # noqa: E402
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
NORMAL = {"normal", "default", "yourself", "reachy", "regular", "usual", "back", "original", "none"}
SYNONYMS = {  # words people use -> persona name
    "butler": "victorian_butler", "victorian": "victorian_butler", "jeeves": "victorian_butler",
    "detective": "noir_detective", "noir": "noir_detective", "private eye": "noir_detective",
    "mars": "mars_rover", "rover": "mars_rover", "astronaut": "mars_rover",
    "teenager": "bored_teenager", "teen": "bored_teenager", "bored": "bored_teenager",
    "chess": "chess_coach", "coach": "chess_coach",
    "chef": "cosmic_kitchen", "cook": "cosmic_kitchen", "kitchen": "cosmic_kitchen",
    "hype": "hype_bot", "cheerleader": "hype_bot", "motivator": "hype_bot",
    "scientist": "mad_scientist_assistant", "igor": "mad_scientist_assistant", "lab": "mad_scientist_assistant",
    "nature": "nature_documentarian", "documentary": "nature_documentarian", "attenborough": "nature_documentarian",
    "narrator": "nature_documentarian", "time traveler": "time_traveler", "time traveller": "time_traveler",
    "captain": "captain_circuit", "circuit": "captain_circuit", "superhero": "captain_circuit",
    "bro": "sorry_bro", "dude": "sorry_bro",
}
SWITCH_WAIT_S = 8


def available() -> list[str]:
    return sorted(p.name[len("local_"):] for p in PROFILES.glob("local_*")
                  if not p.name.endswith("_web") and p.name not in ("local_reachy",))


def match(request: str) -> str | None:
    """Persona name for free text; "" means the normal robot; None if nothing fits."""
    text = " ".join(re.sub(r"[^a-z ]", " ", (request or "").lower()).split())
    names = available()
    if not text or set(text.split()) & NORMAL:
        return ""
    key = text.replace(" ", "_")
    if key in names:
        return key
    for word, name in sorted(SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", text) and name in names:
            return name
    close = get_close_matches(key, names, n=1, cutoff=0.6)
    return close[0] if close else None


def _apply_later(profile: str) -> None:
    """After the confirmation has been spoken, restart the session with the new profile."""
    reachy_bridge.wait_for({"assistant_transcript_done"}, SWITCH_WAIT_S)
    try:
        stream = reachy_bridge.stream()
        reachy_bridge.run_in_app_loop(lambda: stream.apply_personality(profile), timeout=30)
        logger.info("Switched persona to %s", profile)
    except Exception:
        logger.exception("Persona switch to %s failed", profile)


class SwitchPersona(Tool):
    """Change character and voice."""

    name = "switch_persona"
    description = (
        "Switch your character and voice, or go back to normal. Personas: " + ", ".join(available()) + ". "
        "Accepts loose names ('butler', 'detective', 'chef', 'normal'). Tell the user the conversation restarts "
        "in the new character. persona='list' returns the options."
    )
    parameters_schema = {
        "type": "object",
        "properties": {"persona": {"type": "string", "description": "Persona name or description, 'normal', or 'list'."}},
        "required": ["persona"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Resolve the persona and schedule the switch."""
        request = str(kwargs.get("persona") or "")
        logger.info("Tool call: switch_persona %r", request)
        if request.strip().lower() in ("list", "options", "which", "?"):
            return {"personas": available()}
        name = match(request)
        if name is None:
            return {"error": f"No persona like {request!r}.", "personas": available()}
        web = str(config.REACHY_MINI_CUSTOM_PROFILE or "").endswith("_web")
        profile = ("local_reachy" if name == "" else f"local_{name}") + ("_web" if web else "")
        if profile == config.REACHY_MINI_CUSTOM_PROFILE:
            return {"note": "Already in that persona."}
        if reachy_bridge.stream() is None:
            return {"error": "Persona switching needs the conversation app (run via start_conversation.sh)."}
        threading.Thread(target=_apply_later, args=(profile,), daemon=True, name="switch-persona").start()
        return {"switching_to": name or "normal", "profile": profile,
                "note": "Say one short goodbye line; the conversation then restarts in the new character."}
