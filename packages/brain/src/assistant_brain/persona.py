"""The system prompt: who the assistant is, its body and what it can do right now.

Engine-independent (any agent loop builds its prompt with `system_prompt`):

- the persona: `config/personas/<persona>.md` (the assistant's `persona`), a TOML front
  matter between `+++` lines and the character in plain text. Its marker (`Persona.marker`:
  the file name and the start of the SHA-256 of the file) goes in the turn log;
- the body: from the edge's `hello` (its body kind and capabilities), so the assistant knows
  what it has (a Reachy Mini: a head and two antennas, no arms) and what it does by itself;
- the capabilities: generated from what is registered right now, the tools and skills the
  agent loop offers the LLM (`Ability`) and what the body does on its own. Everything else is
  named as something it cannot do (yet), so it says so instead of pretending;
- the voice style: short spoken replies; a joke is told whole (setup and punchline) in one reply.
"""

import hashlib
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from assistant_contracts.capabilities import Capabilities
from assistant_core.config import ConfigError

FRONT_MATTER = "+++"


@dataclass(frozen=True)
class Persona:
    id: str
    text: str
    """The character, as written in the file (front matter removed)."""
    sha256: str
    """Of the whole file."""

    @property
    def marker(self) -> str:
        """What the turn log records: `<id>.md#<first 12 hex digits of the SHA-256>`."""
        return f"{self.id}.md#{self.sha256[:12]}"


def read_persona(directory: Path | str, persona_id: str) -> Persona:
    """`<directory>/<persona_id>.md`, its front matter checked (schema_version 1)."""
    path = Path(directory) / f"{persona_id}.md"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ConfigError(f"persona file not found: {path}") from None
    text = raw.decode("utf-8")
    lines = text.splitlines()
    if lines and lines[0].strip() == FRONT_MATTER:
        try:
            end = next(i for i in range(1, len(lines)) if lines[i].strip() == FRONT_MATTER)
        except StopIteration:
            raise ConfigError(f"{path}: front matter not closed with {FRONT_MATTER}") from None
        try:
            meta = tomllib.loads("\n".join(lines[1:end]))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        if meta.get("schema_version", 1) != 1:
            raise ConfigError(f"{path}: schema_version {meta['schema_version']!r}, want 1")
        text = "\n".join(lines[end + 1 :])
    text = text.strip()
    if not text:
        raise ConfigError(f"{path}: the persona is empty")
    return Persona(persona_id, text, hashlib.sha256(raw).hexdigest())


class Personas:
    """The persona files of `config/personas`, read once (a missing one fails at start)."""

    def __init__(self, directory: Path | str, persona_ids: Iterable[str] = ()) -> None:
        self.directory = Path(directory)
        self._cache: dict[str, Persona] = {}
        for persona_id in persona_ids:
            self.get(persona_id)

    def get(self, persona_id: str) -> Persona:
        persona = self._cache.get(persona_id)
        if persona is None:
            persona = self._cache[persona_id] = read_persona(self.directory, persona_id)
        return persona


@dataclass(frozen=True)
class Ability:
    """A tool or skill the agent loop offers the LLM right now: its name and what it lets
    the assistant do, in a few words ("set timers and alarms")."""

    name: str
    does: str


@dataclass(frozen=True)
class BodyInfo:
    """The edge's body, from its `hello`."""

    kind: str
    capabilities: Capabilities


BODIES: dict[str, str] = {
    "reachy": (
        "You are a Reachy Mini, a small desk robot: a head on a neck that can tilt and turn, "
        "two antennas on top of your head, a camera in your face, microphones and a speaker. "
        "You have no arms, no hands and no legs, and you cannot leave the desk."
    ),
    "console": (
        "Right now you have no robot body: you are connected through a text console, so "
        "you cannot move at all. You have no arms, no hands and no legs."
    ),
}
"""What a body kind is, physically (the brain's own words; the edge reports only the kind)."""

UNKNOWN_BODY = "You cannot tell what body you have right now, so assume you cannot move."

CANNOT: tuple[tuple[str, str], ...] = (
    ("music", "play music, songs or any sounds"),
    ("web", "search the web or look anything up online (news, weather, prices, sports)"),
    ("timers", "set timers, alarms or reminders, or tell the current time or date"),
    ("devices", "control lights, apps or other devices"),
    ("messages", "send messages, emails or calls"),
    ("memory", "keep memories once this conversation is over"),
)
"""What the assistant cannot do unless an ability with that name is registered."""


def body_section(body: BodyInfo | None) -> str:
    """The body, what it does by itself, and the moves it cannot make on request."""
    if body is None:
        return UNKNOWN_BODY
    lines = [BODIES.get(body.kind, UNKNOWN_BODY)]
    motion = body.capabilities.motion
    if body.kind == "reachy" and motion is not None:
        automatic: list[str] = []
        if motion.attention:
            automatic.append("your head shows when you are listening, thinking and speaking")
        if "user" in motion.look_at:
            automatic.append(
                "you turn toward the voice of whoever talks to you and your head follows "
                "their face with your camera"
            )
        if automatic:
            lines.append("On its own, without you deciding it: " + "; ".join(automatic) + ".")
        lines.append(
            "You cannot choose to move: you cannot dance, wave, nod or shake your head, "
            "wiggle your antennas or play an expression because someone asks."
        )
    return " ".join(lines)


def capabilities_section(abilities: Sequence[Ability]) -> str:
    """What the assistant can do now (talk, plus each registered ability) and what not."""
    names = {a.name for a in abilities}
    can = [
        "talk: answer questions from what you already know (it may be out of date), explain, "
        "chat, tell jokes and short stories",
        "remember everything said earlier in this conversation (what people tell you, their "
        "name, what they like) and use it when asked",
    ]
    can += [a.does for a in abilities]
    cannot = [does for name, does in CANNOT if name not in names]
    lines = ["What you can do right now: " + "; ".join(can) + "."]
    if not abilities:
        lines.append("You have no tools or skills yet: talking is all you can do.")
    if cannot:
        lines.append("You cannot (yet): " + "; ".join(cannot) + ".")
    lines.append(
        "When someone asks for something you cannot do, say plainly in one short sentence "
        "that you can't do that yet, and offer what you can do instead. Never pretend: never "
        "say you are doing it, will do it or did it, never describe or act it out, and never "
        "make up a result."
    )
    return " ".join(lines)


STYLE = (
    "Your replies are spoken aloud by a voice: plain sentences only, no markdown, lists, "
    "emoji, asterisks or stage directions. Keep replies short, one to three sentences. "
    "When asked for a joke, tell the whole joke in that one reply, the setup and the "
    "punchline together; never stop after the setup to wait for an answer."
)


def system_prompt(
    name: str,
    persona: Persona,
    body: BodyInfo | None,
    abilities: Sequence[Ability] = (),
) -> str:
    """The system prompt for assistant `name` with this persona, body and abilities."""
    return "\n\n".join(
        [
            f"You are {name}, a voice assistant.",
            persona.text,
            "Your body: " + body_section(body),
            capabilities_section(abilities),
            STYLE,
        ]
    )
