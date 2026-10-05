"""Router: wake word -> assistant (`config/assistants/*.toml`).

With a single edge a wake goes straight here; task S7 puts the WakeArbiter (200 ms across
edges) in front of it. A word no assistant claims (e.g. the edge's energy trigger, `energy`)
goes to the default assistant.
"""

from pathlib import Path

from assistant_contracts.messages import WakeWord
from assistant_core.config import AssistantDef, ConfigError, load_assistant


class Router:
    def __init__(self, config_dir: Path | str, default_assistant: str) -> None:
        directory = Path(config_dir) / "assistants"
        ids = sorted(p.stem for p in directory.glob("*.toml"))
        self.assistants: dict[str, AssistantDef] = {i: load_assistant(config_dir, i) for i in ids}
        if default_assistant not in self.assistants:
            raise ConfigError(
                f"brain.default_assistant {default_assistant!r} has no {directory}/"
                f"{default_assistant}.toml (found: {', '.join(ids) or 'none'})"
            )
        self.default = self.assistants[default_assistant]

    def spoken(self, word: str | None) -> str | None:
        """The spoken form of a wake word an assistant claims (by spoken form or model
        name), or None (e.g. the energy trigger's `energy`)."""
        if word:
            wanted = word.strip().lower()
            for assistant in self.assistants.values():
                for wake in assistant.wake_words:
                    if wanted in (wake.spoken.lower(), wake.model.lower()):
                        return wake.spoken
        return None

    def route(self, word: str | None) -> AssistantDef:
        """The assistant a wake word belongs to (its spoken form or model name)."""
        if word:
            wanted = word.strip().lower()
            for assistant in self.assistants.values():
                for wake in assistant.wake_words:
                    if wanted in (wake.spoken.lower(), wake.model.lower()):
                        return assistant
        return self.default

    def wake_words(self) -> list[WakeWord]:
        """Every assistant's wake words, for `welcome` (deterministic order)."""
        return [
            WakeWord(word=wake.spoken, model=wake.model, threshold=wake.threshold)
            for assistant in self.assistants.values()
            for wake in assistant.wake_words
        ]
