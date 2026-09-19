"""Privacy mute: stop listening (for a while or until resumed), resume, or report status. Local.

See local_backend/reachy_listening.py.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_listening as rl  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class Listening(Tool):
    """Mute or unmute the microphone."""

    name = "listening"
    description = (
        "Privacy mute. action=stop: stop listening (microphone off) for `minutes` (default 60; 0 = until resumed). "
        "action=resume: listen again. action=status: am I listening? While muted you can still talk, but can't hear "
        "the user until the timer ends, they unmute in the app, or they say the wake word. hard=true also disables "
        "the wake word. Confirm briefly, including when you'll listen again."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["stop", "resume", "status"]},
            "minutes": {"type": "number", "description": "For stop: how long (default 60, 0 = until resumed)."},
            "hard": {"type": "boolean", "description": "For stop: also ignore the wake word."},
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Apply the mute action."""
        action = (kwargs.get("action") or "").lower()
        logger.info("Tool call: listening %s minutes=%r hard=%r", action, kwargs.get("minutes"), kwargs.get("hard"))
        mm = getattr(deps, "movement_manager", None)
        if action == "stop":
            try:
                minutes = 60.0 if kwargs.get("minutes") is None else float(kwargs["minutes"])
            except (TypeError, ValueError):
                return {"error": "minutes must be a number"}
            if minutes < 0 or minutes > 24 * 60:
                return {"error": "minutes must be between 0 and 1440"}
            return rl.mute(minutes, bool(kwargs.get("hard")), mm)
        if action == "resume":
            return rl.resume(mm)
        if action == "status":
            return rl.status()
        return {"error": f"Unknown action {action!r}."}
