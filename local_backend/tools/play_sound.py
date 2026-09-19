"""Play a sound effect (alarm, timer, chime, bell, ...) on the robot's speaker. Local, no internet.

See local_backend/reachy_sounds.py for the sound library.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_sounds  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class PlaySound(Tool):
    """Play a sound effect."""

    name = "play_sound"
    description = (
        "Play a sound effect on the robot's speaker. Built-in sounds: alarm, timer, chime, bell, beep, success, "
        "error, wake_up, go_sleep, dance, confused, impatient, count (plus any custom sounds). Use repeat=0 to keep "
        "an alarm ringing until the user says stop (then call stop_sound)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "sound": {"type": "string", "description": "Sound name, e.g. 'alarm', 'timer', 'chime', 'bell'."},
            "repeat": {"type": "integer", "description": "How many times to play it (default 1; 0 = until stopped, max 60 s)."},
            "volume": {"type": "integer", "description": "Optional volume 0-100 (default 70)."},
        },
        "required": ["sound"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Start the sound and return right away (it plays in the background)."""
        sound = (kwargs.get("sound") or "").strip()
        logger.info("Tool call: play_sound %r repeat=%r volume=%r", sound, kwargs.get("repeat"), kwargs.get("volume"))
        try:
            repeat = int(kwargs.get("repeat", 1) if kwargs.get("repeat") is not None else 1)
            volume = kwargs.get("volume")
            return await asyncio.to_thread(reachy_sounds.PLAYER.play, sound, repeat=repeat,
                                           volume_pct=float(volume) if volume is not None else None)
        except (TypeError, ValueError) as e:
            return {"error": f"Bad arguments: {e}"}
