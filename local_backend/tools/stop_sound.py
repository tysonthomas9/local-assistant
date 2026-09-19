"""Stop a ringing alarm or sound effect. Local, no internet."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_sounds  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class StopSound(Tool):
    """Stop the current sound effect."""

    name = "stop_sound"
    description = "Stop an alarm, timer ring or sound effect that is playing. Use when the user says stop, snooze, or turn off the alarm."
    needs_response = False
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stop playback if any."""
        logger.info("Tool call: stop_sound")
        stopped = await asyncio.to_thread(reachy_sounds.PLAYER.stop)
        return {"stopped": stopped} if stopped else {"note": "No sound was playing."}
