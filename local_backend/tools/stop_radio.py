"""Stop the internet radio."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_radio  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class StopRadio(Tool):
    """Stop the radio."""

    name = "stop_radio"
    description = "Stop the radio or music that is playing. Use when the user says stop, turn off the radio/music, or be quiet."
    needs_response = False
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stop playback if any."""
        logger.info("Tool call: stop_radio")
        stopped = await asyncio.to_thread(reachy_radio.RADIO.stop)
        return {"stopped": stopped} if stopped else {"note": "Nothing was playing."}
