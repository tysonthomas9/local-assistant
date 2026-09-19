"""Play an internet radio station through the robot's speaker. Online (radio-browser.info + the stream).

See local_backend/reachy_radio.py.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_radio  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class PlayRadio(Tool):
    """Play internet radio."""

    name = "play_radio"
    description = (
        "Play an internet radio station or a genre through the robot's speaker, e.g. 'jazz', 'classical', "
        "'BBC Radio 1', 'KQED'. Replaces whatever is playing. Say the station name briefly once it starts."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Station name or music genre."},
            "volume": {"type": "integer", "description": "Optional volume 0-100 (default 30)."},
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Search and start a station (network + GStreamer work runs off the event loop)."""
        query = (kwargs.get("query") or "").strip()
        logger.info("Tool call: play_radio query=%r volume=%r", query, kwargs.get("volume"))
        if not query:
            return {"error": "Which station or genre?"}
        volume = kwargs.get("volume")
        try:
            return await asyncio.to_thread(reachy_radio.RADIO.play, query, float(volume) if volume is not None else None)
        except Exception as e:
            logger.warning("play_radio failed: %r", e)
            return {"error": f"Radio unavailable ({type(e).__name__})."}
