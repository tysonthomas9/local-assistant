"""List pending reminders and timers. Local, no internet."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_scheduler as rs  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class ListReminders(Tool):
    """List pending reminders."""

    name = "list_reminders"
    description = "List the user's pending reminders and timers, with when each is due. Use when asked what reminders or timers are set."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return pending reminders, soonest first."""
        logger.info("Tool call: list_reminders")
        items = rs.SCHEDULER.pending()
        return {"reminders": [
            {"id": x["id"], "message": x["message"], "due": rs.spoken_time(datetime.fromisoformat(x["due"])),
             "in": rs.spoken_delta(datetime.fromisoformat(x["due"]) - rs.now())} for x in items
        ]} if items else {"reminders": [], "note": "No reminders are set."}
