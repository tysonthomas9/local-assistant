"""Cancel reminders or timers by id or by words in their message. Local, no internet."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_scheduler as rs  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class CancelReminder(Tool):
    """Cancel pending reminders."""

    name = "cancel_reminder"
    description = (
        "Cancel a pending reminder or timer. Pass words from its message (e.g. 'oven') or its id; "
        "set all to true to cancel every reminder. Call list_reminders first if unsure which one."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "which": {"type": "string", "description": "Words from the reminder's message, or its id."},
            "all": {"type": "boolean", "description": "Cancel all pending reminders."},
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Remove matching reminders."""
        which, all_ = (kwargs.get("which") or "").strip(), bool(kwargs.get("all"))
        logger.info("Tool call: cancel_reminder which=%r all=%s", which, all_)
        if not which and not all_:
            return {"error": "Which reminder? Give words from it, or all=true."}
        removed = rs.SCHEDULER.cancel(which, all_=all_)
        if not removed:
            return {"cancelled": [], "note": f"No reminder matched {which!r}."}
        return {"cancelled": [x["message"] for x in removed]}
