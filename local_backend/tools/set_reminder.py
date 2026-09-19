"""Set a reminder or timer; the robot says it out loud when it's due. Local, no internet.

See local_backend/reachy_scheduler.py for how reminders are stored and delivered.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # local_backend/, for reachy_scheduler
import reachy_scheduler as rs  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class SetReminder(Tool):
    """Schedule a spoken reminder or timer."""

    name = "set_reminder"
    description = (
        "Set a reminder or a timer. The robot will say the message out loud when it is due. Use in_minutes for "
        "relative times ('in 10 minutes', 'a 5 minute timer') or at for a clock time ('at 5:30 pm', 'at 17:30'). "
        "For a plain timer, use a message like 'Your 5 minute timer is done'. When due, the robot first plays "
        "the sound: 'timer' for timers, 'alarm' for alarms or wake-ups (rings until the user says stop), "
        "'chime' for ordinary reminders (default), or 'none'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "What to remind the user of, e.g. 'take the pizza out of the oven'."},
            "in_minutes": {"type": "number", "description": "Minutes from now. Use this OR at."},
            "at": {"type": "string", "description": "Clock time today (or tomorrow if already past), e.g. '17:30' or '5:30 pm'. Use this OR in_minutes."},
            "sound": {"type": "string", "description": "Sound to play when due: 'chime' (default), 'timer', 'alarm', 'bell', or 'none'."},
        },
        "required": ["message"],
    }

    def __init__(self) -> None:
        super().__init__()
        rs.SCHEDULER.ensure_started()  # fire reminders left over from before a restart

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Validate the time, store the reminder and report when it will fire."""
        message = (kwargs.get("message") or "").strip()
        logger.info("Tool call: set_reminder %r in_minutes=%r at=%r", message, kwargs.get("in_minutes"), kwargs.get("at"))
        if not message:
            return {"error": "What should I remind you about?"}
        try:
            due = rs.parse_due(kwargs.get("in_minutes"), kwargs.get("at"))
        except ValueError as e:
            return {"error": str(e)}
        item = rs.SCHEDULER.add(message, due, sound=kwargs.get("sound") or "chime")
        out: Dict[str, Any] = {"scheduled": True, "id": item["id"], "message": message, "sound": item["sound"],
                               "due": rs.spoken_time(due), "in": rs.spoken_delta(due - rs.now())}
        if not await rs.rpc_reachable():
            out["warning"] = "The app's control channel isn't reachable (start it with --ui), so the reminder can't be spoken yet."
        return out
