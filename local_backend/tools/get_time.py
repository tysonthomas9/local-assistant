"""Local replacement for Pollen's hosted time tool: reads this computer's clock."""

import logging
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GetTime(Tool):
    """Report the current date and time from the local clock."""

    name = "get_time"
    description = (
        "Get the current date and time. Leave timezone empty for the user's local time, "
        "or pass an IANA name like 'Europe/Paris' or 'Asia/Tokyo' for another place."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Optional IANA timezone name. Empty means local time.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the current time in the requested timezone."""
        tz_name = (kwargs.get("timezone") or "").strip()
        logger.info("Tool call: get_time timezone=%r", tz_name)
        try:
            now = datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()
        except ZoneInfoNotFoundError:
            return {"error": f"Unknown timezone {tz_name!r}"}
        return {
            "timezone": tz_name or str(now.tzinfo),
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%A, %B %d, %Y"),
        }
