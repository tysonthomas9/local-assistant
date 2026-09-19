"""Forget the saved conversation history of the current assistant (--assistants; reachy_assistants).

Each assistant's recent turns are saved and replayed into every new session, so it remembers the
conversation across switches and app restarts. This clears them. Long-term memory (remember / forget),
lists and reminders are not touched.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_assistants  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class ForgetConversation(Tool):
    """Clear the saved conversation history."""

    name = "forget_conversation"
    description = (
        "Forget the saved history of our conversation (what we talked about), when the user asks to forget or "
        "clear the conversation or start fresh. Not for single facts (use forget) or lists."
    )
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Clear the active assistant's history file."""
        logger.info("Tool call: forget_conversation")
        if not reachy_assistants.enabled():
            return {"note": "Conversation history isn't saved in this mode; it ends when the session restarts."}
        n = reachy_assistants.clear_history()
        return {"cleared_messages": n, "note": "Saved history cleared; it won't come back after a restart or switch."}
