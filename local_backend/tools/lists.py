"""Shopping list, to-do list, notes and any other named list. Local, saved across restarts.

See local_backend/reachy_lists.py.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_lists as rl  # noqa: E402

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class Lists(Tool):
    """Manage named lists."""

    name = "lists"
    description = (
        "Manage the user's lists: shopping, todo, notes, or any other name. action: add (items), remove (words from "
        "an item), read, clear (only after the user confirms), list_lists. Use this, not remember, for list items."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "remove", "read", "clear", "list_lists"]},
            "list_name": {"type": "string", "description": "e.g. 'shopping', 'todo', 'notes', 'packing'. Default 'notes'."},
            "items": {"type": "array", "items": {"type": "string"}, "description": "For add: one or more items."},
            "item": {"type": "string", "description": "For remove: words from the item to remove."},
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch the action."""
        action = (kwargs.get("action") or "").strip().lower()
        name = kwargs.get("list_name") or "notes"
        logger.info("Tool call: lists %s %r %r", action, name, kwargs.get("items") or kwargs.get("item"))
        if action == "add":
            items = kwargs.get("items") or ([kwargs["item"]] if kwargs.get("item") else [])
            if isinstance(items, str):
                items = [items]
            if not items:
                return {"error": "What should I add?"}
            key, added, dup = rl.add(name, [str(i) for i in items])
            out: Dict[str, Any] = {"list": key, "added": added, "count": len(rl.read(key)[1])}
            if dup:
                out["already_on_list"] = dup
            return out
        if action == "remove":
            key, gone = rl.remove(name, str(kwargs.get("item") or (kwargs.get("items") or [""])[0]))
            return {"list": key, "removed": gone} if gone else {"list": key, "note": "Nothing matched."}
        if action == "read":
            key, items = rl.read(name)
            return {"list": key, "items": items} if items else {"list": key, "items": [], "note": "The list is empty."}
        if action == "clear":
            key, n = rl.clear(name)
            return {"list": key, "cleared": n}
        if action == "list_lists":
            return {"lists": rl.lists()}
        return {"error": f"Unknown action {action!r}."}
