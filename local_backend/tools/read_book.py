"""Read a book aloud: start, continue, pause/stop, jump to a chapter, list books, status, download.

See local_backend/reachy_reader.py. Reading happens in the background (the tool returns at once);
any interruption pauses it with a bookmark. `download` needs the internet (web profile).
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reachy_bridge  # noqa: E402
import reachy_reader as rr  # noqa: E402

from reachy_mini_conversation_app.config import config  # noqa: E402  (the settings object, not the module)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies  # noqa: E402


logger = logging.getLogger(__name__)


class ReadBook(Tool):
    """Read books aloud."""

    name = "read_book"
    description = (
        "Read a book aloud, word for word, in the background. action: start (title; optional chapter number; "
        "resumes from the bookmark unless from_start), continue (the last book), stop (pause, keeps the bookmark), "
        "chapter (jump to chapter N of the current book), list (books on this robot), status, download (fetch a "
        "public-domain book from Project Gutenberg; web only). Before starting, say one short line only. The user "
        "can only be heard in the short pauses between passages."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "continue", "stop", "chapter", "list", "status", "download"]},
            "title": {"type": "string", "description": "Book title (or words from it)."},
            "chapter": {"type": "integer", "description": "Chapter number, for start or chapter."},
            "from_start": {"type": "boolean", "description": "Start from the beginning instead of the bookmark."},
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch the action; reading itself runs in reachy_reader's thread."""
        action = (kwargs.get("action") or "").lower()
        title = str(kwargs.get("title") or "").strip()
        logger.info("Tool call: read_book %s %r chapter=%r", action, title, kwargs.get("chapter"))
        if action == "list":
            return {"books": [rr._title_of(p) for p in rr.library()]}
        if action == "status":
            return rr.READER.status()
        if action == "stop":
            was = await rr.READER.stop_now() if reachy_bridge.stream() is not None else rr.READER.stop()
            return {"stopped": was, **rr.READER.status()}
        if action == "download":
            if not str(config.REACHY_MINI_CUSTOM_PROFILE or "").endswith("_web"):
                return {"error": "Downloading books needs the internet (start the app with --web)."}
            if not title:
                return {"error": "Which book?"}
            try:
                path = await asyncio.to_thread(rr.gutenberg_download, title)
            except Exception as e:
                return {"error": str(e)}
            return {"downloaded": rr._title_of(path), "note": "Say you have it and ask if they want you to start reading."}

        if reachy_bridge.stream() is None:
            return {"error": "Reading needs the conversation app (run via start_conversation.sh)."}
        if action in ("start", "continue", "chapter"):
            if action == "start" or (action == "chapter" and title):
                path = rr.find(title) if title else None
                if path is None:
                    web = str(config.REACHY_MINI_CUSTOM_PROFILE or "").endswith("_web")
                    return {"error": f"No book like {title!r} on this robot.", "books": [rr._title_of(p) for p in rr.library()],
                            "hint": "Offer to download it (action=download)." if web else "Books go in local_backend/books/."}
                key = path.stem
            else:
                key = (rr.READER.book.key if rr.READER.book else None) or rr.last_book()
                if not key or not (rr.BOOKS / f"{key}.txt").exists():
                    return {"error": "No book in progress. Which book should I read?"}
                path = rr.BOOKS / f"{key}.txt"
            book = await asyncio.to_thread(rr.parse, path)
            if kwargs.get("chapter"):
                n = int(kwargs["chapter"])
                if not 1 <= n <= len(book.chapters):
                    return {"error": f"{book.title} has {len(book.chapters)} chapters."}
                pos = book.chapters[n - 1][1]
            elif kwargs.get("from_start") or rr.bookmark(key) < 0:
                pos = book.chapters[0][1] if book.chapters else 0
            else:
                pos = rr.bookmark(key)
            rr.READER.start(book, pos)
            ch = book.chunk_chapter[pos] if book.chunks else 0
            return {"reading": book.title, "from": book.chapters[ch][0] if book.chapters else "the beginning",
                    "note": "Reading starts right after your short reply. Any interruption pauses it."}
        return {"error": f"Unknown action {action!r}."}
