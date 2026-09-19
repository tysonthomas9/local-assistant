"""Tech news tool: latest headlines from RSS/Atom feeds (no API key).

Online tool, used only by the `local_reachy_web` profile. Fetches the feeds listed in
local_backend/tech_news_feeds.json (Hacker News, Ars Technica, The Verge by default).
"""

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

FEEDS_FILE = Path(__file__).resolve().parents[1] / "tech_news_feeds.json"
ATOM = "{http://www.w3.org/2005/Atom}"


def _parse(xml_text: str, source: str, limit: int) -> List[Dict[str, str]]:
    """Return up to `limit` {title, source} items from an RSS 2.0 or Atom document."""
    root = ET.fromstring(xml_text)
    items = root.findall("./channel/item") or root.findall(f"{ATOM}entry")
    out = []
    for it in items[:limit]:
        title = (it.findtext("title") or it.findtext(f"{ATOM}title") or "").strip()
        if title:
            out.append({"title": " ".join(title.split()), "source": source})
    return out


class TechNews(Tool):
    """Latest tech headlines from a few RSS feeds."""

    name = "tech_news"
    description = (
        "Get the latest technology news headlines from Hacker News, Ars Technica and The Verge. Call this "
        "when the user asks for tech news, what's new in tech, or headlines from one of those sites. Read out "
        "only the few most interesting headlines, briefly."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Optional: 'Hacker News', 'Ars Technica' or 'The Verge'. Empty for all."},
            "count": {"type": "integer", "description": "Headlines per source (1-10, default 5)."},
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Fetch the configured feeds concurrently and return their newest headlines."""
        wanted = (kwargs.get("source") or "").strip().lower()
        try:
            count = max(1, min(10, int(kwargs.get("count") or 5)))
        except (TypeError, ValueError):
            count = 5
        logger.info("Tool call: tech_news source=%r count=%d", wanted, count)
        feeds = json.loads(FEEDS_FILE.read_text())
        if wanted:
            feeds = [f for f in feeds if wanted in f["name"].lower()] or feeds

        async with httpx.AsyncClient(timeout=8, follow_redirects=True,
                                     headers={"User-Agent": "reachy-mini-tech-news/1.0"}) as client:
            async def fetch(feed: Dict[str, str]) -> List[Dict[str, str]]:
                r = await client.get(feed["url"])
                r.raise_for_status()
                return _parse(r.text, feed["name"], count)

            results = await asyncio.gather(*(fetch(f) for f in feeds), return_exceptions=True)

        headlines, failed = [], []
        for feed, res in zip(feeds, results):
            if isinstance(res, Exception):
                logger.warning("tech_news: %s failed: %r", feed["name"], res)
                failed.append(feed["name"])
            else:
                headlines.extend(res)
        if not headlines:
            return {"error": "Couldn't fetch any news feeds right now."}
        out: Dict[str, Any] = {"headlines": headlines}
        if failed:
            out["unavailable"] = failed
        return out
