"""Web search tool backed by the local SearXNG instance (local_backend/start_searxng.sh).

Online tool, used only by the `local_reachy_web` profile. The query goes to SearXNG on
127.0.0.1:8888, which forwards it to public search engines from this machine.

Settings: REACHY_SEARXNG_URL (default http://127.0.0.1:8888).
"""

import asyncio
import logging
import os
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

MAX_RESULTS = 5


class WebSearch(Tool):
    """Search the web through the local SearXNG metasearch engine."""

    name = "web_search"
    description = (
        "Search the web for current or factual information and return the top results (title, snippet, "
        "source). Call this whenever the user asks to look something up, asks about recent events, prices, "
        "people or facts you are not sure of. Summarize the answer in one or two spoken sentences; never "
        "read out URLs."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for, as a short search-engine query."},
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run the query on SearXNG and return the top results."""
        query = (kwargs.get("query") or "").strip()
        logger.info("Tool call: web_search query=%r", query)
        if not query:
            return {"error": "Empty search query."}
        base = os.environ.get("REACHY_SEARXNG_URL", "http://127.0.0.1:8888").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # SearXNG's upstream engines occasionally all time out at once; one retry fixes most of those.
                for attempt in range(2):
                    r = await client.get(f"{base}/search", params={"q": query, "format": "json", "safesearch": 1})
                    r.raise_for_status()
                    data = r.json()
                    if data.get("results") or data.get("answers"):
                        break
                    if attempt == 0:
                        logger.info("web_search: no results (unresponsive: %s); retrying once", data.get("unresponsive_engines"))
                        await asyncio.sleep(1)
        except httpx.ConnectError:
            return {"error": "The search engine isn't running (start it with local_backend/start_searxng.sh)."}
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("web_search failed: %r", e)
            return {"error": f"Search failed ({type(e).__name__})."}
        results = [
            {"title": x.get("title", ""), "snippet": (x.get("content") or "")[:300], "source": x.get("url", "")}
            for x in data.get("results", [])[:MAX_RESULTS]
        ]
        out: Dict[str, Any] = {"query": query, "results": results}
        if data.get("answers"):
            out["direct_answers"] = [a if isinstance(a, str) else a.get("answer", "") for a in data["answers"][:2]]
        if not results and not out.get("direct_answers"):
            out["note"] = "No results."
        return out
