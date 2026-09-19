"""Web search tool backed by the local SearXNG instance (local_backend/start_searxng.sh).

Online tool, used only by the `local_reachy_web` profile. The query goes to SearXNG on
127.0.0.1:8888, which forwards it to public search engines from this machine.

Pagination: `page` (1, 2, 3, ...) returns results in pages of 5. SearXNG's own pages are uneven
(e.g. 27, 39, 36 results for one query) and can repeat each other, so this tool fetches SearXNG
pages in order (up to 5), drops duplicate URLs, and keeps the merged list in a short-lived cache
(5 minutes, per query), so "more results" doesn't search again and page 2 continues exactly where
page 1 stopped. Each response says whether more results exist (`has_more`).

Settings: REACHY_SEARXNG_URL (default http://127.0.0.1:8888).
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

PAGE_SIZE = 5            # results per tool page (each result goes into the model's context)
MAX_PAGE = 10            # tool pages: up to 50 results per query
MAX_UPSTREAM_PAGES = 5   # SearXNG pages fetched per query at most
CACHE_TTL_S = 300
_CACHE: Dict[str, Dict[str, Any]] = {}


def _query_key(query: str) -> str:
    return " ".join(query.lower().split())


def _url_key(url: str) -> str:
    """Same page, different spelling: scheme, 'www.' and a trailing slash don't matter."""
    return re.sub(r"^https?://(www\.)?", "", url.strip().lower()).rstrip("/")


async def _fetch_page(client: httpx.AsyncClient, base: str, query: str, pageno: int) -> Dict[str, Any]:
    # SearXNG's upstream engines occasionally all time out at once; one retry fixes most of those.
    for attempt in range(2 if pageno == 1 else 1):
        r = await client.get(f"{base}/search", params={"q": query, "format": "json", "safesearch": 1, "pageno": pageno})
        r.raise_for_status()
        data = r.json()
        if data.get("results") or data.get("answers") or pageno > 1 or attempt == 1:
            return data
        logger.info("web_search: no results (unresponsive: %s); retrying once", data.get("unresponsive_engines"))
        await asyncio.sleep(1)
    return data


class WebSearch(Tool):
    """Search the web through the local SearXNG metasearch engine."""

    name = "web_search"
    description = (
        "Search the web for current or factual information and return results (title, snippet, source), 5 per "
        "page. Call this whenever the user asks to look something up, asks about recent events, prices, people or "
        "facts you are not sure of. Summarize the answer in one or two spoken sentences; never read out URLs. If the "
        "user wants more or other results, call it again with the same query and the next page number; has_more "
        "says whether there are more."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for, as a short search-engine query."},
            "page": {"type": "integer", "description": "Results page, 1 (default) for the top 5, 2 for results 6-10, and so on."},
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return one page of de-duplicated results, fetching more from SearXNG only as needed."""
        query = (kwargs.get("query") or "").strip()
        try:
            page = max(1, min(MAX_PAGE, int(kwargs.get("page") or 1)))
        except (TypeError, ValueError):
            page = 1
        logger.info("Tool call: web_search query=%r page=%d", query, page)
        if not query:
            return {"error": "Empty search query."}
        base = os.environ.get("REACHY_SEARXNG_URL", "http://127.0.0.1:8888").rstrip("/")

        now = time.monotonic()
        for k in [k for k, v in _CACHE.items() if now - v["t"] > CACHE_TTL_S]:
            del _CACHE[k]
        key = _query_key(query)
        entry = _CACHE.get(key) or {"results": [], "seen": set(), "answers": [], "next": 1, "done": False, "t": now}

        need = page * PAGE_SIZE + 1  # one extra, to know whether there is a next page
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                while len(entry["results"]) < need and not entry["done"]:
                    pageno = entry["next"]
                    try:
                        data = await _fetch_page(client, base, query, pageno)
                    except (httpx.HTTPError, ValueError):
                        if pageno == 1:
                            raise
                        logger.warning("web_search: SearXNG page %d failed; returning what we have", pageno)
                        break  # not marked done: a later call may fetch it
                    entry["next"] += 1
                    if pageno == 1:
                        entry["answers"] = [a if isinstance(a, str) else a.get("answer", "") for a in data.get("answers", [])[:2]]
                    fresh = 0
                    for x in data.get("results", []):
                        u = _url_key(x.get("url", ""))
                        if u and u not in entry["seen"]:
                            entry["seen"].add(u)
                            entry["results"].append({"title": x.get("title", ""), "snippet": (x.get("content") or "")[:300],
                                                     "source": x.get("url", "")})
                            fresh += 1
                    if not data.get("results") or fresh == 0 or entry["next"] > MAX_UPSTREAM_PAGES:
                        entry["done"] = True
        except httpx.ConnectError:
            return {"error": "The search engine isn't running (start it with local_backend/start_searxng.sh)."}
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("web_search failed: %r", e)
            return {"error": f"Search failed ({type(e).__name__})."}
        if entry["results"] or entry["done"]:
            _CACHE[key] = entry

        start = (page - 1) * PAGE_SIZE
        results = entry["results"][start:start + PAGE_SIZE]
        out: Dict[str, Any] = {"query": query, "page": page, "results": results,
                               "has_more": len(entry["results"]) > start + PAGE_SIZE}
        if page == 1 and entry["answers"]:
            out["direct_answers"] = entry["answers"]
        if not results and not out.get("direct_answers"):
            out["note"] = "No results." if page == 1 else "No more results."
        return out
