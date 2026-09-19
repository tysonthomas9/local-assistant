"""Currency conversion with European Central Bank reference rates via frankfurter.dev (free, no key).

Online tool, web profile only. Rates are daily (ECB publishes once per working day), cached for a
day per base currency in local_backend/state/currency_rates.json.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

API = "https://api.frankfurter.dev/v1/latest"
CACHE = Path(__file__).resolve().parents[1] / "state" / "currency_rates.json"
TTL_S = 24 * 3600
NAMES = {
    "dollar": "USD", "dollars": "USD", "us dollar": "USD", "usd": "USD", "$": "USD", "buck": "USD", "bucks": "USD",
    "euro": "EUR", "euros": "EUR", "€": "EUR", "pound": "GBP", "pounds": "GBP", "sterling": "GBP", "£": "GBP",
    "yen": "JPY", "¥": "JPY", "rupee": "INR", "rupees": "INR", "₹": "INR", "yuan": "CNY", "renminbi": "CNY",
    "canadian dollar": "CAD", "canadian dollars": "CAD", "australian dollar": "AUD", "australian dollars": "AUD",
    "swiss franc": "CHF", "swiss francs": "CHF", "franc": "CHF", "won": "KRW", "peso": "MXN", "pesos": "MXN",
    "mexican peso": "MXN", "real": "BRL", "reais": "BRL", "krona": "SEK", "kronor": "SEK", "krone": "NOK",
    "zloty": "PLN", "baht": "THB", "rand": "ZAR", "lira": "TRY", "singapore dollar": "SGD", "hong kong dollar": "HKD",
    "new zealand dollar": "NZD",
}


def code(name: str) -> str:
    n = " ".join(str(name).strip().lower().split())
    return NAMES.get(n, n.upper())


_CACHE_LOCK = threading.Lock()


def _rates(base: str) -> Dict[str, Any]:
    with _CACHE_LOCK:
        try:
            cache = json.loads(CACHE.read_text())
        except (OSError, ValueError):
            cache = {}
        hit = cache.get(base)
        if hit and time.time() - hit.get("fetched", 0) < TTL_S and isinstance(hit.get("rates"), dict):
            return hit
        r = httpx.get(API, params={"base": base}, timeout=8)
        if r.status_code == 404:
            raise ValueError(f"unknown currency {base!r}")
        r.raise_for_status()
        data = r.json()
        if not isinstance(data.get("rates"), dict):
            raise ValueError("unexpected answer from the currency service")
        cache[base] = {"date": data.get("date", ""), "rates": data["rates"], "fetched": time.time()}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(CACHE)
        return cache[base]


class ConvertCurrency(Tool):
    """Convert money between currencies."""

    name = "convert_currency"
    description = (
        "Convert an amount of money between currencies (e.g. 50 euros to dollars) using today's European Central "
        "Bank reference rates. Use currency names or codes (USD, EUR, GBP, JPY, INR, ...)."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "e.g. 'EUR' or 'euros'"},
            "to_currency": {"type": "string", "description": "e.g. 'USD' or 'dollars'"},
        },
        "required": ["amount", "from_currency", "to_currency"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Look up the rate (cached for a day) and convert."""
        logger.info("Tool call: convert_currency %r", kwargs)
        try:
            amount = float(kwargs.get("amount"))
            a, b = code(kwargs.get("from_currency", "")), code(kwargs.get("to_currency", ""))
            if a == b:
                return {"result": amount, "spoken": f"{amount:,.2f} {a}"}
            import asyncio
            data = await asyncio.to_thread(_rates, a)
            if b not in data["rates"]:
                return {"error": f"No rate from {a} to {b!r}."}
        except (TypeError, ValueError) as e:
            return {"error": str(e)}
        except httpx.HTTPError as e:
            return {"error": f"Currency service unavailable ({type(e).__name__})."}
        result = amount * data["rates"][b]
        return {"amount": amount, "from": a, "to": b, "rate": data["rates"][b], "rate_date": data["date"],
                "result": round(result, 2), "spoken": f"{amount:,.2f} {a} is about {result:,.2f} {b}"}
