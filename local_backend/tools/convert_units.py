"""Unit conversion: length, mass, volume (incl. cooking), temperature, area, speed, data, time.

Local, no internet. Units are normalised from common spellings ("°F", "fahrenheit", "cups",
"fl oz", "kph"). Volume uses US customary measures (cup = 236.588 ml, fl oz = 29.5735 ml).
"""

import logging
import re
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# dimension -> unit -> factor to the base unit (m, kg, l, m2, m/s, byte, s)
FACTORS = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1, "km": 1000, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
               "nmi": 1852},
    "mass": {"mg": 1e-6, "g": 0.001, "kg": 1, "t": 1000, "oz": 0.028349523125, "lb": 0.45359237, "st": 6.35029318},
    "volume": {"ml": 0.001, "cl": 0.01, "dl": 0.1, "l": 1, "tsp": 0.00492892, "tbsp": 0.0147868, "floz": 0.0295735,
               "cup": 0.236588, "pt": 0.473176, "qt": 0.946353, "gal": 3.78541, "m3": 1000},
    "area": {"cm2": 1e-4, "m2": 1, "km2": 1e6, "ft2": 0.09290304, "in2": 0.00064516, "acre": 4046.8564224, "ha": 10000},
    "speed": {"m/s": 1, "km/h": 1 / 3.6, "mph": 0.44704, "kn": 0.514444, "ft/s": 0.3048},
    "data": {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "kib": 1024, "mib": 1024**2, "gib": 1024**3},
    "time": {"ms": 0.001, "s": 1, "min": 60, "h": 3600, "day": 86400, "week": 604800, "year": 31557600},
}
ALIASES = {
    "millimeter": "mm", "centimeter": "cm", "meter": "m", "metre": "m", "kilometer": "km", "inch": "in", "inches": "in",
    '"': "in", "foot": "ft", "feet": "ft", "'": "ft", "yard": "yd", "mile": "mi", "nautical mile": "nmi",
    "milligram": "mg", "gram": "g", "kilogram": "kg", "kilo": "kg", "tonne": "t", "metric ton": "t", "ounce": "oz",
    "pound": "lb", "lbs": "lb", "stone": "st",
    "milliliter": "ml", "millilitre": "ml", "centiliter": "cl", "deciliter": "dl", "liter": "l", "litre": "l",
    "teaspoon": "tsp", "tablespoon": "tbsp", "fluid ounce": "floz", "fl oz": "floz", "fl. oz": "floz", "cups": "cup",
    "pint": "pt", "quart": "qt", "gallon": "gal", "cubic meter": "m3",
    "square meter": "m2", "sq m": "m2", "square kilometer": "km2", "square foot": "ft2", "square feet": "ft2",
    "sq ft": "ft2", "square inch": "in2", "acres": "acre", "hectare": "ha",
    "kph": "km/h", "kmh": "km/h", "kilometers per hour": "km/h", "miles per hour": "mph", "knot": "kn", "knots": "kn",
    "meters per second": "m/s", "byte": "b", "kilobyte": "kb", "megabyte": "mb", "gigabyte": "gb", "terabyte": "tb",
    "millisecond": "ms", "second": "s", "sec": "s", "minute": "min", "hour": "h", "hr": "h", "days": "day",
    "weeks": "week", "years": "year",
    "c": "celsius", "°c": "celsius", "degc": "celsius", "centigrade": "celsius", "f": "fahrenheit", "°f": "fahrenheit",
    "degf": "fahrenheit", "k": "kelvin",
}
TEMPS = {"celsius", "fahrenheit", "kelvin"}


def unit(name: str) -> str:
    u = " ".join(str(name).strip().lower().replace("degrees ", "").split())
    if u in ALIASES:
        return ALIASES[u]
    if u.endswith("s") and u[:-1] in ALIASES:   # plurals: "grams", "liters"
        return ALIASES[u[:-1]]
    if u.endswith("es") and u[:-2] in ALIASES:
        return ALIASES[u[:-2]]
    return re.sub(r"\s+", "", u)


def to_celsius(v: float, u: str) -> float:
    return {"celsius": v, "fahrenheit": (v - 32) * 5 / 9, "kelvin": v - 273.15}[u]


def from_celsius(c: float, u: str) -> float:
    return {"celsius": c, "fahrenheit": c * 9 / 5 + 32, "kelvin": c + 273.15}[u]


def convert(value: float, frm: str, to: str) -> float:
    a, b = unit(frm), unit(to)
    if a in TEMPS or b in TEMPS:
        if not (a in TEMPS and b in TEMPS):
            raise ValueError(f"can't convert {frm} to {to}")
        return from_celsius(to_celsius(value, a), b)
    for dim, table in FACTORS.items():
        if a in table and b in table:
            return value * table[a] / table[b]
    known = a in {u for t in FACTORS.values() for u in t}, b in {u for t in FACTORS.values() for u in t}
    raise ValueError(f"unknown unit {frm!r}" if not known[0] else f"unknown unit {to!r}" if not known[1]
                     else f"can't convert {frm} to {to} (different kinds of quantity)")


class ConvertUnits(Tool):
    """Convert between units."""

    name = "convert_units"
    description = (
        "Convert a value between units: length, weight, volume and cooking measures (cups, tbsp, tsp, fl oz, ml), "
        "temperature (C/F/K), area, speed, data size, time. Always use this instead of converting yourself."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The amount, e.g. 3.5"},
            "from_unit": {"type": "string", "description": "e.g. 'cups', '°F', 'miles', 'lb'"},
            "to_unit": {"type": "string", "description": "e.g. 'ml', 'celsius', 'km', 'kg'"},
        },
        "required": ["value", "from_unit", "to_unit"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Convert and return a spoken-friendly result."""
        logger.info("Tool call: convert_units %r", kwargs)
        try:
            value = float(kwargs.get("value"))
            result = convert(value, kwargs.get("from_unit", ""), kwargs.get("to_unit", ""))
        except (TypeError, ValueError) as e:
            return {"error": str(e)}
        pretty = f"{result:,.4g}" if abs(result) < 1e6 else f"{result:,.0f}"
        return {"value": value, "from": kwargs.get("from_unit"), "to": kwargs.get("to_unit"), "result": result,
                "spoken": f"{value:g} {kwargs.get('from_unit')} is about {pretty} {kwargs.get('to_unit')}"}
