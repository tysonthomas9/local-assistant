"""Weather tool: current conditions and today's forecast from Open-Meteo (free, no API key).

Online tool, used only by the `local_reachy_web` profile. Sends the place name to
geocoding-api.open-meteo.com and its coordinates to api.open-meteo.com.

Settings: REACHY_HOME_LOCATION (used when no place is given), REACHY_WEATHER_UNITS
(fahrenheit | celsius, default fahrenheit).
"""

import logging
import os
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (https://open-meteo.com/en/docs)
WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


class GetWeather(Tool):
    """Current weather and today's forecast for a place."""

    name = "get_weather"
    description = (
        "Get the current weather and today's forecast (high, low, chance of rain) for a place. "
        "Call this whenever the user asks about the weather or temperature. Leave place empty for the "
        "user's home location. Never guess the weather without calling this."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "place": {"type": "string", "description": "City or place name, e.g. 'Paris' or 'San Jose, California'. Empty for home."},
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Look up the place, then fetch its weather."""
        place = (kwargs.get("place") or "").strip() or os.environ.get("REACHY_HOME_LOCATION", "").strip()
        logger.info("Tool call: get_weather place=%r", place)
        if not place:
            return {"error": "No place given and REACHY_HOME_LOCATION is not set; ask the user which city."}
        units = "celsius" if os.environ.get("REACHY_WEATHER_UNITS", "fahrenheit").lower().startswith("c") else "fahrenheit"
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                # Open-Meteo matches on the city name only, so try "San Jose, California" as "San Jose".
                geo = (await client.get(GEOCODE_URL, params={"name": place.split(",")[0].strip(), "count": 5})).json()
                hits = geo.get("results") or []
                if not hits:
                    return {"error": f"Couldn't find a place called {place!r}."}
                qualifier = place.split(",")[1].strip().lower() if "," in place else ""
                loc = next((h for h in hits if qualifier and qualifier in f"{h.get('admin1', '')} {h.get('country', '')}".lower()), hits[0])
                r = await client.get(FORECAST_URL, params={
                    "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto", "forecast_days": 1,
                    "temperature_unit": units, "wind_speed_unit": "mph" if units == "fahrenheit" else "kmh",
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                })
                r.raise_for_status()
                w = r.json()
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning("get_weather failed: %r", e)
            return {"error": f"Weather service unavailable ({type(e).__name__})."}
        cur, day, u = w["current"], w["daily"], "°F" if units == "fahrenheit" else "°C"
        return {
            "place": ", ".join(x for x in (loc.get("name"), loc.get("admin1"), loc.get("country")) if x),
            "now": f"{WMO.get(cur['weather_code'], 'unknown')}, {round(cur['temperature_2m'])}{u} "
                   f"(feels like {round(cur['apparent_temperature'])}{u}), wind {round(cur['wind_speed_10m'])} "
                   f"{'mph' if units == 'fahrenheit' else 'km/h'}",
            "today": f"{WMO.get(day['weather_code'][0], 'unknown')}, high {round(day['temperature_2m_max'][0])}{u}, "
                     f"low {round(day['temperature_2m_min'][0])}{u}, {day['precipitation_probability_max'][0]}% chance of rain",
            "local_time": cur.get("time"),
        }
