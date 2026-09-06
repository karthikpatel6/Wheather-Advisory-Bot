"""
weather.py — Open-Meteo geocoding and weather forecast calls.

Two public functions:
    geocode(location_name)  → {"lat": float, "lon": float, "display_name": str}
    fetch_weather(lat, lon) → flat dict of current weather fields

Both raise typed exceptions on failure so the graph can route cleanly
to honest_failure without try/except at the call site.

Open-Meteo is free and requires no API key.
All field names in the returned weather dict exactly match the Open-Meteo
response field names so they can be used directly in SOP condition.field
checks and placeholder substitution.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Typed exceptions
# --------------------------------------------------------------------------


class LocationNotFoundError(Exception):
    """Raised when geocoding finds no results for the requested location name."""


class WeatherFetchError(Exception):
    """Raised when the Open-Meteo forecast API call fails or returns bad data."""


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Exact field names from Open-Meteo's `current` parameter list.
# These names are used in SOP condition.field and in {{placeholder}} substitution.
_CURRENT_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "uv_index",
    "visibility",
    "surface_pressure",
]

_REQUEST_TIMEOUT_S = 10  # seconds


# --------------------------------------------------------------------------
# Geocoding
# --------------------------------------------------------------------------


def geocode(location_name: str) -> dict[str, Any]:
    """Resolve a free-text location name to lat/lon via Open-Meteo geocoding.

    Returns:
        {"lat": float, "lon": float, "display_name": str}

    Raises:
        LocationNotFoundError: if no results are returned or request fails.
    """
    params = {
        "name": location_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    try:
        resp = requests.get(_GEOCODE_URL, params=params, timeout=_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Geocoding request failed for '%s': %s", location_name, exc)
        raise LocationNotFoundError(
            f"Network error while geocoding '{location_name}'"
        ) from exc

    results = data.get("results")
    if not results:
        logger.info("No geocoding results for '%s'", location_name)
        raise LocationNotFoundError(
            f"No location found for '{location_name}'"
        )

    top = results[0]
    lat = top.get("latitude")
    lon = top.get("longitude")
    if lat is None or lon is None:
        raise LocationNotFoundError(
            f"Geocoding returned incomplete data for '{location_name}'"
        )

    # Build a human-readable display name from whatever fields are available
    parts = [top.get("name", location_name)]
    if top.get("admin1"):
        parts.append(top["admin1"])
    if top.get("country"):
        parts.append(top["country"])
    display_name = ", ".join(parts)

    logger.info("Geocoded '%s' → %s (%.4f, %.4f)", location_name, display_name, lat, lon)
    return {"lat": lat, "lon": lon, "display_name": display_name}


# --------------------------------------------------------------------------
# Weather forecast
# --------------------------------------------------------------------------


def fetch_weather(lat: float, lon: float) -> dict[str, Any]:
    """Fetch current weather conditions from Open-Meteo for the given coordinates.

    Returns a flat dict whose keys are exactly the Open-Meteo field names
    (matching _CURRENT_FIELDS above). All values are the raw API values
    (floats, ints, or None if the field was missing).

    Raises:
        WeatherFetchError: if the request fails or the payload is malformed.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(_CURRENT_FIELDS),
        "wind_speed_unit": "kmh",      # keep units consistent with SOP thresholds
        "timezone": "auto",
        "forecast_days": 1,
    }
    try:
        resp = requests.get(_FORECAST_URL, params=params, timeout=_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Weather fetch failed for (%.4f, %.4f): %s", lat, lon, exc)
        raise WeatherFetchError(
            f"Network error while fetching weather for ({lat}, {lon})"
        ) from exc

    current = data.get("current")
    if not current or not isinstance(current, dict):
        logger.error("Unexpected weather payload structure: %s", data)
        raise WeatherFetchError("Open-Meteo returned an unexpected payload structure")

    # Flatten: extract just the field values (drop 'time', 'interval' metadata)
    weather: dict[str, Any] = {}
    for field in _CURRENT_FIELDS:
        val = current.get(field)
        weather[field] = val
        if val is None:
            logger.warning("Field '%s' missing in Open-Meteo response", field)

    logger.info(
        "Weather fetched for (%.4f, %.4f): temp=%.1f°C, wind=%.1f km/h, uv=%.1f",
        lat,
        lon,
        weather.get("temperature_2m", float("nan")),
        weather.get("wind_speed_10m", float("nan")),
        weather.get("uv_index", float("nan")),
    )
    return weather
