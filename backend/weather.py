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


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WeatherAdvisoryBot/1.0",
    "Accept": "application/json",
}


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
    last_exc = None
    for attempt in range(2):
        try:
            resp = requests.get(_GEOCODE_URL, params=params, headers=_HEADERS, timeout=_REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Geocoding attempt %d failed for '%s': %s", attempt + 1, location_name, exc)
    else:
        raise LocationNotFoundError(
            f"Network error while geocoding '{location_name}': {last_exc}"
        )

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


def fetch_weather(lat: float, lon: float, target_hour: int | None = None) -> dict[str, Any]:
    """Fetch current or target hourly weather conditions from Open-Meteo for the given coordinates.

    If target_hour (0-23) is provided, extracts forecast values for that specific hour.
    Otherwise fetches real-time current conditions.

    Returns a flat dict whose keys match _CURRENT_FIELDS.

    Raises:
        WeatherFetchError: if the request fails or the payload is malformed.
    """
    # Open-Meteo hourly API supports a slightly different field set — only request hourly when needed
    _HOURLY_FIELDS = [
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
        "surface_pressure",
    ]

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(_CURRENT_FIELDS),
        "wind_speed_unit": "kmh",      # keep units consistent with SOP thresholds
        "timezone": "auto",
        "forecast_days": 1,
    }

    # Only request hourly data if we actually need it for a specific time
    if target_hour is not None:
        params["hourly"] = ",".join(_HOURLY_FIELDS)
        params["forecast_days"] = 2

    last_exc = None
    for attempt in range(2):
        try:
            resp = requests.get(_FORECAST_URL, params=params, headers=_HEADERS, timeout=15)
            if not resp.ok:
                logger.error(
                    "Weather API returned HTTP %d for (%.4f, %.4f): %s",
                    resp.status_code, lat, lon, resp.text[:300]
                )
                resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Weather fetch attempt %d failed for (%.4f, %.4f): %s", attempt + 1, lat, lon, exc)
    else:
        raise WeatherFetchError(
            f"Network error while fetching weather for ({lat}, {lon}): {last_exc}"
        )

    current = data.get("current", {})
    hourly = data.get("hourly", {})

    weather: dict[str, Any] = {}

    # If target_hour is requested, attempt to extract hourly forecast for that hour
    if target_hour is not None and hourly and "time" in hourly:
        times = hourly["time"]
        matched_idx = None
        for idx, t_str in enumerate(times):
            try:
                # Open-Meteo hourly time format: "2026-09-07T11:00"
                h = int(t_str.split("T")[1].split(":")[0])
                if h == target_hour:
                    matched_idx = idx
                    break
            except Exception:
                continue

        if matched_idx is not None:
            for field in _CURRENT_FIELDS:
                field_list = hourly.get(field, [])
                if matched_idx < len(field_list):
                    weather[field] = field_list[matched_idx]
                else:
                    weather[field] = current.get(field)
            logger.info(
                "Hourly weather fetched for target_hour=%d at (%.4f, %.4f): temp=%.1f°C, wind=%.1f km/h, uv=%.1f",
                target_hour,
                lat,
                lon,
                weather.get("temperature_2m", float("nan")),
                weather.get("wind_speed_10m", float("nan")),
                weather.get("uv_index", float("nan")),
            )
            return weather

    # Default to current real-time weather
    if not current or not isinstance(current, dict):
        logger.error("Unexpected weather payload structure: %s", data)
        raise WeatherFetchError("Open-Meteo returned an unexpected payload structure")

    for field in _CURRENT_FIELDS:
        val = current.get(field)
        weather[field] = val
        if val is None:
            logger.warning("Field '%s' missing in Open-Meteo response", field)

    logger.info(
        "Current weather fetched for (%.4f, %.4f): temp=%.1f°C, wind=%.1f km/h, uv=%.1f",
        lat,
        lon,
        weather.get("temperature_2m", float("nan")),
        weather.get("wind_speed_10m", float("nan")),
        weather.get("uv_index", float("nan")),
    )
    return weather
