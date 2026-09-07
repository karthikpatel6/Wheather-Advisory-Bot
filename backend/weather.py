"""
weather.py — Open-Meteo geocoding and weather forecast calls.

Two public functions:
    geocode(location_name)  → {"lat": float, "lon": float, "display_name": str}
    fetch_weather(lat, lon) → flat dict of current weather fields

Both raise typed exceptions on failure so the graph can route cleanly
to honest_failure without try/except at the call site.

Open-Meteo is free and requires no API key by default.
All field names in the returned weather dict exactly match the Open-Meteo
response field names so they can be used directly in SOP condition.field
checks and placeholder substitution.
"""

from __future__ import annotations

import logging
import os
import time
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
# Constants & Caches
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

# Global in-memory TTL caches across all sessions
_GEOCODE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_GEOCODE_TTL_S = 3600  # 1 hour

_WEATHER_CACHE: dict[tuple[float, float, int | None], tuple[float, dict[str, Any]]] = {}
_WEATHER_TTL_S = 900  # 15 minutes fresh TTL
_WEATHER_STALE_TTL_S = 86400  # 24 hours stale fallback TTL on API failure / 429


# --------------------------------------------------------------------------
# Geocoding
# --------------------------------------------------------------------------


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WeatherAdvisoryBot/1.0",
    "Accept": "application/json",
}


def _geocode_openweathermap(location_name: str, api_key: str) -> dict[str, Any]:
    """Geocode using OpenWeatherMap Direct Geocoding API."""
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": location_name, "limit": 1, "appid": api_key}
    resp = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    results = resp.json()
    if not results or not isinstance(results, list):
        raise LocationNotFoundError(f"OpenWeatherMap geocoding found no results for '{location_name}'")
    top = results[0]
    lat = top.get("lat")
    lon = top.get("lon")
    if lat is None or lon is None:
        raise LocationNotFoundError(f"OpenWeatherMap geocoding returned incomplete data for '{location_name}'")
    
    parts = [top.get("name", location_name)]
    if top.get("state"):
        parts.append(top["state"])
    if top.get("country"):
        parts.append(top["country"])
    display_name = ", ".join(parts)
    logger.info("OpenWeatherMap geocoded '%s' → %s (%.4f, %.4f)", location_name, display_name, lat, lon)
    return {"lat": float(lat), "lon": float(lon), "display_name": display_name}


def geocode(location_name: str) -> dict[str, Any]:
    """Resolve a free-text location name to lat/lon via OpenWeatherMap (if key present) or Open-Meteo geocoding.

    Returns:
        {"lat": float, "lon": float, "display_name": str}

    Raises:
        LocationNotFoundError: if no results are returned or request fails.
    """
    key = location_name.strip().lower()
    now = time.time()

    # Check global in-memory cache
    if key in _GEOCODE_CACHE:
        ts, cached_loc = _GEOCODE_CACHE[key]
        if now - ts < _GEOCODE_TTL_S:
            logger.info("Geocode cache hit for '%s' → %s", location_name, cached_loc.get("display_name"))
            return cached_loc

    # Priority 1: OpenWeatherMap Geocoding API if key is present
    owm_key = os.getenv("OPENWEATHERMAP_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
    if owm_key and not owm_key.startswith("your_"):
        try:
            res = _geocode_openweathermap(location_name, owm_key)
            _GEOCODE_CACHE[key] = (now, res)
            return res
        except Exception as exc:
            logger.warning("OpenWeatherMap geocoding failed for '%s' (%s); falling back to Open-Meteo", location_name, exc)

    # Priority 2: Open-Meteo Geocoding API
    params = {
        "name": location_name,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    last_exc = None
    for attempt in range(2):
        if attempt > 0:
            time.sleep(1)
        try:
            resp = requests.get(_GEOCODE_URL, params=params, headers=_HEADERS, timeout=_REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Geocoding attempt %d failed for '%s': %s", attempt + 1, location_name, exc)
    else:
        # Check if stale cached item exists before giving up
        if key in _GEOCODE_CACHE:
            _, cached_loc = _GEOCODE_CACHE[key]
            logger.warning("Geocoding failed for '%s'; serving cached fallback result", location_name)
            return cached_loc
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

    res = {"lat": lat, "lon": lon, "display_name": display_name}
    _GEOCODE_CACHE[key] = (now, res)
    logger.info("Geocoded '%s' → %s (%.4f, %.4f)", location_name, display_name, lat, lon)
    return res


# --------------------------------------------------------------------------
# Weather forecast
# --------------------------------------------------------------------------


def _fetch_weather_openweathermap(lat: float, lon: float, api_key: str, target_hour: int | None = None) -> dict[str, Any]:
    """Fetch current weather from OpenWeatherMap 2.5 API and map fields to standard schema."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",  # Temp in °C, speed in m/s
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    main = data.get("main", {})
    wind = data.get("wind", {})
    clouds = data.get("clouds", {})
    rain_dict = data.get("rain", {})
    snow_dict = data.get("snow", {})
    weather_list = data.get("weather", [{}])
    weather_first = weather_list[0] if weather_list else {}

    temp_c = float(main.get("temp", 20.0))
    feels_c = float(main.get("feels_like", temp_c))
    humidity = float(main.get("humidity", 50.0))
    pressure = float(main.get("pressure", 1013.0))

    # Convert wind speed from m/s to km/h (1 m/s = 3.6 km/h)
    wind_ms = float(wind.get("speed", 0.0))
    wind_kmh = round(wind_ms * 3.6, 1)
    gust_ms = float(wind.get("gust", wind_ms * 1.25))
    gust_kmh = round(gust_ms * 3.6, 1)

    rain_1h = float(rain_dict.get("1h", rain_dict.get("3h", 0.0)))
    snow_1h = float(snow_dict.get("1h", snow_dict.get("3h", 0.0)))
    precip_total = round(rain_1h + snow_1h, 1)

    cloud_pct = float(clouds.get("all", 0.0))
    vis_meters = float(data.get("visibility", 10000))
    weather_code = int(weather_first.get("id", 800))

    weather = {
        "temperature_2m": temp_c,
        "apparent_temperature": feels_c,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": wind_kmh,
        "wind_gusts_10m": gust_kmh,
        "precipitation": precip_total,
        "rain": rain_1h,
        "showers": 0.0,
        "snowfall": snow_1h,
        "weather_code": weather_code,
        "cloud_cover": cloud_pct,
        "uv_index": 0.0,  # OpenWeatherMap basic endpoint default
        "visibility": vis_meters,
        "surface_pressure": pressure,
    }
    logger.info(
        "OpenWeatherMap weather fetched for (%.4f, %.4f): temp=%.1f°C, wind=%.1f km/h",
        lat, lon, temp_c, wind_kmh
    )
    return weather


def _fetch_weather_wttr(lat: float, lon: float, target_hour: int | None = None) -> dict[str, Any]:
    """Fallback weather fetcher using wttr.in JSON API when Open-Meteo returns HTTP 429."""
    url = f"https://wttr.in/{lat:.4f},{lon:.4f}?format=j1"
    headers = {"User-Agent": "curl/7.68.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    current_list = data.get("current_condition", [])
    if not current_list:
        raise WeatherFetchError("wttr.in returned no current_condition")

    cur = current_list[0]
    precip_mm = float(cur.get("precipMM", 0.0))
    wind_kmh = float(cur.get("windspeedKmph", 0.0))
    temp_c = float(cur.get("temp_C", 20.0))
    feels_c = float(cur.get("FeelsLikeC", temp_c))
    humidity = float(cur.get("humidity", 50.0))
    uv = float(cur.get("uvIndex", 0.0))
    cloud = float(cur.get("cloudcover", 0.0))
    vis_km = float(cur.get("visibility", 10.0))
    pressure = float(cur.get("pressure", 1013.0))
    code = int(cur.get("weatherCode", 113))

    weather = {
        "temperature_2m": temp_c,
        "apparent_temperature": feels_c,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": wind_kmh,
        "wind_gusts_10m": round(wind_kmh * 1.25, 1),
        "precipitation": precip_mm,
        "rain": precip_mm,
        "showers": 0.0,
        "snowfall": 0.0,
        "weather_code": code,
        "cloud_cover": cloud,
        "uv_index": uv,
        "visibility": vis_km * 1000.0,  # convert km to meters
        "surface_pressure": pressure,
    }
    logger.info("wttr.in fallback weather fetched successfully for (%.4f, %.4f)", lat, lon)
    return weather


def fetch_weather(lat: float, lon: float, target_hour: int | None = None) -> dict[str, Any]:
    """Fetch current or target hourly weather conditions for the given coordinates.

    Uses OpenWeatherMap if OPENWEATHERMAP_API_KEY is present, with Open-Meteo & wttr.in fallbacks.

    Returns a flat dict whose keys match _CURRENT_FIELDS.

    Raises:
        WeatherFetchError: if all weather APIs fail.
    """
    cache_key = (round(lat, 2), round(lon, 2), target_hour)
    now = time.time()

    # Step 1: Check fresh global in-memory cache (15 min)
    if cache_key in _WEATHER_CACHE:
        ts, cached_weather = _WEATHER_CACHE[cache_key]
        age = now - ts
        if age < _WEATHER_TTL_S:
            logger.info(
                "Weather cache hit for (%.4f, %.4f, target_hour=%s) (age=%.1fs)",
                lat, lon, str(target_hour), age,
            )
            return cached_weather

    # Priority 1: OpenWeatherMap API if API key is configured
    owm_key = os.getenv("OPENWEATHERMAP_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
    if owm_key and not owm_key.startswith("your_"):
        try:
            owm_weather = _fetch_weather_openweathermap(lat, lon, owm_key, target_hour)
            _WEATHER_CACHE[cache_key] = (now, owm_weather)
            return owm_weather
        except Exception as exc:
            logger.warning("OpenWeatherMap fetch failed for (%.4f, %.4f): %s; falling back to Open-Meteo", lat, lon, exc)

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

    open_meteo_key = os.getenv("OPEN_METEO_API_KEY")
    if open_meteo_key:
        params["apikey"] = open_meteo_key

    # Only request hourly data if we actually need it for a specific time
    if target_hour is not None:
        params["hourly"] = ",".join(_HOURLY_FIELDS)
        params["forecast_days"] = 2

    last_exc = None
    data = None
    for attempt in range(2):
        if attempt > 0:
            time.sleep(1)  # 1s backoff before retrying
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
        # On failure (e.g. 429 Too Many Requests), check if stale cache item exists
        if cache_key in _WEATHER_CACHE:
            ts, stale_weather = _WEATHER_CACHE[cache_key]
            stale_age = now - ts
            if stale_age < _WEATHER_STALE_TTL_S:
                logger.warning(
                    "Serving STALE cached weather for (%.4f, %.4f, target_hour=%s) (age=%.1fs) due to API error: %s",
                    lat, lon, str(target_hour), stale_age, last_exc
                )
                return stale_weather

        # Secondary fallback: wttr.in JSON API
        try:
            wttr_weather = _fetch_weather_wttr(lat, lon, target_hour)
            _WEATHER_CACHE[cache_key] = (now, wttr_weather)
            return wttr_weather
        except Exception as wttr_exc:
            logger.warning("wttr.in fallback also failed for (%.4f, %.4f): %s", lat, lon, wttr_exc)

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
            _WEATHER_CACHE[cache_key] = (now, weather)
            return weather

    # Default to current real-time weather
    if not current or not isinstance(current, dict):
        logger.error("Unexpected weather payload structure: %s", data)
        # Stale cache check for malformed responses
        if cache_key in _WEATHER_CACHE:
            return _WEATHER_CACHE[cache_key][1]
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
    _WEATHER_CACHE[cache_key] = (now, weather)
    return weather

