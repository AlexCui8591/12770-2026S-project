"""Agent tool implementations backed by live APIs and local data."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import requests
import yaml
from agent.mock_pid import DEFAULT_CONTROLLER

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = 5
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "smart_home.db"
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_SOLAR_LATITUDE = 40.4406
DEFAULT_SOLAR_LONGITUDE = -79.9959
DEFAULT_SOLAR_TIMEZONE = "America/New_York"
SOLAR_PAST_DAYS = 7
SOLAR_FORECAST_DAYS = 16
TOOL_OVERRIDE_REGISTRY: dict[str, Callable[..., dict] | None] = {}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_db() -> sqlite3.Connection:
    """Open the SQLite database in read-only mode."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _get_ha_connection() -> tuple[str, str]:
    """Resolve Home Assistant URL and token from env vars or config."""
    try:
        conn = _get_db()
        cur = conn.cursor()
        rows = {r["key"]: r["value"] for r in cur.execute("SELECT key, value FROM limits")}
        conn.close()
        db_url = rows.get("ha_url", "http://homeassistant.local:8123")
        db_token = rows.get("ha_token", "")
    except Exception:
        limits = _load_yaml(CONFIG_DIR / "limits.yaml")
        db_url = limits.get("ha_url", "http://homeassistant.local:8123")
        db_token = limits.get("ha_token", "")

    ha_url = os.environ.get("HA_URL", db_url)
    ha_token = os.environ.get("HA_TOKEN", db_token)
    return ha_url.rstrip("/"), ha_token


def set_tool_override(name: str, func: Callable[..., dict] | None) -> None:
    """Install or clear a temporary override for one tool."""
    if func is None:
        TOOL_OVERRIDE_REGISTRY.pop(name, None)
        return
    TOOL_OVERRIDE_REGISTRY[name] = func


def clear_tool_overrides(names: list[str] | None = None) -> None:
    """Clear temporary tool overrides."""
    if names is None:
        TOOL_OVERRIDE_REGISTRY.clear()
        return
    for name in names:
        TOOL_OVERRIDE_REGISTRY.pop(name, None)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the CURRENT outdoor weather for a city: temperature, humidity, condition, wind speed. Use this only for the present moment; for future hours use get_weather_forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, for example Pittsburgh.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_habits",
            "description": "Get the user's historical HVAC preferences, such as sleep temperature, daytime temperature, and fan preference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID.",
                    },
                    "time_of_day": {
                        "type": "string",
                        "enum": ["morning", "afternoon", "night"],
                        "description": "Time of day: morning, afternoon, or night.",
                    },
                },
                "required": ["user_id", "time_of_day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_room_status",
            "description": "Get the live room sensor and HVAC status, including temperature, humidity, HVAC mode, and window status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": "Room ID, for example bedroom.",
                    },
                },
                "required": ["room"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_energy_price",
            "description": "Get the CURRENT electricity pricing tier for a given time, including tier and next off-peak time. For a forward-looking price window use get_tariff_schedule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Time in ISO 8601 format, for example 2026-03-29T22:30:00.",
                    },
                },
                "required": ["time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schedule",
            "description": "Get the user's schedule for the day, such as leaving home or returning home.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User ID.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date, for example 2026-03-29.",
                    },
                },
                "required": ["user_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_solar_radiation",
            "description": (
                "Get solar radiation for a specific time and coordinates. Defaults to Pittsburgh and returns "
                "global horizontal irradiance (GHI), direct normal irradiance (DNI), and diffuse horizontal irradiance (DHI)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Time in ISO 8601 format, for example 2026-03-29T14:00:00.",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude. Defaults to 40.4406 (Pittsburgh).",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude. Defaults to -79.9959 (Pittsburgh).",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone for the input time. Defaults to America/New_York.",
                    },
                },
                "required": ["time"],
            },
        },
    },
    # -- Phase 2: forecast tools (C3 proactive-mode enablers) --
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get the OUTDOOR WEATHER FORECAST for the next hours_ahead hours (up to 72h, returned at 3-hour granularity). Use this when deciding whether to pre-heat or pre-cool ahead of a forecasted temperature swing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, for example Pittsburgh.",
                    },
                    "hours_ahead": {
                        "type": "integer",
                        "description": "How many hours of forecast to return (1-72). Default 24.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tariff_schedule",
            "description": "Get the ELECTRICITY TARIFF SCHEDULE for the next hours_ahead hours. Returns hourly (tier, price_per_kwh) plus the timestamp of the next off_peak window. Use this when deciding whether to shift load to cheaper hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Starting time in ISO 8601 format, for example 2026-03-29T22:30:00.",
                    },
                    "hours_ahead": {
                        "type": "integer",
                        "description": "How many hours ahead to retrieve (1-48). Default 24.",
                    },
                },
                "required": ["time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pid_telemetry",
            "description": (
                "Read the current PID controller state for supervisory reasoning. "
                "Returns gains (kp, ki, kd), setpoint, indoor_temp, tracking_error, "
                "control_signal (W), cumulative_energy_kwh, oscillation_count, "
                "cost_J (Milestone eq. 2), and timestamp. Use this to decide "
                "whether to HOLD, adjust gains, shift setpoint, or modify cost weights."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "description": "Room ID. Defaults to 'default' if your system is single-zone.",
                    },
                },
                "required": [],
            },
        },
    },
]


def _coerce_time_to_utc(time: str, timezone: str) -> datetime:
    """Convert an input timestamp into a UTC hour-aligned datetime."""
    tz = ZoneInfo(timezone)
    try:
        dt = datetime.fromisoformat(time)
    except (TypeError, ValueError):
        dt = datetime.now(tz)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)

    return dt.astimezone(ZoneInfo("UTC"))


def _clean_solar_value(value: Any) -> float | None:
    """Normalize invalid solar values to None."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric < 0:
        return None
    return numeric


def _request_open_meteo_hourly(
    latitude: float,
    longitude: float,
    timezone: str,
) -> dict:
    response = requests.get(
        OPEN_METEO_BASE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "shortwave_radiation,direct_normal_irradiance,diffuse_radiation",
            "past_days": SOLAR_PAST_DAYS,
            "forecast_days": SOLAR_FORECAST_DAYS,
            "timezone": timezone,
        },
        timeout=TOOL_TIMEOUT_SECONDS,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def get_weather(city: str) -> dict:
    """Query wttr.in for current outdoor weather."""
    logger.info("Tool call: get_weather(city=%s)", city)

    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=TOOL_TIMEOUT_SECONDS,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()
    current = data["current_condition"][0]

    return {
        "outdoor_temp": float(current["temp_C"]),
        "humidity": int(current["humidity"]),
        "condition": current["weatherDesc"][0]["value"],
        "wind_speed_kmh": int(current["windspeedKmph"]),
    }


def get_user_habits(user_id: str, time_of_day: str) -> dict:
    """Read historical HVAC preferences from SQLite."""
    logger.info("Tool call: get_user_habits(user_id=%s, time_of_day=%s)", user_id, time_of_day)

    conn = _get_db()
    cur = conn.cursor()

    row = cur.execute(
        "SELECT * FROM user_habits WHERE user_id = ? AND time_of_day = ?",
        (user_id, time_of_day),
    ).fetchone()

    if row is None:
        row = cur.execute(
            "SELECT * FROM user_habits WHERE user_id = 'default' AND time_of_day = ?",
            (time_of_day,),
        ).fetchone()

    conn.close()

    if row is None:
        return {
            "preferred_sleep_temp": 24.0,
            "preferred_day_temp": 25.0,
            "bedtime": "23:00",
            "wake_time": "07:30",
            "fan_preference": "auto",
        }

    return {
        "preferred_sleep_temp": (
            row["preferred_sleep_temp"]
            if row["preferred_sleep_temp"] is not None
            else row["preferred_temp"]
        ),
        "preferred_day_temp": row["preferred_temp"],
        "bedtime": row["bedtime"] or "23:00",
        "wake_time": row["wake_time"] or "07:30",
        "fan_preference": row["fan_preference"],
    }


def get_room_status(room: str) -> dict:
    """Query Home Assistant for live room state."""
    logger.info("Tool call: get_room_status(room=%s)", room)

    ha_url, ha_token = _get_ha_connection()

    conn = _get_db()
    row = conn.cursor().execute(
        "SELECT entity_id, sensor_entity_id FROM rooms WHERE room_id = ?",
        (room,),
    ).fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"Room '{room}' is not defined in the rooms table")

    climate_entity = row["entity_id"]
    sensor_entity = row["sensor_entity_id"]
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }

    sensor_response = requests.get(
        f"{ha_url}/api/states/{sensor_entity}",
        headers=headers,
        timeout=TOOL_TIMEOUT_SECONDS,
    )
    sensor_response.raise_for_status()
    sensor_data = sensor_response.json()

    climate_response = requests.get(
        f"{ha_url}/api/states/{climate_entity}",
        headers=headers,
        timeout=TOOL_TIMEOUT_SECONDS,
    )
    climate_response.raise_for_status()
    climate_data = climate_response.json()
    attrs = climate_data.get("attributes", {})

    return {
        "current_temperature": float(sensor_data.get("state", 0)),
        "current_humidity": attrs.get("current_humidity", 0),
        "current_hvac_mode": climate_data.get("state", "off"),
        "current_fan_mode": attrs.get("fan_mode", "auto"),
        "occupied": True,
        "window_open": False,
    }


def get_energy_price(time: str) -> dict:
    """Read the electricity price tier for a given time from SQLite."""
    logger.info("Tool call: get_energy_price(time=%s)", time)

    try:
        dt = datetime.fromisoformat(time)
        hour = dt.hour
    except (ValueError, TypeError):
        hour = datetime.now().hour

    conn = _get_db()
    cur = conn.cursor()

    row = cur.execute(
        "SELECT tier, price_per_kwh FROM energy_price WHERE hour = ?",
        (hour,),
    ).fetchone()

    if row:
        tier = row["tier"]
        price = row["price_per_kwh"]
    else:
        tier = "off_peak"
        price = 0.35

    off_peak_rows = cur.execute(
        "SELECT hour FROM energy_price WHERE tier = 'off_peak' ORDER BY hour"
    ).fetchall()
    conn.close()

    off_peak_hours = [r["hour"] for r in off_peak_rows]
    next_off_peak_hour = None
    for candidate_hour in off_peak_hours:
        if candidate_hour > hour:
            next_off_peak_hour = candidate_hour
            break
    if next_off_peak_hour is None and off_peak_hours:
        next_off_peak_hour = off_peak_hours[0]

    try:
        parsed_dt = datetime.fromisoformat(time)
        if next_off_peak_hour is not None and next_off_peak_hour <= hour:
            next_day = parsed_dt.date() + timedelta(days=1)
            next_off_peak = f"{next_day}T{next_off_peak_hour:02d}:00:00"
        elif next_off_peak_hour is not None:
            next_off_peak = f"{parsed_dt.date()}T{next_off_peak_hour:02d}:00:00"
        else:
            next_off_peak = None
    except (ValueError, TypeError):
        next_off_peak = None

    return {
        "price_per_kwh": price,
        "tier": tier,
        "next_off_peak": next_off_peak,
    }


def get_schedule(user_id: str, date: str) -> dict:
    """Read the user's schedule from SQLite."""
    logger.info("Tool call: get_schedule(user_id=%s, date=%s)", user_id, date)

    try:
        dt = datetime.fromisoformat(date)
        is_weekend = dt.weekday() >= 5
    except (ValueError, TypeError):
        is_weekend = False

    day_type = "weekend" if is_weekend else "weekday"

    conn = _get_db()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT event_type, event_time, note FROM user_schedule WHERE user_id = ? AND day_type = ?",
        (user_id, day_type),
    ).fetchall()

    if not rows:
        rows = cur.execute(
            "SELECT event_type, event_time, note FROM user_schedule WHERE user_id = 'default' AND day_type = ?",
            (day_type,),
        ).fetchall()

    conn.close()

    events = []
    for row in rows:
        event = {"type": row["event_type"], "at": row["event_time"]}
        if row["note"]:
            event["note"] = row["note"]
        events.append(event)

    return {"events": events}


def get_solar_radiation(
    time: str,
    latitude: float = DEFAULT_SOLAR_LATITUDE,
    longitude: float = DEFAULT_SOLAR_LONGITUDE,
    timezone: str = DEFAULT_SOLAR_TIMEZONE,
) -> dict:
    """Query Open-Meteo for hourly solar radiation."""
    logger.info(
        "Tool call: get_solar_radiation(time=%s, latitude=%s, longitude=%s, timezone=%s)",
        time,
        latitude,
        longitude,
        timezone,
    )

    dt_utc = _coerce_time_to_utc(time, timezone)
    requested_hour_utc = dt_utc.replace(minute=0, second=0, microsecond=0)
    payload = _request_open_meteo_hourly(latitude, longitude, "GMT")
    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError("Open-Meteo response did not include hourly data")

    result = {
        "source": "Open-Meteo Forecast API",
        "time_standard": "UTC",
        "requested_time": time,
        "requested_time_utc": requested_hour_utc.isoformat().replace("+00:00", "Z"),
        "resolved_time_utc": requested_hour_utc.isoformat().replace("+00:00", "Z"),
        "latitude": float(latitude),
        "longitude": float(longitude),
    }

    times = hourly.get("time", [])
    shortwave = hourly.get("shortwave_radiation", [])
    dni_values = hourly.get("direct_normal_irradiance", [])
    dhi_values = hourly.get("diffuse_radiation", [])

    found_index = None
    for idx, time_str in enumerate(times):
        try:
            candidate = datetime.fromisoformat(time_str).replace(tzinfo=ZoneInfo("UTC"))
        except ValueError:
            continue
        if candidate == requested_hour_utc:
            found_index = idx
            break

    if found_index is None:
        for idx in range(len(times) - 1, -1, -1):
            try:
                candidate = datetime.fromisoformat(times[idx]).replace(tzinfo=ZoneInfo("UTC"))
            except ValueError:
                continue
            if candidate <= requested_hour_utc:
                if any(
                    _clean_solar_value(values[idx]) is not None
                    for values in (shortwave, dni_values, dhi_values)
                    if idx < len(values)
                ):
                    found_index = idx
                    break

    result["ghi"] = None
    result["dni"] = None
    result["dhi"] = None
    result["ghi_unit"] = "W/m^2"
    result["dni_unit"] = "W/m^2"
    result["dhi_unit"] = "W/m^2"

    if found_index is not None:
        resolved_hour_utc = datetime.fromisoformat(times[found_index]).replace(tzinfo=ZoneInfo("UTC"))
        result["resolved_time_utc"] = resolved_hour_utc.isoformat().replace("+00:00", "Z")
        if found_index < len(shortwave):
            result["ghi"] = _clean_solar_value(shortwave[found_index])
        if found_index < len(dni_values):
            result["dni"] = _clean_solar_value(dni_values[found_index])
        if found_index < len(dhi_values):
            result["dhi"] = _clean_solar_value(dhi_values[found_index])
        result["data_status"] = "ok"
        result["fallback_applied"] = resolved_hour_utc != requested_hour_utc
        if resolved_hour_utc != requested_hour_utc:
            result["note"] = "No solar data was available for the requested hour, so the nearest hour with valid data was used."
    else:
        result["data_status"] = "unavailable"
        result["fallback_applied"] = False
        result["note"] = (
            "Open-Meteo did not return valid solar data for this time. "
            "The request may be outside the available forecast or past-days window."
        )

    return result


# =============================================================================
# Phase 2: forecast tools (C3 proactive-mode enablers).
# Added by the LLM stream; do not modify the functions above.
# =============================================================================


def get_weather_forecast(city: str, hours_ahead: int = 24) -> dict:
    """Query wttr.in for hourly outdoor weather forecast.

    wttr.in's ?format=j1 response contains a `weather` array of up to 3 days,
    each with an `hourly` sub-array at 3-hour granularity. We flatten it to
    a list of forecast points truncated to roughly hours_ahead hours.
    """
    logger.info("Tool call: get_weather_forecast(city=%s, hours_ahead=%d)", city, int(hours_ahead))

    hours_ahead = max(1, min(72, int(hours_ahead)))  # wttr.in's 3-day window

    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=TOOL_TIMEOUT_SECONDS,
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json()

    forecast_points: list[dict] = []
    for day in data.get("weather", []):
        date = day.get("date")
        for hourly in day.get("hourly", []):
            # wttr.in encodes "time" as unpadded HHMM strings: "0", "300", "600", ..., "2100"
            time_str = str(hourly.get("time", "0")).zfill(4)
            try:
                hh = int(time_str[:2])
            except ValueError:
                continue
            forecast_points.append({
                "timestamp": f"{date}T{hh:02d}:00:00",
                "outdoor_temp": float(hourly.get("tempC", 0)),
                "humidity": int(hourly.get("humidity", 0)),
                "condition": hourly.get("weatherDesc", [{}])[0].get("value", "unknown"),
                "wind_speed_kmh": int(hourly.get("windspeedKmph", 0)),
                "chance_of_rain": int(hourly.get("chanceofrain", 0)),
            })

    # ceil(hours_ahead / 3) since data is 3-hour spaced
    max_points = max(1, (hours_ahead + 2) // 3)
    forecast_points = forecast_points[:max_points]

    return {
        "city": city,
        "hours_ahead": hours_ahead,
        "granularity_hours": 3,
        "points": forecast_points,
    }


def get_tariff_schedule(time: str, hours_ahead: int = 24) -> dict:
    """Return the electricity tariff schedule for a forward window.

    The `energy_price` table has one row per hour-of-day (0-23). We walk
    `hours_ahead` hours forward from `time`, wrapping past midnight, and
    return the (timestamp, hour_of_day, tier, price) for each.
    """
    logger.info("Tool call: get_tariff_schedule(time=%s, hours_ahead=%d)", time, int(hours_ahead))

    try:
        dt = datetime.fromisoformat(time)
    except (ValueError, TypeError):
        dt = datetime.now()

    hours_ahead = max(1, min(48, int(hours_ahead)))

    conn = _get_db()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT hour, tier, price_per_kwh FROM energy_price ORDER BY hour"
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "start_time": time,
            "hours_ahead": hours_ahead,
            "schedule": [],
            "next_off_peak": None,
            "data_status": "unavailable",
            "note": "energy_price table is empty",
        }

    price_by_hour = {int(row["hour"]): (row["tier"], float(row["price_per_kwh"])) for row in rows}

    schedule: list[dict] = []
    for offset in range(hours_ahead):
        target_dt = dt + timedelta(hours=offset)
        hour_of_day = target_dt.hour
        if hour_of_day not in price_by_hour:
            continue
        tier, price = price_by_hour[hour_of_day]
        schedule.append({
            "timestamp": target_dt.isoformat(timespec="seconds"),
            "hour_of_day": hour_of_day,
            "tier": tier,
            "price_per_kwh": price,
        })

    next_off_peak: str | None = None
    for entry in schedule:
        if entry["tier"] == "off_peak":
            next_off_peak = entry["timestamp"]
            break

    return {
        "start_time": time,
        "hours_ahead": hours_ahead,
        "schedule": schedule,
        "next_off_peak": next_off_peak,
        "data_status": "ok",
    }

def get_pid_telemetry(room: str = "default") -> dict:
    """..."""
    logger.info("Tool call: get_pid_telemetry(room=%s)", room)
    # Import at call time so install_scenario() rebindings are visible.
    # If we used `from agent.mock_pid import DEFAULT_CONTROLLER` at module
    # top, we'd see a stale reference to the original controller.
    from agent import mock_pid
    telemetry = mock_pid.DEFAULT_CONTROLLER.get_telemetry()
    telemetry["room"] = room
    telemetry["source"] = "mock"
    return telemetry

TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    # Original names (parser.py and existing callers use these; keep them)
    "get_weather": get_weather,
    "get_user_habits": get_user_habits,
    "get_room_status": get_room_status,
    "get_energy_price": get_energy_price,
    "get_schedule": get_schedule,
    "get_solar_radiation": get_solar_radiation,

    # Phase 2: new forecast tools (C3 enablers)
    "get_weather_forecast": get_weather_forecast,
    "get_tariff_schedule": get_tariff_schedule,

    # Phase 2: Milestone-compliant aliases. Same implementations; these exist
    # so the LLM can call tools by the names used in the Milestone Report
    # without us having to rename the functions themselves.
    "get_current_weather": get_weather,
    "get_current_tariff": get_energy_price,
    "get_room_state": get_room_status,
    "get_user_schedule": get_schedule,
    "get_pid_telemetry": get_pid_telemetry,
    "get_pid_state":     get_pid_telemetry,
}


def dispatch_tool_call(name: str, arguments: dict) -> dict | None:
    """Dispatch a tool call and return None on handled failure."""
    override = TOOL_OVERRIDE_REGISTRY.get(name)
    if override is not None:
        try:
            return override(**arguments)
        except Exception:
            logger.exception("Tool override failed: %s(%s)", name, arguments)
            return None

    func = TOOL_REGISTRY.get(name)
    if func is None:
        logger.warning("Unknown tool: %s", name)
        return None
    try:
        return func(**arguments)
    except Exception:
        logger.exception("Tool call failed: %s(%s)", name, arguments)
        return None
