"""Interactive test script for the smart home agent.

Features:
1. Read user commands interactively from the terminal.
2. Build a live context with weather, energy price, solar, and habit data.
3. Present agent output in a clean, user-friendly English format.
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from agent.parser import check_ollama_connection, parse
from agent.schema import AgentInput
from agent.tools import (
    DEFAULT_SOLAR_LATITUDE,
    DEFAULT_SOLAR_LONGITUDE,
    dispatch_tool_call,
    get_energy_price,
    get_user_habits,
    get_weather,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(levelname)s | %(message)s")

DEFAULT_ROOM = "bedroom"
DEFAULT_USER = "default"
DEFAULT_CITY = "Pittsburgh"

DB_PATH = Path(__file__).resolve().parent / "data" / "smart_home.db"


def _list_available_options():
    """Read available rooms and users from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        rooms = [r[0] for r in cur.execute("SELECT room_id FROM rooms").fetchall()]
        users = list({
            r[0] for r in cur.execute("SELECT DISTINCT user_id FROM user_habits").fetchall()
        })
        conn.close()
        return rooms, users
    except Exception:
        return ["bedroom", "living_room"], ["default", "user1"]


def _get_current_time_of_day(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "night"


def build_context_from_tools(room: str, user_id: str, city: str) -> dict:
    """Build a realistic current context from tool calls."""
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    time_of_day = _get_current_time_of_day(now.hour)

    print("  Querying outdoor weather...")
    weather = get_weather(city)
    print(f"     -> {city}: {weather['outdoor_temp']}C, {weather['condition']}")

    print("  Querying energy price...")
    energy = get_energy_price(time_str)
    print(f"     -> {energy['tier']} tier, {energy['price_per_kwh']}/kWh")

    print("  Querying solar radiation (Open-Meteo, Pittsburgh defaults)...")
    solar = dispatch_tool_call("get_solar_radiation", {"time": time_str})
    if solar and solar.get("data_status") == "ok":
        ghi = solar.get("ghi")
        dni = solar.get("dni")
        dhi = solar.get("dhi")
        unit = solar.get("ghi_unit") or "W/m^2"
        print(
            "     -> "
            f"GHI={ghi} {unit}, DNI={dni} {unit}, DHI={dhi} {unit} "
            f"({DEFAULT_SOLAR_LATITUDE}, {DEFAULT_SOLAR_LONGITUDE})"
        )
    elif solar:
        note = solar.get("note", "No valid solar radiation data was available for this hour.")
        print(f"     -> Data unavailable: {note}")
    else:
        print("     -> Query failed. Solar radiation data was skipped.")

    print("  Querying user preferences...")
    habits = get_user_habits(user_id, time_of_day)
    print(f"     -> {time_of_day} preferred temperature: {habits['preferred_day_temp']}C")

    print("  Querying room status...")
    tools_logger = logging.getLogger("agent.tools")
    original_level = tools_logger.level
    tools_logger.setLevel(logging.CRITICAL)
    room_data = dispatch_tool_call("get_room_status", {"room": room})
    tools_logger.setLevel(original_level)

    if room_data:
        current_temp = room_data["current_temperature"]
        hvac_mode = room_data["current_hvac_mode"]
        fan_mode = room_data["current_fan_mode"]
        occupied = room_data["occupied"]
        window_open = room_data["window_open"]
        print(f"     -> Room temp: {current_temp}C, HVAC: {hvac_mode}")
    else:
        current_temp = weather["outdoor_temp"] + 2.0
        hvac_mode = "cool" if current_temp > 26 else "auto"
        fan_mode = habits.get("fan_preference", "auto")
        occupied = True
        window_open = False
        print(f"     -> Home Assistant unavailable. Using estimated room temp: {current_temp}C")

    return {
        "room": room,
        "current_temperature": round(current_temp, 1),
        "current_hvac_mode": hvac_mode,
        "current_fan_mode": fan_mode,
        "occupied": occupied,
        "window_open": window_open,
        "time": time_str,
    }


def format_output(result) -> str:
    """Render AgentOutput in a user-friendly format."""
    data = result.to_dict()
    reason = data.get("reason", "")

    for marker in ("(Validation adjusted:", "Validation adjusted:"):
        if marker in reason:
            reason = reason.split(marker)[0].strip()
            break

    lines = [
        f"  Room:               {data['room']}",
        f"  Target Temperature: {data['target_temperature']}C",
        f"  HVAC Mode:          {data['hvac_mode']}",
        f"  Fan Mode:           {data['fan_mode']}",
        f"  Preset Mode:        {data['preset_mode']}",
        f"  Deadband:           +/-{data['deadband']}C",
    ]

    solar = data.get("solar_radiation") or {}
    if solar.get("data_status") == "ok":
        unit = solar.get("ghi_unit") or "W/m^2"
        lines.append(
            f"  Solar Data:         GHI={solar.get('ghi')} {unit}, "
            f"DNI={solar.get('dni')} {unit}, DHI={solar.get('dhi')} {unit}"
        )
    elif solar:
        lines.append(f"  Solar Data:         {solar.get('data_status', 'unavailable')}")

    if data.get("valid_until"):
        lines.append(f"  Valid Until:        {data['valid_until']}")
    if reason:
        lines.append(f"  Reason:             {reason}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Smart Home Agent Interactive Test")
    print("=" * 60)

    print("\nChecking Ollama connection...")
    if not check_ollama_connection():
        print("ERROR: Could not connect to Ollama. Make sure `ollama serve` is running.")
        sys.exit(1)
    print("Ollama connection OK.\n")

    rooms, users = _list_available_options()
    print(f"Available rooms: {', '.join(rooms)}")
    print(f"Available users: {', '.join(users)}")
    print()

    room = input(f"Room ID [{DEFAULT_ROOM}]: ").strip() or DEFAULT_ROOM
    if room not in rooms:
        print(f"WARNING: '{room}' is not in the available room list. Using '{DEFAULT_ROOM}'.")
        room = DEFAULT_ROOM

    user_id = input(f"User ID [{DEFAULT_USER}]: ").strip() or DEFAULT_USER
    if user_id not in users:
        print(f"WARNING: '{user_id}' is not in the available user list. Using '{DEFAULT_USER}'.")
        user_id = DEFAULT_USER

    city = input(f"City [{DEFAULT_CITY}]: ").strip() or DEFAULT_CITY

    print("\nCollecting live context from tools...")
    context = build_context_from_tools(room, user_id, city)

    print(f"\n{'=' * 60}")
    print(
        "Context ready: "
        f"room temp {context['current_temperature']}C, "
        f"HVAC mode {context['current_hvac_mode']}"
    )
    print("Enter a natural-language command. Type `q` to quit or `r` to refresh context.")
    print("=" * 60)

    while True:
        try:
            user_cmd = input("\nYour command > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_cmd:
            continue
        if user_cmd.lower() == "q":
            print("Goodbye.")
            break
        if user_cmd.lower() == "r":
            print("Refreshing context...")
            context = build_context_from_tools(room, user_id, city)
            print(f"Context refreshed. Room temp: {context['current_temperature']}C")
            continue

        agent_input = AgentInput.from_dict({
            "user_command": user_cmd,
            "current_context": context,
        })

        print("\nRunning Qwen2.5:7b inference...\n")
        try:
            result = parse(agent_input)
        except Exception as exc:
            print(f"ERROR: Agent inference failed: {exc}")
            continue

        print("-" * 60)
        print("Agent recommended configuration:")
        print(format_output(result))
        print("-" * 60)

        show_raw = input("\nShow raw JSON? [y/N]: ").strip().lower()
        if show_raw == "y":
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
