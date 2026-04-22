"""Phase 2 smoke test.

Calls each of the 8 tools (plus 4 aliases = 12 registry entries) directly,
no LLM involved. Reports which succeed / fail with structured data.

Usage:  uv run python -m tests.phase2_tools_smoke
"""

from __future__ import annotations

import json
from datetime import datetime

from agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS, dispatch_tool_call

NOW_ISO = datetime.now().isoformat(timespec="seconds")
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")


TEST_CALLS: list[tuple[str, dict]] = [
    # -- Current-state tools (C1/C2) --
    ("get_weather",           {"city": "Pittsburgh"}),
    ("get_user_habits",       {"user_id": "default", "time_of_day": "afternoon"}),
    ("get_room_status",       {"room": "bedroom"}),
    ("get_energy_price",      {"time": NOW_ISO}),
    ("get_schedule",          {"user_id": "default", "date": TODAY_DATE}),
    ("get_solar_radiation",   {"time": NOW_ISO}),

    # -- NEW: forecast tools (C3 enablers) --
    ("get_weather_forecast",  {"city": "Pittsburgh", "hours_ahead": 24}),
    ("get_tariff_schedule",   {"time": NOW_ISO, "hours_ahead": 24}),

    # -- Milestone aliases (same implementations; verify registration) --
    ("get_current_weather",   {"city": "Pittsburgh"}),
    ("get_current_tariff",    {"time": NOW_ISO}),
    ("get_room_state",        {"room": "bedroom"}),
    ("get_user_schedule",     {"user_id": "default", "date": TODAY_DATE}),
]


def _summarize(result: dict | None, max_chars: int = 200) -> str:
    if result is None:
        return "None (failed silently — check logs)"
    as_json = json.dumps(result, default=str, ensure_ascii=False)
    if len(as_json) <= max_chars:
        return as_json
    return as_json[:max_chars] + f"... ({len(as_json)} chars total)"


def main() -> None:
    print(f"Registered tools: {len(TOOL_REGISTRY)} (expected 12)")
    print(f"Schema entries:   {len(TOOL_SCHEMAS)} (expected 8 — aliases share impls, no separate schema)")
    print(f"Registry keys:    {sorted(TOOL_REGISTRY.keys())}")
    print()

    results: list[dict] = []
    for tool_name, args in TEST_CALLS:
        print(f"--- {tool_name}({args}) ---")
        if tool_name not in TOOL_REGISTRY:
            print(f"  [UNREGISTERED]")
            results.append({"tool": tool_name, "status": "unregistered"})
            continue

        result = dispatch_tool_call(tool_name, args)
        if result is None:
            print(f"  [FAIL] dispatch returned None (see logs for exception)")
            results.append({"tool": tool_name, "status": "fail"})
        else:
            print(f"  [OK]   {_summarize(result)}")
            results.append({"tool": tool_name, "status": "ok", "result_keys": list(result.keys())})
        print()

    print("=" * 60)
    print("  PHASE 2 TOOL SMOKE TEST SUMMARY")
    print("=" * 60)
    status_counts: dict[str, int] = {}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"  {status:15s} {count}/{len(results)}")

    fail_tools = [r["tool"] for r in results if r["status"] != "ok"]
    if fail_tools:
        print(f"\n  Failing: {fail_tools}")
        print("  Expected fails: get_room_status/get_room_state (Home Assistant not connected)")
        print("  Other fails should be investigated.")
    else:
        print("\n  All 12 calls returned data. Phase 2 (T1/T2/T3/T6) is GREEN.")


if __name__ == "__main__":
    main()
