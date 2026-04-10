"""Quick verification: all new test data is readable by tools."""
from agent.tools import get_user_habits, get_schedule, get_energy_price
import json

print("=== User Habits ===")
for uid in ["user_alex", "user_bob", "user_carol"]:
    for tod in ["morning", "afternoon", "night"]:
        r = get_user_habits(uid, tod)
        print(f"  {uid}/{tod}: day_temp={r['preferred_day_temp']}, "
              f"sleep_temp={r['preferred_sleep_temp']}, fan={r['fan_preference']}")

print("\n=== User Schedule ===")
for uid in ["user_alex", "user_bob", "user_carol"]:
    for d in ["2026-03-30", "2026-03-29"]:
        r = get_schedule(uid, d)
        print(f"  {uid}/{d}: {json.dumps(r, ensure_ascii=False)}")

print("\n=== Energy Price ===")
print(f"  Peak (14:00): {get_energy_price('2026-03-29T14:00:00')}")
print(f"  Off-peak (23:00): {get_energy_price('2026-03-29T23:00:00')}")

print("\nAll tests passed!")
