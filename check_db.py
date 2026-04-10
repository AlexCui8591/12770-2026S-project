"""Check database contents."""
import sqlite3

conn = sqlite3.connect("data/smart_home.db")
cur = conn.cursor()

tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for (name,) in tables:
    count = cur.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    print(f"{name}: {count} rows")

print("\n--- Sample: user_habits ---")
for row in cur.execute("SELECT user_id, time_of_day, preferred_temp, fan_preference FROM user_habits"):
    print(f"  {row}")

print("\n--- Sample: rooms ---")
for row in cur.execute("SELECT * FROM rooms"):
    print(f"  {row}")

print("\n--- Sample: energy_price ---")
for row in cur.execute("SELECT * FROM energy_price ORDER BY hour"):
    print(f"  {row}")

conn.close()
