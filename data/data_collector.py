import requests
import time
import csv
import os
from datetime import datetime

HA_URL = "http://homeassistant.local:8123"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI5MGZkOTY0NzE4Zjg0ZjM1OTJhM2UxMTE4ZjNmMTUxOCIsImlhdCI6MTc3NTA5MDMzOCwiZXhwIjoyMDkwNDUwMzM4fQ.HKrD-DS3VY6G2PZ-Q8QKBG4fHnOZSnJBttfVJ3WNFlA"
ENTITY_TEMP = "sensor.timmerflotte_temp_hmd_sensor_temperature"
SETPOINT_C = 26
LATITUDE = 40.4406
LONGITUDE = -79.9959
INTERVAL = 5
DURATION = 2 *3600
OUTPUT_FILE = "data.csv"

HA_HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def get_indoor_temp():
    r = requests.get(f"{HA_URL}/api/states/{ENTITY_TEMP}", headers=HA_HEADERS, timeout=5)
    return round((float(r.json()["state"]) - 32) * 5/9,2)

def get_outdoor_data():
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={LATITUDE}&longitude={LONGITUDE}"
           f"&current=temperature_2m,global_tilted_irradiance"
           f"&temperature_unit=celsius")
    r = requests.get(url, timeout=5).json()
    to    = r["current"]["temperature_2m"]
    i_sol = r["current"].get("global_tilted_irradiance", 0)
    return round(to, 2), round(i_sol, 2)

file_exists = os.path.exists(OUTPUT_FILE)
f = open(OUTPUT_FILE, "a", newline="")
writer = csv.writer(f)
if not file_exists:
    writer.writerow(["Time", "To", "Ti", "I_sol", "heat_on", "Virtual Power", "dt"])

print(f"Saving to {OUTPUT_FILE} (append mode)")
print("Press Ctrl+C to stop.\n")

start_time = time.time()
count =0

try:
    while time.time() - start_time < DURATION:
        try:
            ti = get_indoor_temp()
            to, i_sol = get_outdoor_data()
            heat_on = 1 if ti < SETPOINT_C else 0
            timestamp = datetime.now().strftime("%m/%d/%Y %H:%M")
            writer.writerow([timestamp, to, ti, i_sol, heat_on, 100, INTERVAL])
            f.flush()
            count += 1
            print(f"[{count}] {timestamp}  To={to}°C  Ti={ti}°C  I_sol={i_sol}  heat_on={heat_on}")

        except Exception as e:
            print(f"Error: {e}, retrying...")
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    pass

f.close()
print(f"\nStopped. {count} rows saved to {OUTPUT_FILE}")