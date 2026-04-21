"""
generate_two_days_temperature.py

Purpose
-------
Generate two different 24-hour outdoor-temperature profiles.
Each hour has one temperature value.
The values stay within 20-26 C and follow a reasonable daily shape:
coolest near early morning, warmest in the afternoon.

Usage
-----
python generate_two_days_temperature.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


OUTPUT_CSV = "data/synthetic_outdoor_temperature_two_days.csv"
T_MIN = 20.0
T_MAX = 26.0
SEED_DAY1 = 2026
SEED_DAY2 = 2027


def generate_one_day(seed: int, day_label: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = np.arange(24, dtype=float)

    mean_temp = (T_MIN + T_MAX) / 2.0
    amplitude = (T_MAX - T_MIN) / 2.0

    peak_hour = rng.uniform(14.0, 16.5)
    coolest_hour = rng.uniform(4.0, 6.5)
    base_shift = rng.uniform(-0.35, 0.35)

    diurnal = mean_temp + base_shift + amplitude * np.cos(2.0 * np.pi * (hours - peak_hour) / 24.0)

    morning_dip = -rng.uniform(0.2, 0.8) * np.exp(-0.5 * ((hours - coolest_hour) / 1.8) ** 2)
    evening_bump = rng.uniform(0.05, 0.35) * np.exp(-0.5 * ((hours - 19.0) / 2.4) ** 2)

    noise = rng.normal(0.0, 0.15, size=24)

    temp = diurnal + morning_dip + evening_bump + noise
    temp = np.clip(temp, T_MIN, T_MAX)
    temp = np.round(temp, 2)

    return pd.DataFrame({
        "day": day_label,
        "hour": np.arange(24, dtype=int),
        "Ta_C": temp
    })


def main() -> None:
    day1 = generate_one_day(SEED_DAY1, "day_1")
    day2 = generate_one_day(SEED_DAY2, "day_2")
    out = pd.concat([day1, day2], ignore_index=True)

    output_path = Path(OUTPUT_CSV)
    out.to_csv(output_path, index=False)

    print(f"Saved temperature profiles to: {output_path.resolve()}")
    print("\nDay 1:")
    print(day1.to_string(index=False))
    print("\nDay 2:")
    print(day2.to_string(index=False))


if __name__ == "__main__":
    main()
