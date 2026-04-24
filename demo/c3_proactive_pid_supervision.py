"""
run_c3_proactive_pid_supervision.py

C3 baseline:
- Initial PID parameters are directly inherited from the C0 run
- Setpoint schedule is the same as the updated C1 / C2 schedule
- The controller is supervised every 5 minutes
- Only PID parameters are adjusted online
- Setpoint is NOT adjusted online in this version
- Cost weights are NOT adjusted online in this version
- Proactive adjustment is triggered when the generated outdoor temperature
  changes by more than 0.5°C over the next 1 hour
- Future weather uses the generated temperature data directly
- User habits are used as additional proactive cues
- Initial indoor temperature equals the initial outdoor temperature
- Heater power is saturated to [0, 14.25] W

Outputs
-------
- outputs/c3_proactive_pid_outputs/c3_day2_timeseries.csv
- outputs/c3_proactive_pid_outputs/c3_supervision_log.csv
- outputs/c3_proactive_pid_outputs/c3_user_instruction_log.csv
- outputs/c3_proactive_pid_outputs/c3_day2_temperature_plot.png
- outputs/c3_proactive_pid_outputs/c3_day2_power_plot.png
- outputs/c3_proactive_pid_outputs/c3_day2_metrics.json
- outputs/c3_proactive_pid_outputs/c3_run_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.mock_pid import update_default_controller
from agent.proactive_supervisor import (
    DEFAULT_MODEL,
    ProactiveSupervisorInput,
    decide_proactive_pid,
)
from agent.supervisor import check_ollama_connection, create_ollama_client
from agent.tools import clear_tool_overrides, set_tool_override

BASE_DIR = Path(__file__).resolve().parent

RC_JSON = BASE_DIR / "config" / "rc_fit_results.json"
TEMP_CSV = BASE_DIR / "data" / "synthetic_outdoor_temperature_two_days.csv"
C0_SUMMARY_JSON = BASE_DIR / "outputs" / "c0_tuned_pid_outputs" / "c0_run_summary.json"

# Manual fallback if the C0 summary file is not available.
C0_KP = None
C0_KI = None
C0_KD = None

# Shared setpoint schedule with updated C1 / C2
SP_DEFAULT_NIGHT = 39.0
SP_DAY = 38.0
SP_EVENING = 40.0
SP_EVENT_18 = 41.0
SP_EVENT_02 = 38.0

# Simulation settings
DT_SECONDS = 60.0
PMAX = 14.25
DEADBAND = 0.2
IGNORE_FIRST_HOUR_FOR_COMFORT = 1.0
IGNORE_FIRST_HOUR_FOR_MO = 1.0

# C3 supervision settings
SUPERVISION_INTERVAL_MIN = 5
WINDOW_MIN = 5
FORECAST_HORIZON_MIN = 60
SIGNIFICANT_FORECAST_DELTA_C = 0.5

# PID bounds and step limits
KP_MIN, KP_MAX = 0.05, 15.0
KI_MIN, KI_MAX = 0.0, 5.0
KD_MIN, KD_MAX = 0.0, 3.0
MAX_KP_STEP = 0.30
MAX_KI_STEP = 0.20
MAX_KD_STEP = 0.20

# Evaluation thresholds
CVR_THRESHOLD_C = 1.5
SETTLING_BAND_C = 0.5

# Simple built-in TOU tariff for EC metric ($/kWh)
TARIFF_OFFPEAK = 0.12
TARIFF_ONPEAK = 0.20
ONPEAK_HOURS = set(range(16, 21))  # 16:00-20:59

OUTPUT_DIR = BASE_DIR / "outputs" / "c3_proactive_pid_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_USER_OBJECTIVE = (
    "Keep the bedroom comfortable with proactive adjustments ahead of forecast "
    "changes and scheduled comfort needs."
)
DEFAULT_ROOM = "bedroom"
DEFAULT_CITY = "Pittsburgh"
DEFAULT_USER_ID = "default"
TOOL_OVERRIDE_NAMES = ["get_weather_forecast", "get_schedule"]


def build_instruction_log() -> pd.DataFrame:
    records = [
        {
            "time": "2026-04-02 02:00:00",
            "user_utterance": "It's too warm now, lower it to 38 degrees.",
            "machine_command": json.dumps(
                {
                    "action": "set_setpoint",
                    "effective_start": "2026-04-02 02:00:00",
                    "effective_end": "2026-04-02 03:00:00",
                    "setpoint_C": 38.0,
                    "reason": "temporary night adjustment",
                },
                ensure_ascii=False,
            ),
        },
        {
            "time": "2026-04-02 03:00:00",
            "user_utterance": "No new instruction received. Revert to default night setting.",
            "machine_command": json.dumps(
                {
                    "action": "set_setpoint",
                    "effective_start": "2026-04-02 03:00:00",
                    "effective_end": "2026-04-02 09:00:00",
                    "setpoint_C": 39.0,
                    "reason": "return to default because no follow-up instruction",
                },
                ensure_ascii=False,
            ),
        },
        {
            "time": "2026-04-02 18:00:00",
            "user_utterance": "Turn it up to 41 degrees for now.",
            "machine_command": json.dumps(
                {
                    "action": "set_setpoint",
                    "effective_start": "2026-04-02 18:00:00",
                    "effective_end": "2026-04-02 19:00:00",
                    "setpoint_C": 41.0,
                    "reason": "temporary evening request",
                },
                ensure_ascii=False,
            ),
        },
        {
            "time": "2026-04-02 19:00:00",
            "user_utterance": "No new instruction received. Revert to regular evening target.",
            "machine_command": json.dumps(
                {
                    "action": "set_setpoint",
                    "effective_start": "2026-04-02 19:00:00",
                    "effective_end": "2026-04-02 22:00:00",
                    "setpoint_C": 40.0,
                    "reason": "return to default because no follow-up instruction",
                },
                ensure_ascii=False,
            ),
        },
        {
            "time": "2026-04-02 22:00:00",
            "user_utterance": "I'm going to sleep, set it to 39 degrees.",
            "machine_command": json.dumps(
                {
                    "action": "set_setpoint",
                    "effective_start": "2026-04-02 22:00:00",
                    "effective_end": "2026-04-03 02:00:00",
                    "setpoint_C": 39.0,
                    "reason": "sleep preference",
                },
                ensure_ascii=False,
            ),
        },
    ]
    return pd.DataFrame(records)


def habit_event_schedule() -> list[dict[str, object]]:
    return [
        {"hour": 2, "label": "night_cooler_preference", "type": "cooler"},
        {"hour": 18, "label": "evening_high_temperature_demand", "type": "warmer"},
        {"hour": 22, "label": "sleep_stability_preference", "type": "stability"},
    ]


def load_rc_params(path: str = RC_JSON) -> tuple[float, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find RC parameter file: {p.resolve()}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data["R_hat_K_per_W"]), float(data["C_hat_J_per_K"])


def load_hourly_temperature_days(path: str = TEMP_CSV) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find temperature file: {p.resolve()}")

    df = pd.read_csv(p)
    required = ["day", "hour", "Ta_C"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    day1 = df[df["day"] == "day_1"].copy().sort_values("hour").reset_index(drop=True)
    day2 = df[df["day"] == "day_2"].copy().sort_values("hour").reset_index(drop=True)

    if len(day1) != 24 or len(day2) != 24:
        raise ValueError("Expected exactly 24 rows for day_1 and 24 rows for day_2.")

    return day1, day2


def expand_hourly_to_minute(hourly_values: np.ndarray) -> np.ndarray:
    return np.repeat(np.asarray(hourly_values, dtype=float), 60)


def build_minute_time_index(day_label: str) -> pd.DatetimeIndex:
    start = "2026-04-01 00:00:00" if day_label == "day_1" else "2026-04-02 00:00:00"
    return pd.date_range(start=start, periods=24 * 60, freq="min")


def setpoint_from_timestamp(ts: pd.Timestamp) -> float:
    hour = ts.hour
    if 0 <= hour < 2:
        return SP_DEFAULT_NIGHT
    if 2 <= hour < 3:
        return SP_EVENT_02
    if 3 <= hour < 9:
        return SP_DEFAULT_NIGHT
    if 9 <= hour < 18:
        return SP_DAY
    if 18 <= hour < 19:
        return SP_EVENT_18
    if 19 <= hour < 22:
        return SP_EVENING
    return SP_DEFAULT_NIGHT


def build_setpoint_schedule(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([setpoint_from_timestamp(ts) for ts in time_index], dtype=float)


def build_tariff_schedule(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.array(
        [TARIFF_ONPEAK if ts.hour in ONPEAK_HOURS else TARIFF_OFFPEAK for ts in time_index],
        dtype=float,
    )


def _ignore_start_index(hours_to_ignore: float, dt_seconds: float) -> int:
    return max(0, int(round(hours_to_ignore * 3600.0 / dt_seconds)))


def load_c0_pid_parameters(
    c0_summary_path: str = C0_SUMMARY_JSON,
    kp_fallback: float | None = C0_KP,
    ki_fallback: float | None = C0_KI,
    kd_fallback: float | None = C0_KD,
) -> tuple[float, float, float, str]:
    p = Path(c0_summary_path)

    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        pid = data.get("fixed_pid_after_tuning", {})
        kp = pid.get("KP")
        ki = pid.get("KI")
        kd = pid.get("KD")
        if kp is None or ki is None or kd is None:
            raise ValueError(
                f"C0 summary file exists but fixed_pid_after_tuning is incomplete: {p.resolve()}"
            )
        return float(kp), float(ki), float(kd), f"loaded from {p.as_posix()}"

    if kp_fallback is None or ki_fallback is None or kd_fallback is None:
        raise FileNotFoundError(
            "Could not find outputs/c0_tuned_pid_outputs/c0_run_summary.json, "
            "and manual fallback C0_KP/C0_KI/C0_KD are not filled."
        )

    return float(kp_fallback), float(ki_fallback), float(kd_fallback), "manual fallback constants"


def clip_with_step(old: float, new: float, low: float, high: float, max_step: float) -> float:
    new = max(low, min(high, new))
    delta = max(-max_step, min(max_step, new - old))
    return float(max(low, min(high, old + delta)))


def pid_controller(
    ti: float,
    tsp: float,
    kp: float,
    ki: float,
    kd: float,
    dt_seconds: float,
    integral_prev: float,
    error_prev: float,
    pmax: float,
    deadband: float = 0.0,
) -> tuple[float, float, float, float]:
    error = tsp - ti

    if abs(error) <= deadband:
        error = 0.0

    dt_hours = dt_seconds / 3600.0
    derivative = (error - error_prev) / dt_hours if dt_hours > 0 else 0.0

    integral_candidate = integral_prev + error * dt_hours
    u_unsat = kp * error + ki * integral_candidate + kd * derivative
    u = float(np.clip(u_unsat, 0.0, pmax))

    if (0.0 < u_unsat < pmax) or (u == 0.0 and error > 0.0) or (u == pmax and error < 0.0):
        integral_new = integral_candidate
    else:
        integral_new = integral_prev

    return u, integral_new, error, derivative


@dataclass
class Building1R1C:
    ti: float
    r: float
    c: float

    def next_step(self, p_hea: float, ta: float, dt_seconds: float) -> None:
        a = self.c / dt_seconds
        inv_r = 1.0 / self.r
        self.ti = (a * self.ti + inv_r * ta + p_hea) / (a + inv_r)

    def get_state(self) -> float:
        return float(self.ti)


def summarize_window(
    time_index: pd.DatetimeIndex,
    ti_hist: np.ndarray,
    tsp_hist: np.ndarray,
    phea_hist: np.ndarray,
    err_hist: np.ndarray,
    start_idx: int,
    end_idx: int,
    dt_seconds: float,
) -> dict[str, float | int | str]:
    ti_w = ti_hist[start_idx:end_idx]
    tsp_w = tsp_hist[start_idx:end_idx]
    p_w = phea_hist[start_idx:end_idx]
    e_w = err_hist[start_idx:end_idx]

    if len(e_w) == 0:
        return {
            "window_start": "",
            "window_end": "",
            "mean_abs_error": 0.0,
            "mean_error": 0.0,
            "max_abs_error": 0.0,
            "power_mean_W": 0.0,
            "power_std_W": 0.0,
            "last_power_W": 0.0,
            "energy_kWh": 0.0,
            "error_sign_changes": 0,
            "last_error": 0.0,
            "current_ti": 0.0,
            "current_tsp": 0.0,
        }

    signs = np.sign(e_w)
    nonzero = [int(s) for s in signs if s != 0]
    sign_changes = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])

    dt_hours = dt_seconds / 3600.0
    energy_kwh = float(np.sum(p_w * dt_hours) / 1000.0)

    return {
        "window_start": str(time_index[start_idx]),
        "window_end": str(time_index[end_idx - 1]),
        "mean_abs_error": float(np.mean(np.abs(e_w))),
        "mean_error": float(np.mean(e_w)),
        "max_abs_error": float(np.max(np.abs(e_w))),
        "power_mean_W": float(np.mean(p_w)),
        "power_std_W": float(np.std(p_w)),
        "last_power_W": float(p_w[-1]),
        "energy_kWh": energy_kwh,
        "error_sign_changes": int(sign_changes),
        "last_error": float(e_w[-1]),
        "current_ti": float(ti_w[-1]),
        "current_tsp": float(tsp_w[-1]),
    }


def summarize_forecast(
    time_index: pd.DatetimeIndex,
    ta_array: np.ndarray,
    current_idx: int,
    horizon_min: int,
) -> dict[str, float | int | str]:
    horizon_steps = horizon_min
    end_idx = min(len(ta_array), current_idx + horizon_steps + 1)

    ta_now = float(ta_array[current_idx])
    ta_future = float(ta_array[end_idx - 1])
    delta_1h = ta_future - ta_now
    ta_mean_future = float(np.mean(ta_array[current_idx:end_idx]))

    current_ts = time_index[current_idx]
    minutes_to_next_habit = None
    next_habit_label = "none"
    next_habit_type = "none"

    for event in habit_event_schedule():
        event_ts = current_ts.normalize() + pd.Timedelta(hours=int(event["hour"]))
        delta_min = (event_ts - current_ts).total_seconds() / 60.0
        if 0 <= delta_min <= horizon_min:
            minutes_to_next_habit = int(delta_min)
            next_habit_label = str(event["label"])
            next_habit_type = str(event["type"])
            break

    return {
        "forecast_horizon_min": horizon_min,
        "Ta_now_C": ta_now,
        "Ta_future_C": ta_future,
        "Ta_mean_future_C": ta_mean_future,
        "Ta_delta_next_1h_C": float(delta_1h),
        "next_habit_label": next_habit_label,
        "next_habit_type": next_habit_type,
        "minutes_to_next_habit": -1 if minutes_to_next_habit is None else minutes_to_next_habit,
    }


def reactive_pid_supervisor(
    telemetry: dict[str, float | int | str],
    kp: float,
    ki: float,
    kd: float,
) -> tuple[float, float, float, str]:
    mean_abs_error = float(telemetry["mean_abs_error"])
    mean_error = float(telemetry["mean_error"])
    max_abs_error = float(telemetry["max_abs_error"])
    power_std = float(telemetry["power_std_W"])
    sign_changes = int(telemetry["error_sign_changes"])
    last_error = float(telemetry["last_error"])

    new_kp, new_ki, new_kd = kp, ki, kd
    action = "hold"

    if mean_abs_error > 1.5 and mean_error > 0.5:
        new_kp = clip_with_step(kp, kp * 1.10, KP_MIN, KP_MAX, MAX_KP_STEP)
        new_ki = clip_with_step(ki, ki * 1.08 + 0.01, KI_MIN, KI_MAX, MAX_KI_STEP)
        action = "increase_kp_ki_for_underheating"
    elif mean_abs_error > 1.0 and mean_error < -0.5:
        new_kp = clip_with_step(kp, kp * 0.93, KP_MIN, KP_MAX, MAX_KP_STEP)
        new_ki = clip_with_step(ki, ki * 0.90, KI_MIN, KI_MAX, MAX_KI_STEP)
        action = "decrease_kp_ki_for_overshoot"
    elif sign_changes >= 2 or (power_std > 2.0 and max_abs_error < 1.2):
        new_kp = clip_with_step(kp, kp * 0.92, KP_MIN, KP_MAX, MAX_KP_STEP)
        new_kd = clip_with_step(kd, kd * 1.10 + 0.01, KD_MIN, KD_MAX, MAX_KD_STEP)
        action = "reduce_oscillation_lower_kp_raise_kd"
    elif 0.3 < mean_abs_error < 1.2 and last_error > 0.2:
        new_ki = clip_with_step(ki, ki * 1.08 + 0.005, KI_MIN, KI_MAX, MAX_KI_STEP)
        action = "raise_ki_for_small_persistent_error"

    return float(new_kp), float(new_ki), float(new_kd), action


def proactive_pid_supervisor(
    telemetry: dict[str, float | int | str],
    forecast: dict[str, float | int | str],
    kp: float,
    ki: float,
    kd: float,
) -> tuple[float, float, float, str, str]:
    delta_1h = float(forecast["Ta_delta_next_1h_C"])
    next_habit_type = str(forecast["next_habit_type"])
    mins_to_habit = int(forecast["minutes_to_next_habit"])

    new_kp, new_ki, new_kd = kp, ki, kd
    action = "hold"
    reason = "no proactive trigger"

    if delta_1h <= -SIGNIFICANT_FORECAST_DELTA_C:
        new_kp = clip_with_step(kp, kp * 1.10, KP_MIN, KP_MAX, MAX_KP_STEP)
        new_ki = clip_with_step(ki, ki * 1.08 + 0.01, KI_MIN, KI_MAX, MAX_KI_STEP)
        action = "proactive_prepare_for_outdoor_cooling"
        reason = f"forecast next-1h outdoor temperature drop = {delta_1h:.3f}°C"
        return new_kp, new_ki, new_kd, action, reason

    if delta_1h >= SIGNIFICANT_FORECAST_DELTA_C:
        new_kp = clip_with_step(kp, kp * 0.94, KP_MIN, KP_MAX, MAX_KP_STEP)
        new_ki = clip_with_step(ki, ki * 0.92, KI_MIN, KI_MAX, MAX_KI_STEP)
        action = "proactive_prepare_for_outdoor_warming"
        reason = f"forecast next-1h outdoor temperature rise = {delta_1h:.3f}°C"
        return new_kp, new_ki, new_kd, action, reason

    if 0 <= mins_to_habit <= FORECAST_HORIZON_MIN:
        if next_habit_type == "warmer":
            new_kp = clip_with_step(kp, kp * 1.08, KP_MIN, KP_MAX, MAX_KP_STEP)
            new_ki = clip_with_step(ki, ki * 1.06 + 0.005, KI_MIN, KI_MAX, MAX_KI_STEP)
            action = "proactive_prepare_for_warmer_habit"
            reason = f"user habit event '{forecast['next_habit_label']}' in {mins_to_habit} min"
            return new_kp, new_ki, new_kd, action, reason

        if next_habit_type == "stability":
            new_kp = clip_with_step(kp, kp * 0.94, KP_MIN, KP_MAX, MAX_KP_STEP)
            new_kd = clip_with_step(kd, kd * 1.10 + 0.01, KD_MIN, KD_MAX, MAX_KD_STEP)
            action = "proactive_prepare_for_stability_habit"
            reason = f"user habit event '{forecast['next_habit_label']}' in {mins_to_habit} min"
            return new_kp, new_ki, new_kd, action, reason

        if next_habit_type == "cooler":
            new_ki = clip_with_step(ki, ki * 0.90, KI_MIN, KI_MAX, MAX_KI_STEP)
            new_kp = clip_with_step(kp, kp * 0.96, KP_MIN, KP_MAX, MAX_KP_STEP)
            action = "proactive_prepare_for_cooler_habit"
            reason = f"user habit event '{forecast['next_habit_label']}' in {mins_to_habit} min"
            return new_kp, new_ki, new_kd, action, reason

    return new_kp, new_ki, new_kd, action, reason


def _sync_agent_pid_snapshot(
    ts: pd.Timestamp,
    telemetry: dict[str, float | int | str],
    kp: float,
    ki: float,
    kd: float,
    ta_now: float,
) -> None:
    cost_proxy = float(telemetry["mean_abs_error"]) ** 2 + float(telemetry["energy_kWh"]) * 10.0
    update_default_controller(
        kp=kp,
        ki=ki,
        kd=kd,
        setpoint=float(telemetry["current_tsp"]),
        indoor_temp=float(telemetry["current_ti"]),
        tracking_error=float(telemetry["last_error"]),
        control_signal=float(telemetry["last_power_W"]),
        cumulative_energy_kwh=float(telemetry["energy_kWh"]),
        oscillation_count=int(telemetry["error_sign_changes"]),
        cost_J=cost_proxy,
        outdoor_temp=float(ta_now),
        timestamp=ts.to_pydatetime(),
    )


def _build_weather_forecast_override(
    time_index: pd.DatetimeIndex,
    ta_array: np.ndarray,
    current_idx: int,
):
    def _override(city: str, hours_ahead: int = 24) -> dict:
        hours_ahead = max(1, min(72, int(hours_ahead)))
        horizon_steps = min(len(ta_array) - current_idx - 1, hours_ahead * 60)
        sample_offsets = list(range(0, max(1, horizon_steps) + 1, 180))
        if not sample_offsets:
            sample_offsets = [0]

        points = []
        for offset in sample_offsets:
            idx = min(len(ta_array) - 1, current_idx + offset)
            points.append(
                {
                    "timestamp": pd.Timestamp(time_index[idx]).isoformat(),
                    "outdoor_temp": float(ta_array[idx]),
                    "humidity": 50,
                    "condition": "synthetic",
                    "wind_speed_kmh": 0,
                    "chance_of_rain": 0,
                }
            )

        return {
            "city": city,
            "hours_ahead": hours_ahead,
            "granularity_hours": 3,
            "points": points,
            "source": "synthetic_scenario",
        }

    return _override


def _build_schedule_override(current_ts: pd.Timestamp):
    def _override(user_id: str, date: str) -> dict:
        events = []
        base_day = current_ts.normalize()
        for event in habit_event_schedule():
            event_ts = base_day + pd.Timedelta(hours=int(event["hour"]))
            if event_ts >= current_ts:
                events.append(
                    {
                        "type": str(event["type"]),
                        "at": event_ts.isoformat(),
                        "note": str(event["label"]),
                    }
                )
        return {"events": events, "source": "synthetic_schedule"}

    return _override


def _install_proactive_tool_context(
    time_index: pd.DatetimeIndex,
    ta_array: np.ndarray,
    current_idx: int,
) -> None:
    set_tool_override(
        "get_weather_forecast",
        _build_weather_forecast_override(time_index, ta_array, current_idx),
    )
    set_tool_override(
        "get_schedule",
        _build_schedule_override(time_index[current_idx]),
    )


def _agent_proactive_pid_supervisor(
    time_index: pd.DatetimeIndex,
    idx: int,
    ta_array: np.ndarray,
    telemetry: dict[str, float | int | str],
    forecast: dict[str, float | int | str],
    kp: float,
    ki: float,
    kd: float,
    *,
    room: str,
    city: str,
    user_id: str,
    user_objective: str,
    llm_model: str,
    allow_fallback: bool,
    supervisor_client=None,
) -> tuple[float, float, float, str, str, str, str]:
    _sync_agent_pid_snapshot(time_index[idx], telemetry, kp, ki, kd, ta_array[idx])
    _install_proactive_tool_context(time_index, ta_array, idx)

    supervisor_input = ProactiveSupervisorInput(
        room=room,
        city=city,
        user_id=user_id,
        user_objective=user_objective,
        current_time=str(time_index[idx]),
        telemetry_window=telemetry,
        forecast_summary=forecast,
        current_kp=kp,
        current_ki=ki,
        current_kd=kd,
    )

    try:
        decision = decide_proactive_pid(
            supervisor_input,
            client=supervisor_client,
            model=llm_model,
        )
        if decision.mode == "hold":
            return float(kp), float(ki), float(kd), "hold", "hold", decision.source, decision.rationale

        kp_new = clip_with_step(kp, kp if decision.kp is None else decision.kp, KP_MIN, KP_MAX, MAX_KP_STEP)
        ki_new = clip_with_step(ki, ki if decision.ki is None else decision.ki, KI_MIN, KI_MAX, MAX_KI_STEP)
        kd_new = clip_with_step(kd, kd if decision.kd is None else decision.kd, KD_MIN, KD_MAX, MAX_KD_STEP)
        return float(kp_new), float(ki_new), float(kd_new), decision.mode, decision.action, decision.source, decision.rationale
    except Exception as exc:
        if not allow_fallback:
            raise

        kp_prop, ki_prop, kd_prop, proactive_action, proactive_reason = proactive_pid_supervisor(
            telemetry=telemetry,
            forecast=forecast,
            kp=kp,
            ki=ki,
            kd=kd,
        )

        decision_mode = "proactive"
        action = proactive_action
        rationale = proactive_reason

        if proactive_action == "hold":
            kp_prop, ki_prop, kd_prop, reactive_action = reactive_pid_supervisor(
                telemetry=telemetry,
                kp=kp,
                ki=ki,
                kd=kd,
            )
            decision_mode = "reactive" if reactive_action != "hold" else "hold"
            action = reactive_action
            rationale = "no proactive trigger; fallback to reactive rule"

        fallback_rationale = f"agent failed; fell back to rules ({type(exc).__name__}: {exc}); {rationale}"
        return (
            float(kp_prop),
            float(ki_prop),
            float(kd_prop),
            decision_mode,
            action,
            "rules_fallback",
            fallback_rationale,
        )
    finally:
        clear_tool_overrides(TOOL_OVERRIDE_NAMES)


def simulate_proactive_pid_supervision(
    time_index: pd.DatetimeIndex,
    ta_array: np.ndarray,
    tsp_array: np.ndarray,
    r: float,
    c: float,
    ti0: float,
    kp0: float,
    ki0: float,
    kd0: float,
    pmax: float = PMAX,
    deadband: float = DEADBAND,
    dt_seconds: float = DT_SECONDS,
    supervision_interval_min: int = SUPERVISION_INTERVAL_MIN,
    window_min: int = WINDOW_MIN,
    supervisor_mode: str = "agent_auto",
    user_objective: str = DEFAULT_USER_OBJECTIVE,
    room: str = DEFAULT_ROOM,
    city: str = DEFAULT_CITY,
    user_id: str = DEFAULT_USER_ID,
    llm_model: str = DEFAULT_MODEL,
    supervisor_client=None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    n = len(time_index)
    bldg = Building1R1C(ti=float(ti0), r=float(r), c=float(c))

    ti_hist = np.zeros(n, dtype=float)
    phea_hist = np.zeros(n, dtype=float)
    err_hist = np.zeros(n, dtype=float)
    int_hist = np.zeros(n, dtype=float)
    der_hist = np.zeros(n, dtype=float)
    kp_hist = np.zeros(n, dtype=float)
    ki_hist = np.zeros(n, dtype=float)
    kd_hist = np.zeros(n, dtype=float)

    integral = 0.0
    error_prev = 0.0
    kp, ki, kd = float(kp0), float(ki0), float(kd0)

    sup_every = max(1, int(round(supervision_interval_min * 60 / dt_seconds)))
    window_steps = max(1, int(round(window_min * 60 / dt_seconds)))
    supervision_records = []

    for k in range(n):
        if k > 0 and k % sup_every == 0:
            start_idx = max(0, k - window_steps)
            telemetry = summarize_window(
                time_index=time_index,
                ti_hist=ti_hist,
                tsp_hist=tsp_array,
                phea_hist=phea_hist,
                err_hist=err_hist,
                start_idx=start_idx,
                end_idx=k,
                dt_seconds=dt_seconds,
            )
            forecast = summarize_forecast(
                time_index=time_index,
                ta_array=ta_array,
                current_idx=k,
                horizon_min=FORECAST_HORIZON_MIN,
            )

            if supervisor_mode == "rules":
                kp_prop, ki_prop, kd_prop, proactive_action, proactive_reason = proactive_pid_supervisor(
                    telemetry=telemetry,
                    forecast=forecast,
                    kp=kp,
                    ki=ki,
                    kd=kd,
                )

                decision_mode = "proactive"
                action = proactive_action
                reason = proactive_reason
                decision_source = "rules"

                if proactive_action == "hold":
                    kp_prop, ki_prop, kd_prop, reactive_action = reactive_pid_supervisor(
                        telemetry=telemetry,
                        kp=kp,
                        ki=ki,
                        kd=kd,
                    )
                    decision_mode = "reactive" if reactive_action != "hold" else "hold"
                    action = reactive_action
                    reason = "no proactive trigger; fallback to reactive rule"
            else:
                kp_prop, ki_prop, kd_prop, decision_mode, action, decision_source, reason = _agent_proactive_pid_supervisor(
                    time_index=time_index,
                    idx=k,
                    ta_array=ta_array,
                    telemetry=telemetry,
                    forecast=forecast,
                    kp=kp,
                    ki=ki,
                    kd=kd,
                    room=room,
                    city=city,
                    user_id=user_id,
                    user_objective=user_objective,
                    llm_model=llm_model,
                    allow_fallback=(supervisor_mode == "agent_auto"),
                    supervisor_client=supervisor_client,
                )

            supervision_records.append({
                "time": str(time_index[k]),
                "decision_mode": decision_mode,
                "window_start": telemetry["window_start"],
                "window_end": telemetry["window_end"],
                "current_ti_C": telemetry["current_ti"],
                "current_tsp_C": telemetry["current_tsp"],
                "mean_abs_error_C": telemetry["mean_abs_error"],
                "mean_error_C": telemetry["mean_error"],
                "max_abs_error_C": telemetry["max_abs_error"],
                "power_mean_W": telemetry["power_mean_W"],
                "power_std_W": telemetry["power_std_W"],
                "last_power_W": telemetry["last_power_W"],
                "energy_kWh": telemetry["energy_kWh"],
                "error_sign_changes": telemetry["error_sign_changes"],
                "Ta_now_C": forecast["Ta_now_C"],
                "Ta_future_C": forecast["Ta_future_C"],
                "Ta_mean_future_C": forecast["Ta_mean_future_C"],
                "Ta_delta_next_1h_C": forecast["Ta_delta_next_1h_C"],
                "next_habit_label": forecast["next_habit_label"],
                "next_habit_type": forecast["next_habit_type"],
                "minutes_to_next_habit": forecast["minutes_to_next_habit"],
                "kp_before": kp,
                "ki_before": ki,
                "kd_before": kd,
                "action": action,
                "decision_source": decision_source,
                "reason": reason,
                "rationale": reason,
                "kp_after": kp_prop,
                "ki_after": ki_prop,
                "kd_after": kd_prop,
            })

            kp, ki, kd = kp_prop, ki_prop, kd_prop

        ti_k = bldg.get_state()
        tsp_k = float(tsp_array[k])

        p_k, integral, error_now, derivative_now = pid_controller(
            ti=ti_k,
            tsp=tsp_k,
            kp=kp,
            ki=ki,
            kd=kd,
            dt_seconds=dt_seconds,
            integral_prev=integral,
            error_prev=error_prev,
            pmax=pmax,
            deadband=deadband,
        )

        ti_hist[k] = ti_k
        phea_hist[k] = p_k
        err_hist[k] = error_now
        int_hist[k] = integral
        der_hist[k] = derivative_now
        kp_hist[k] = kp
        ki_hist[k] = ki
        kd_hist[k] = kd

        if k < n - 1:
            bldg.next_step(
                p_hea=p_k,
                ta=float(ta_array[k + 1]),
                dt_seconds=dt_seconds,
            )

        error_prev = error_now

    sim = {
        "Ti": ti_hist,
        "Phea": phea_hist,
        "err": err_hist,
        "int": int_hist,
        "der": der_hist,
        "Kp": kp_hist,
        "Ki": ki_hist,
        "Kd": kd_hist,
    }
    sup_df = pd.DataFrame(supervision_records)
    return sim, sup_df


def _first_constant_setpoint_segment(tsp: np.ndarray) -> tuple[int, int]:
    tsp = np.asarray(tsp, dtype=float)
    if len(tsp) == 0:
        return 0, 0

    start = 0
    end = len(tsp)
    for i in range(1, len(tsp)):
        if not np.isclose(tsp[i], tsp[0]):
            end = i
            break
    return start, end


def _compute_rt_seconds(ti_seg: np.ndarray, tsp_seg: np.ndarray, dt_seconds: float) -> float | None:
    if len(ti_seg) == 0:
        return None

    ti0 = float(ti_seg[0])
    target = float(tsp_seg[0])
    step_mag = abs(target - ti0)
    if step_mag < 1e-12:
        return 0.0

    threshold = 0.1 * step_mag
    err = np.abs(ti_seg - target)
    idx = np.where(err <= threshold)[0]
    if len(idx) == 0:
        return None
    return float(idx[0] * dt_seconds)


def _compute_st_seconds(ti_seg: np.ndarray, tsp_seg: np.ndarray, dt_seconds: float, band_c: float) -> float | None:
    if len(ti_seg) == 0:
        return None

    inside = np.abs(ti_seg - tsp_seg) <= band_c
    for i in range(len(inside)):
        if np.all(inside[i:]):
            return float(i * dt_seconds)
    return None


def _compute_oc_after_settling(ti_seg: np.ndarray, tsp_seg: np.ndarray, settling_idx: int | None) -> int | None:
    if len(ti_seg) == 0 or settling_idx is None or settling_idx >= len(ti_seg):
        return None

    residual = ti_seg[settling_idx:] - tsp_seg[settling_idx:]
    signs = np.sign(residual)
    nonzero = [int(s) for s in signs if s != 0]
    if len(nonzero) < 2:
        return 0

    return int(sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1]))



def _find_setpoint_change_indices(
    tsp: np.ndarray,
    dt_seconds: float,
    hours_to_ignore: float,
) -> list[int]:
    tsp = np.asarray(tsp, dtype=float)
    if len(tsp) < 2:
        return []

    ignore_idx = _ignore_start_index(hours_to_ignore, dt_seconds)
    change_indices: list[int] = []

    for i in range(1, len(tsp)):
        if i < ignore_idx:
            continue
        if not np.isclose(tsp[i], tsp[i - 1]):
            change_indices.append(i)

    return change_indices


def _compute_mo_over_all_setpoint_changes(
    ti: np.ndarray,
    tsp: np.ndarray,
    dt_seconds: float,
    hours_to_ignore: float,
) -> float | None:
    ti = np.asarray(ti, dtype=float)
    tsp = np.asarray(tsp, dtype=float)

    if len(ti) == 0 or len(tsp) == 0 or len(ti) != len(tsp):
        return None

    change_indices = _find_setpoint_change_indices(
        tsp=tsp,
        dt_seconds=dt_seconds,
        hours_to_ignore=hours_to_ignore,
    )

    if len(change_indices) == 0:
        return None

    mo_values: list[float] = []

    for j, start_idx in enumerate(change_indices):
        end_idx = change_indices[j + 1] if j + 1 < len(change_indices) else len(tsp)
        ti_seg = ti[start_idx:end_idx]
        tsp_seg = tsp[start_idx:end_idx]

        if len(ti_seg) == 0:
            continue

        seg_err = np.abs(ti_seg - tsp_seg)
        if len(seg_err) > 0:
            mo_values.append(float(np.max(seg_err)))

    if len(mo_values) == 0:
        return None

    return float(np.max(mo_values))


def compute_evaluation_metrics(
    time_index: pd.DatetimeIndex,
    ti: np.ndarray,
    tsp: np.ndarray,
    phea_w: np.ndarray,
    dt_seconds: float = DT_SECONDS,
) -> dict[str, float | int | None]:
    ti = np.asarray(ti, dtype=float)
    tsp = np.asarray(tsp, dtype=float)
    phea_w = np.asarray(phea_w, dtype=float)

    start_idx = _ignore_start_index(IGNORE_FIRST_HOUR_FOR_COMFORT, dt_seconds)
    ti_c = ti[start_idx:]
    tsp_c = tsp[start_idx:]
    if len(ti_c) == 0:
        ti_c = ti
        tsp_c = tsp

    abs_err = np.abs(ti_c - tsp_c)
    mad = float(np.mean(abs_err))
    cvr = float(np.mean(abs_err > CVR_THRESHOLD_C))

    dt_hours = dt_seconds / 3600.0
    ce_kwh = float(np.sum(phea_w * dt_hours) / 1000.0)

    tariff = build_tariff_schedule(time_index)
    ec_usd = float(np.sum((phea_w / 1000.0) * dt_hours * tariff))

    seg_start, seg_end = _first_constant_setpoint_segment(tsp)
    ti_seg = ti[seg_start:seg_end]
    tsp_seg = tsp[seg_start:seg_end]

    rt_seconds = _compute_rt_seconds(ti_seg, tsp_seg, dt_seconds)
    st_seconds = _compute_st_seconds(ti_seg, tsp_seg, dt_seconds, SETTLING_BAND_C)
    settling_idx = None if st_seconds is None else int(round(st_seconds / dt_seconds))
    oc_count = _compute_oc_after_settling(ti_seg, tsp_seg, settling_idx)
    mo_c = _compute_mo_over_all_setpoint_changes(
        ti=ti,
        tsp=tsp,
        dt_seconds=dt_seconds,
        hours_to_ignore=IGNORE_FIRST_HOUR_FOR_MO,
    )

    return {
        "MAD_C": mad,
        "CVR_fraction": cvr,
        "CE_kWh": ce_kwh,
        "EC_USD": ec_usd,
        "RT_s": rt_seconds,
        "ST_s": st_seconds,
        "OC_count": oc_count,
        "MO_C": mo_c,
    }


def save_temperature_plot(
    output_path: Path,
    ta_array: np.ndarray,
    tsp_array: np.ndarray,
    ti_array: np.ndarray,
    title: str,
) -> None:
    hours = np.arange(len(ti_array)) / 60.0
    plt.figure(figsize=(12, 5))
    plt.plot(hours, ti_array, label="Indoor Ti")
    plt.plot(hours, tsp_array, "--", label="Setpoint")
    plt.plot(hours, ta_array, label="Outdoor Ta")
    if IGNORE_FIRST_HOUR_FOR_COMFORT > 0:
        plt.axvline(IGNORE_FIRST_HOUR_FOR_COMFORT, linestyle=":", linewidth=1.2, label="Warm-up cutoff")
    plt.xlabel("Hour")
    plt.ylabel("Temperature [°C]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_power_plot(
    output_path: Path,
    phea_array: np.ndarray,
    title: str,
) -> None:
    hours = np.arange(len(phea_array)) / 60.0
    plt.figure(figsize=(12, 4))
    plt.plot(hours, phea_array, label="Heater power")
    if IGNORE_FIRST_HOUR_FOR_COMFORT > 0:
        plt.axvline(IGNORE_FIRST_HOUR_FOR_COMFORT, linestyle=":", linewidth=1.2, label="Warm-up cutoff")
    plt.xlabel("Hour")
    plt.ylabel("Power [W]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the C3 proactive PID supervision demo.")
    parser.add_argument("--supervisor-mode", choices=["rules", "agent_auto", "agent_only"], default="agent_auto")
    parser.add_argument("--user-objective", default=DEFAULT_USER_OBJECTIVE)
    parser.add_argument("--room", default=DEFAULT_ROOM)
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effective_supervisor_mode = args.supervisor_mode
    supervisor_client = None
    if args.supervisor_mode in {"agent_auto", "agent_only"}:
        if check_ollama_connection():
            supervisor_client = create_ollama_client()
        elif args.supervisor_mode == "agent_only":
            raise RuntimeError("Ollama is unavailable, so agent_only mode cannot run.")
        else:
            effective_supervisor_mode = "rules"
            print("Ollama unavailable; using rule-based fallback for all C3 supervision updates.")

    r_hat, c_hat = load_rc_params()
    _, day2 = load_hourly_temperature_days()
    kp0, ki0, kd0, pid_source = load_c0_pid_parameters()

    hourly_ta_day2 = day2["Ta_C"].to_numpy(dtype=float)
    ta_day2 = expand_hourly_to_minute(hourly_ta_day2)
    time_day2 = build_minute_time_index("day_2")
    tsp_day2 = build_setpoint_schedule(time_day2)
    ti0_day2 = float(ta_day2[0])

    sim_day2, sup_log = simulate_proactive_pid_supervision(
        time_index=time_day2,
        ta_array=ta_day2,
        tsp_array=tsp_day2,
        r=r_hat,
        c=c_hat,
        ti0=ti0_day2,
        kp0=kp0,
        ki0=ki0,
        kd0=kd0,
        supervisor_mode=effective_supervisor_mode,
        user_objective=args.user_objective,
        room=args.room,
        city=args.city,
        user_id=args.user_id,
        llm_model=args.model,
        supervisor_client=supervisor_client,
    )

    metrics_day2 = compute_evaluation_metrics(
        time_index=time_day2,
        ti=sim_day2["Ti"],
        tsp=tsp_day2,
        phea_w=sim_day2["Phea"],
    )

    tariff_day2 = build_tariff_schedule(time_day2)
    day2_out = pd.DataFrame({
        "time": time_day2,
        "Ta_C": ta_day2,
        "Tsp_C": tsp_day2,
        "Ti_C": sim_day2["Ti"],
        "Phea_W": sim_day2["Phea"],
        "tariff_USD_per_kWh": tariff_day2,
        "error_C": sim_day2["err"],
        "integral_term": sim_day2["int"],
        "derivative_term": sim_day2["der"],
        "Kp": sim_day2["Kp"],
        "Ki": sim_day2["Ki"],
        "Kd": sim_day2["Kd"],
    })
    day2_out.to_csv(OUTPUT_DIR / "c3_day2_timeseries.csv", index=False)

    build_instruction_log().to_csv(OUTPUT_DIR / "c3_user_instruction_log.csv", index=False)
    sup_log.to_csv(OUTPUT_DIR / "c3_supervision_log.csv", index=False)

    save_temperature_plot(
        OUTPUT_DIR / "c3_day2_temperature_plot.png",
        ta_day2,
        tsp_day2,
        sim_day2["Ti"],
        "C3 proactive PID supervision temperature response on Day 2",
    )
    save_power_plot(
        OUTPUT_DIR / "c3_day2_power_plot.png",
        sim_day2["Phea"],
        "C3 proactive PID supervision heater power on Day 2",
    )

    (OUTPUT_DIR / "c3_day2_metrics.json").write_text(
        json.dumps(metrics_day2, indent=2),
        encoding="utf-8",
    )

    summary = {
        "method": "C3 proactive PID supervision with fixed setpoint schedule",
        "condition_definition": (
            "setpoint is fixed by the shared C1/C2 schedule; only PID parameters are adjusted "
            "every 5 minutes using forecasted outdoor temperature and user habits"
        ),
        "initial_pid_source": pid_source,
        "initial_pid_used": {"KP": kp0, "KI": ki0, "KD": kd0},
        "requested_supervisor_mode": args.supervisor_mode,
        "effective_supervisor_mode": effective_supervisor_mode,
        "room": args.room,
        "city": args.city,
        "user_id": args.user_id,
        "user_objective": args.user_objective,
        "llm_model": args.model if effective_supervisor_mode != "rules" else None,
        "supervision_interval_min": SUPERVISION_INTERVAL_MIN,
        "window_min": WINDOW_MIN,
        "forecast_horizon_min": FORECAST_HORIZON_MIN,
        "significant_forecast_delta_C": SIGNIFICANT_FORECAST_DELTA_C,
        "weather_source": "generated outdoor temperature data",
        "setpoint_adjustment_online": False,
        "cost_weight_adjustment_online": False,
        "pid_bounds": {
            "KP_MIN": KP_MIN, "KP_MAX": KP_MAX,
            "KI_MIN": KI_MIN, "KI_MAX": KI_MAX,
            "KD_MIN": KD_MIN, "KD_MAX": KD_MAX,
        },
        "setpoint_schedule": {
            "00:00-02:00": SP_DEFAULT_NIGHT,
            "02:00-03:00": SP_EVENT_02,
            "03:00-09:00": SP_DEFAULT_NIGHT,
            "09:00-18:00": SP_DAY,
            "18:00-19:00": SP_EVENT_18,
            "19:00-22:00": SP_EVENING,
            "22:00-24:00": SP_DEFAULT_NIGHT,
        },
        "instruction_log_file": str(OUTPUT_DIR / "c3_user_instruction_log.csv"),
        "supervision_log_file": str(OUTPUT_DIR / "c3_supervision_log.csv"),
        "num_supervision_updates": int(len(sup_log)),
        "decision_source_counts": sup_log["decision_source"].value_counts().to_dict() if len(sup_log) and "decision_source" in sup_log.columns else {},
        "decision_mode_counts": sup_log["decision_mode"].value_counts().to_dict() if len(sup_log) and "decision_mode" in sup_log.columns else {},
        "num_proactive_updates": int((sup_log["decision_mode"] == "proactive").sum()) if len(sup_log) else 0,
        "num_reactive_updates": int((sup_log["decision_mode"] == "reactive").sum()) if len(sup_log) else 0,
        "num_hold_updates": int((sup_log["decision_mode"] == "hold").sum()) if len(sup_log) else 0,
        "day2_evaluation_metrics": metrics_day2,
    }
    (OUTPUT_DIR / "c3_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("C3 proactive PID supervision run completed.")
    print(f"Initial C0 PID: KP={kp0}, KI={ki0}, KD={kd0}")
    print(f"PID source: {pid_source}")
    print(f"Number of supervision updates: {len(sup_log)}")
    if len(sup_log):
        print(f"Proactive updates: {(sup_log['decision_mode'] == 'proactive').sum()}")
        print(f"Reactive updates: {(sup_log['decision_mode'] == 'reactive').sum()}")
        print(f"Hold updates: {(sup_log['decision_mode'] == 'hold').sum()}")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
