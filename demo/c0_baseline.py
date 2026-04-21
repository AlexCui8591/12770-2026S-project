"""
run_c0_baseline_tuned_pid.py

C0 baseline:
- Day 1 is used only to tune fixed PID parameters
- Tuning method: grid search + normalized discrete cost
- Day 2 is used for the formal C0 baseline run
- Initial indoor temperature equals the initial outdoor temperature
- Heater power is saturated to [0, 14.25] W
- Outdoor temperature file is hourly, but simulation runs at 1-minute steps

Updates in this version:
- Comfort cost ignores the first 1 hour warm-up period
- Cost terms are normalized across the Day-1 grid-search pool
- KD is enabled and included in the grid search
- In evaluation metrics, MAD/CVR ignore the first 1 hour, and MO is taken as the maximum over all post-warmup setpoint-change transients

Added evaluation metrics from the project PDF:
- Thermal Comfort: MAD, CVR
- Energy Efficiency: CE, EC
- Responsiveness: RT, ST
- Stability: OC, MO

Expected inputs
---------------
- config/rc_fit_results.json
- data/synthetic_outdoor_temperature_two_days.csv

Outputs
-------
- outputs/c0_tuned_pid_outputs/c0_pid_tuning_day1.csv
- outputs/c0_tuned_pid_outputs/c0_day2_timeseries.csv
- outputs/c0_tuned_pid_outputs/c0_day2_temperature_plot.png
- outputs/c0_tuned_pid_outputs/c0_day2_power_plot.png
- outputs/c0_tuned_pid_outputs/c0_day2_metrics.json
- outputs/c0_tuned_pid_outputs/c0_run_summary.json

Usage
-----
python run_c0_baseline_tuned_pid.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RC_JSON = "config/rc_fit_results.json"
TEMP_CSV = "data/synthetic_outdoor_temperature_two_days.csv"

# Fixed setpoint schedule
SP_DAY = 38.0
SP_NIGHT = 40.0

# Simulation settings
DT_SECONDS = 60.0  # internal simulation step: 1 minute
PMAX = 14.25
DEADBAND = 0.2
IGNORE_FIRST_HOUR_FOR_COMFORT = 1.0
IGNORE_FIRST_HOUR_FOR_MO = 1.0

# Grid search space: full PID
KP_GRID = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
KI_GRID = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
KD_GRID = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]

# Normalized cost weights
W_COMFORT = 5.0
W_ENERGY = 1.0
W_RESPONSE = 10.0

# Evaluation thresholds from the PDF
CVR_THRESHOLD_C = 1.5
SETTLING_BAND_C = 0.5

# Simple built-in TOU tariff for EC metric ($/kWh)
TARIFF_OFFPEAK = 0.12
TARIFF_ONPEAK = 0.20
ONPEAK_HOURS = set(range(16, 21))  # 16:00-20:59

OUTPUT_DIR = Path("outputs/c0_tuned_pid_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def setpoint_from_timestamp(ts: pd.Timestamp, sp_day: float = SP_DAY, sp_night: float = SP_NIGHT) -> float:
    return sp_day if 9 <= ts.hour < 18 else sp_night


def build_setpoint_schedule(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([setpoint_from_timestamp(ts) for ts in time_index], dtype=float)


def build_tariff_schedule(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.array(
        [TARIFF_ONPEAK if ts.hour in ONPEAK_HOURS else TARIFF_OFFPEAK for ts in time_index],
        dtype=float,
    )


def _ignore_start_index(hours_to_ignore: float, dt_seconds: float) -> int:
    return max(0, int(round(hours_to_ignore * 3600.0 / dt_seconds)))


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

    # Saturate power to [0, PMAX]
    u = float(np.clip(u_unsat, 0.0, pmax))

    # Simple anti-windup
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


def simulate_closed_loop_pid_1r1c(
    time_index: pd.DatetimeIndex,
    ta_array: np.ndarray,
    tsp_array: np.ndarray,
    r: float,
    c: float,
    ti0: float,
    kp: float,
    ki: float,
    kd: float,
    pmax: float = PMAX,
    deadband: float = DEADBAND,
    dt_seconds: float = DT_SECONDS,
) -> dict[str, np.ndarray]:
    n = len(time_index)
    bldg = Building1R1C(ti=float(ti0), r=float(r), c=float(c))

    ti_hist = np.zeros(n, dtype=float)
    phea_hist = np.zeros(n, dtype=float)
    err_hist = np.zeros(n, dtype=float)
    int_hist = np.zeros(n, dtype=float)
    der_hist = np.zeros(n, dtype=float)

    integral = 0.0
    error_prev = 0.0

    for k in range(n):
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

        if k < n - 1:
            bldg.next_step(
                p_hea=p_k,
                ta=float(ta_array[k + 1]),
                dt_seconds=dt_seconds,
            )

        error_prev = error_now

    return {
        "Ti": ti_hist,
        "Phea": phea_hist,
        "err": err_hist,
        "int": int_hist,
        "der": der_hist,
    }


def discrete_normalized_cost(
    ti: np.ndarray,
    tsp: np.ndarray,
    u: np.ndarray,
    norm_scales: dict[str, float],
    dt_seconds: float = DT_SECONDS,
    w_comfort: float = W_COMFORT,
    w_energy: float = W_ENERGY,
    w_response: float = W_RESPONSE,
) -> dict[str, float]:
    """
    Normalized discrete cost:
        J = w_c * (J_comfort_raw / S_comfort)
          + w_e * (J_energy_raw  / S_energy)
          + w_r * (J_response_raw / S_response)

    Notes:
    - Comfort ignores the first IGNORE_FIRST_HOUR_FOR_COMFORT hours.
    - Energy and response are kept over the whole tuning window.
    """
    ti = np.asarray(ti, dtype=float)
    tsp = np.asarray(tsp, dtype=float)
    u = np.asarray(u, dtype=float)

    start_idx = _ignore_start_index(IGNORE_FIRST_HOUR_FOR_COMFORT, dt_seconds)

    ti_c = ti[start_idx:]
    tsp_c = tsp[start_idx:]
    if len(ti_c) == 0:
        ti_c = ti
        tsp_c = tsp

    err = ti_c - tsp_c
    comfort = float(np.sum((err ** 2) * dt_seconds))
    energy = float(np.sum(u * dt_seconds))

    if len(u) > 1:
        du = np.diff(u)
        response = float(np.sum(((du / dt_seconds) ** 2) * dt_seconds))
    else:
        response = 0.0

    comfort_norm = comfort / max(norm_scales["comfort"], 1e-12)
    energy_norm = energy / max(norm_scales["energy"], 1e-12)
    response_norm = response / max(norm_scales["response"], 1e-12)

    total = (
        w_comfort * comfort_norm
        + w_energy * energy_norm
        + w_response * response_norm
    )

    return {
        "J_total": total,
        "J_comfort_raw": comfort,
        "J_energy_raw": energy,
        "J_response_raw": response,
        "J_comfort_norm": comfort_norm,
        "J_energy_norm": energy_norm,
        "J_response_norm": response_norm,
    }


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

    crossings = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])
    return int(crossings)



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
    """
    Metrics defined from the project PDF.

    Updates in this version:
    - MAD and CVR ignore the first 1 hour warm-up period.
    - MO ignores the first 1 hour of the initial constant-setpoint segment.
    - CE/EC still use the full simulation horizon.
    - RT/ST/OC still evaluate cold-start behavior on the initial constant-setpoint segment.
    """
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


def _build_normalization_scales(raw_records: list[dict[str, float]]) -> dict[str, float]:
    comfort_vals = np.array([r["J_comfort_raw"] for r in raw_records], dtype=float)
    energy_vals = np.array([r["J_energy_raw"] for r in raw_records], dtype=float)
    response_vals = np.array([r["J_response_raw"] for r in raw_records], dtype=float)

    def robust_scale(x: np.ndarray) -> float:
        positive = x[x > 0]
        if len(positive) == 0:
            return 1.0
        return float(np.median(positive))

    return {
        "comfort": robust_scale(comfort_vals),
        "energy": robust_scale(energy_vals),
        "response": robust_scale(response_vals),
    }


def tune_pid_on_day1(
    time_index: pd.DatetimeIndex,
    ta_array: np.ndarray,
    tsp_array: np.ndarray,
    r_hat: float,
    c_hat: float,
    ti0: float,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, float]]:
    raw_records = []
    sim_cache: list[dict[str, object]] = []

    for kp in KP_GRID:
        for ki in KI_GRID:
            for kd in KD_GRID:
                sim = simulate_closed_loop_pid_1r1c(
                    time_index=time_index,
                    ta_array=ta_array,
                    tsp_array=tsp_array,
                    r=r_hat,
                    c=c_hat,
                    ti0=ti0,
                    kp=kp,
                    ki=ki,
                    kd=kd,
                )

                start_idx = _ignore_start_index(IGNORE_FIRST_HOUR_FOR_COMFORT, DT_SECONDS)
                ti_c = sim["Ti"][start_idx:]
                tsp_c = tsp_array[start_idx:]
                if len(ti_c) == 0:
                    ti_c = sim["Ti"]
                    tsp_c = tsp_array

                err = ti_c - tsp_c
                comfort = float(np.sum((err ** 2) * DT_SECONDS))
                energy = float(np.sum(sim["Phea"] * DT_SECONDS))

                if len(sim["Phea"]) > 1:
                    du = np.diff(sim["Phea"])
                    response = float(np.sum(((du / DT_SECONDS) ** 2) * DT_SECONDS))
                else:
                    response = 0.0

                raw_record = {
                    "KP": float(kp),
                    "KI": float(ki),
                    "KD": float(kd),
                    "J_comfort_raw": comfort,
                    "J_energy_raw": energy,
                    "J_response_raw": response,
                }
                raw_records.append(raw_record)
                sim_cache.append({"params": (kp, ki, kd), "sim": sim})

    norm_scales = _build_normalization_scales(raw_records)

    final_records = []
    for rec, cached in zip(raw_records, sim_cache):
        sim = cached["sim"]
        cost = discrete_normalized_cost(
            ti=sim["Ti"],
            tsp=tsp_array,
            u=sim["Phea"],
            norm_scales=norm_scales,
        )
        final_records.append({
            "KP": rec["KP"],
            "KI": rec["KI"],
            "KD": rec["KD"],
            **cost,
        })

    table = pd.DataFrame(final_records).sort_values("J_total").reset_index(drop=True)
    best = table.iloc[0].to_dict()
    return best, table, norm_scales


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


def main() -> None:
    r_hat, c_hat = load_rc_params()
    day1, day2 = load_hourly_temperature_days()

    # Day 1: only for tuning PID parameters
    hourly_ta_day1 = day1["Ta_C"].to_numpy(dtype=float)
    ta_day1 = expand_hourly_to_minute(hourly_ta_day1)
    time_day1 = build_minute_time_index("day_1")
    tsp_day1 = build_setpoint_schedule(time_day1)

    # Requirement: initial indoor temperature equals initial outdoor temperature
    ti0_day1 = float(ta_day1[0])

    best_pid, tuning_table, norm_scales = tune_pid_on_day1(
        time_index=time_day1,
        ta_array=ta_day1,
        tsp_array=tsp_day1,
        r_hat=r_hat,
        c_hat=c_hat,
        ti0=ti0_day1,
    )

    best_kp = float(best_pid["KP"])
    best_ki = float(best_pid["KI"])
    best_kd = float(best_pid["KD"])

    tuning_table.to_csv(OUTPUT_DIR / "c0_pid_tuning_day1.csv", index=False)

    # Day 2: formal C0 baseline run with fixed tuned PID
    hourly_ta_day2 = day2["Ta_C"].to_numpy(dtype=float)
    ta_day2 = expand_hourly_to_minute(hourly_ta_day2)
    time_day2 = build_minute_time_index("day_2")
    tsp_day2 = build_setpoint_schedule(time_day2)
    ti0_day2 = float(ta_day2[0])

    sim_day2 = simulate_closed_loop_pid_1r1c(
        time_index=time_day2,
        ta_array=ta_day2,
        tsp_array=tsp_day2,
        r=r_hat,
        c=c_hat,
        ti0=ti0_day2,
        kp=best_kp,
        ki=best_ki,
        kd=best_kd,
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
    })
    day2_out.to_csv(OUTPUT_DIR / "c0_day2_timeseries.csv", index=False)

    save_temperature_plot(
        OUTPUT_DIR / "c0_day2_temperature_plot.png",
        ta_day2,
        tsp_day2,
        sim_day2["Ti"],
        "C0 baseline temperature response on Day 2",
    )
    save_power_plot(
        OUTPUT_DIR / "c0_day2_power_plot.png",
        sim_day2["Phea"],
        "C0 baseline heater power on Day 2",
    )

    (OUTPUT_DIR / "c0_day2_metrics.json").write_text(
        json.dumps(metrics_day2, indent=2),
        encoding="utf-8",
    )

    summary = {
        "method": "C0 baseline with tuned fixed PID",
        "day1_role": "PID tuning only",
        "day2_role": "formal C0 baseline run",
        "initial_condition_rule": "initial indoor temperature equals initial outdoor temperature",
        "pid_tuning_method": "grid search + normalized discrete cost",
        "fixed_pid_after_tuning": {
            "KP": best_kp,
            "KI": best_ki,
            "KD": best_kd,
        },
        "power_limit_W": PMAX,
        "deadband_C": DEADBAND,
        "warmup_ignored_for_cost_hours": IGNORE_FIRST_HOUR_FOR_COMFORT,
        "warmup_ignored_for_MO_hours": IGNORE_FIRST_HOUR_FOR_MO,
        "setpoint_schedule": {
            "09:00-18:00": SP_DAY,
            "other_hours": SP_NIGHT,
        },
        "cost_weights_on_normalized_terms": {
            "W_COMFORT": W_COMFORT,
            "W_ENERGY": W_ENERGY,
            "W_RESPONSE": W_RESPONSE,
        },
        "normalization_scales": norm_scales,
        "tariff_schedule_for_EC": {
            "offpeak_USD_per_kWh": TARIFF_OFFPEAK,
            "onpeak_USD_per_kWh": TARIFF_ONPEAK,
            "onpeak_hours": sorted(list(ONPEAK_HOURS)),
        },
        "plant_parameters": {
            "R_hat_K_per_W": r_hat,
            "C_hat_J_per_K": c_hat,
        },
        "best_day1_cost": {
            "J_total": float(best_pid["J_total"]),
            "J_comfort_raw": float(best_pid["J_comfort_raw"]),
            "J_energy_raw": float(best_pid["J_energy_raw"]),
            "J_response_raw": float(best_pid["J_response_raw"]),
            "J_comfort_norm": float(best_pid["J_comfort_norm"]),
            "J_energy_norm": float(best_pid["J_energy_norm"]),
            "J_response_norm": float(best_pid["J_response_norm"]),
        },
        "day2_evaluation_metrics": metrics_day2,
    }
    (OUTPUT_DIR / "c0_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("C0 tuned-PID baseline completed.")
    print(f"Chosen PID parameters: KP={best_kp}, KI={best_ki}, KD={best_kd}")
    print("Normalization scales used:")
    for k, v in norm_scales.items():
        print(f"  {k}: {v}")
    print("Day 2 evaluation metrics:")
    for k, v in metrics_day2.items():
        print(f"  {k}: {v}")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
