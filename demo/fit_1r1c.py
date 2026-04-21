"""
fit_1r1c_from_cleaned_excel.py

Purpose
-------
Fit a 1R1C model from the cleaned insulated_concrete Excel file.

Input
-----
data/test_box_insulated_cleaned.xlsx

Outputs
-------
1. config/rc_fit_results.json                 -> fitted R and C values only
2. outputs/1r1c_results/rc_fit_metrics.json   -> fit metrics for the 1R1C model
3. outputs/1r1c_results/rc_fit_plot.png       -> measured vs fitted curve

The saved R and C can be reused in later experiments.

Usage
-----
python fit_1r1c_from_cleaned_excel.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


INPUT_XLSX = Path("data") / "test_box_insulated_cleaned.xlsx"
SHEET_NAME = "insulated_concrete"

CONFIG_DIR = Path("config")
OUTPUT_DIR = Path("outputs") / "1r1c_results"

PARAM_JSON = CONFIG_DIR / "rc_fit_results.json"
METRICS_JSON = OUTPUT_DIR / "rc_fit_metrics.json"
OUTPUT_PLOT = OUTPUT_DIR / "rc_fit_plot.png"


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cleaned_data(filepath: Path = INPUT_XLSX, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find cleaned Excel file: {path.resolve()}")

    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Ti", "Ta", "P", "dt"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "time" in df.columns:
        parsed = pd.to_datetime(df["time"], errors="coerce")
        if parsed.notna().any():
            df["time"] = parsed
            df = df.sort_values("time").reset_index(drop=True)

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 3:
        raise ValueError("Not enough valid rows for fitting.")

    return df


def simulate_1r1c_open_loop(
    Ta: np.ndarray,
    P: np.ndarray,
    dt: np.ndarray,
    Ti0: float,
    R: float,
    C: float,
) -> np.ndarray:
    """
    Discrete 1R1C model:
        Ti[k] = (a * Ti[k-1] + (1/R) * Ta[k] + P[k]) / (a + 1/R)
    where a = C / dt[k]
    """
    Ta = np.asarray(Ta, dtype=float)
    P = np.asarray(P, dtype=float)
    dt = np.asarray(dt, dtype=float)

    Ti = np.empty_like(Ta, dtype=float)
    Ti[0] = Ti0
    inv_r = 1.0 / R

    for k in range(1, len(Ti)):
        a = C / dt[k]
        Ti[k] = (a * Ti[k - 1] + inv_r * Ta[k] + P[k]) / (a + inv_r)

    return Ti


def fit_rc(df: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    Ta = df["Ta"].to_numpy(dtype=float)
    P = df["P"].to_numpy(dtype=float)
    dt = df["dt"].to_numpy(dtype=float)
    Ti_measured = df["Ti"].to_numpy(dtype=float)
    Ti0 = float(Ti_measured[0])

    xdata = np.vstack([Ta, P, dt]).T

    def model(x: np.ndarray, R: float, C: float) -> np.ndarray:
        Ta_local = x[:, 0]
        P_local = x[:, 1]
        dt_local = x[:, 2]
        return simulate_1r1c_open_loop(Ta_local, P_local, dt_local, Ti0, R, C)

    popt, _ = curve_fit(
        model,
        xdata,
        Ti_measured,
        p0=(0.1, 1e5),
        bounds=([1e-8, 1e-3], [1e3, 1e9]),
        maxfev=50000,
    )

    R_hat, C_hat = float(popt[0]), float(popt[1])
    Ti_fitted = simulate_1r1c_open_loop(Ta, P, dt, Ti0, R_hat, C_hat)
    return R_hat, C_hat, Ti_fitted


def summarize_fit(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 else math.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main() -> None:
    ensure_dirs()

    df = load_cleaned_data()
    R_hat, C_hat, Ti_fitted = fit_rc(df)

    Ti_measured = df["Ti"].to_numpy(dtype=float)
    fit_metrics = summarize_fit(Ti_measured, Ti_fitted)

    # Save only the parameters needed by later scripts
    params_for_later = {
        "R_hat_K_per_W": R_hat,
        "C_hat_J_per_K": C_hat,
    }
    PARAM_JSON.write_text(json.dumps(params_for_later, indent=2), encoding="utf-8")

    metrics_results = {
        "model": "1R1C",
        "input_file": str(INPUT_XLSX),
        "sheet_name": SHEET_NAME,
        "num_samples": int(len(df)),
        "metrics": {
            "MAE": fit_metrics["MAE"],
            "RMSE": fit_metrics["RMSE"],
            "R2": fit_metrics["R2"],
        },
        "parameter_file": str(PARAM_JSON),
        "plot_file": str(OUTPUT_PLOT),
    }
    METRICS_JSON.write_text(json.dumps(metrics_results, indent=2), encoding="utf-8")

    x = df["time"] if "time" in df.columns else np.arange(len(df))

    plt.figure(figsize=(12, 5))
    plt.plot(x, df["Ti"], label="Measured Ti")
    plt.plot(x, Ti_fitted, label="1R1C fitted Ti")
    plt.plot(x, df["Ta"], label="Ambient Ta")
    plt.xlabel("Time")
    plt.ylabel("Temperature [°C]")
    plt.title("1R1C fit from cleaned insulated_concrete data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

    print(f"Saved fitted parameters to: {PARAM_JSON.resolve()}")
    print(f"Saved fit metrics to: {METRICS_JSON.resolve()}")
    print(f"Saved plot to: {OUTPUT_PLOT.resolve()}")
    print("\nFitted parameters for later use:")
    for k, v in params_for_later.items():
        print(f"{k}: {v}")
    print("\nFit metrics:")
    for k, v in fit_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
