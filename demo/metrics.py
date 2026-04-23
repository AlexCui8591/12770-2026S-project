"""Demo Version A — metrics computation.

Input: a list of per-minute samples from a simulation run.
Each sample is a dict: {
    "t_min": int,              # minute index (0, 1, 2, ...)
    "indoor_temp": float,      # °C
    "setpoint": float,         # °C
    "control_signal": float,   # watts (0 to Q_max)
}

Output: 4 numbers:
    MAD  — Mean Absolute Deviation of indoor_temp from setpoint (°C)
    CE   — Cumulative Energy (kWh)
    RT   — Response Time: minutes until |temp - setpoint| first enters ±0.5°C band
    OC   — Oscillation Count: times temp trajectory crosses setpoint after settling

Usage:
    from demo.metrics import compute_metrics
    result = compute_metrics(samples)
    print(f"MAD={result['MAD']:.2f}°C  CE={result['CE']:.2f}kWh  "
          f"RT={result['RT']}min  OC={result['OC']}")
"""

from __future__ import annotations


COMFORT_BAND_C = 0.5  # ±0.5°C is considered "at setpoint"
STEP_SECONDS = 60.0   # one sample = one minute


def compute_metrics(samples: list[dict]) -> dict:
    """Given a list of per-minute samples, return the 4 demo metrics."""
    if not samples:
        return {"MAD": 0.0, "CE": 0.0, "RT": None, "OC": 0}

    # -- MAD: mean |indoor_temp - setpoint| --
    abs_errors = [abs(s["indoor_temp"] - s["setpoint"]) for s in samples]
    mad = sum(abs_errors) / len(abs_errors)

    # -- CE: cumulative energy in kWh.
    # control_signal is in watts; each sample covers STEP_SECONDS.
    # Energy_per_sample (Wh) = watts * seconds / 3600
    # kWh = Wh / 1000
    energy_wh = sum(s["control_signal"] * STEP_SECONDS / 3600.0 for s in samples)
    ce = energy_wh / 1000.0

    # -- RT: first minute at which |temp - setpoint| <= COMFORT_BAND_C --
    rt = None
    for s in samples:
        if abs(s["indoor_temp"] - s["setpoint"]) <= COMFORT_BAND_C:
            rt = s["t_min"]
            break
    # If never entered the band, RT = length of simulation (penalty)
    if rt is None:
        rt = samples[-1]["t_min"] + 1

    # -- OC: crossings of setpoint after the system first reached it --
    # Only count crossings AFTER settling (otherwise the initial approach
    # counts as a "crossing", which is not what OC is meant to measure).
    oc = 0
    settled = False
    prev_sign = 0
    for s in samples:
        err = s["indoor_temp"] - s["setpoint"]
        if not settled:
            if abs(err) <= COMFORT_BAND_C:
                settled = True
            continue
        sign = 1 if err > 0 else -1 if err < 0 else 0
        if prev_sign != 0 and sign != 0 and sign != prev_sign:
            oc += 1
        if sign != 0:
            prev_sign = sign

    return {
        "MAD": mad,
        "CE": ce,
        "RT": rt,
        "OC": oc,
    }


def format_metrics(result: dict, condition: str = "") -> str:
    """Pretty-print metrics as a one-liner for terminal output."""
    prefix = f"[{condition}] " if condition else ""
    return (
        f"{prefix}MAD={result['MAD']:.2f}°C  "
        f"CE={result['CE']:.3f}kWh  "
        f"RT={result['RT']}min  "
        f"OC={result['OC']}"
    )


if __name__ == "__main__":
    # Quick self-test with a synthetic trajectory
    import math
    samples = []
    setpoint = 22.0
    # Simulate: start cold at 18°C, approach setpoint exponentially, noisy
    for t in range(120):
        # exponential approach: temp = setpoint - 4 * exp(-t/20)
        temp = setpoint - 4.0 * math.exp(-t / 20.0)
        # Add a small oscillation after settling
        if t > 30:
            temp += 0.3 * math.sin((t - 30) / 5.0)
        # Control signal: simple P-like output
        u = max(0.0, min(3000.0, 500.0 * (setpoint - temp)))
        samples.append({
            "t_min": t,
            "indoor_temp": temp,
            "setpoint": setpoint,
            "control_signal": u,
        })

    result = compute_metrics(samples)
    print("Self-test on synthetic trajectory:")
    print(f"  120 samples, approach from 18°C toward 22°C with mild oscillation")
    print(f"  Expected: MAD small (~0.5), CE moderate, RT ~10-20 min, OC > 0")
    print("")
    print(format_metrics(result, "SELF_TEST"))
