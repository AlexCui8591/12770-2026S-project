import requests
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression

BASE_URL = "https://api.boptest.net"
TESTCASE = "testcase1"

START_TIME = 0
WARMUP_PERIOD = 12 * 3600
SIM_DAYS = 2
STEP = 300

np.random.seed(42)


def unwrap_json_response(r):
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "payload" in data:
        return data["payload"]
    return data


def api_get(path, timeout=60):
    r = requests.get(f"{BASE_URL}{path}", timeout=timeout)
    return unwrap_json_response(r)


def api_post(path, json_data=None, timeout=120):
    r = requests.post(f"{BASE_URL}{path}", json=json_data, timeout=timeout)
    return unwrap_json_response(r)


def api_put(path, json_data=None, timeout=60):
    r = requests.put(f"{BASE_URL}{path}", json=json_data, timeout=timeout)
    return unwrap_json_response(r)


def create_testcase(testcase_name, max_retries=3, timeout=60):
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Attempt {attempt}/{max_retries}] Creating testcase '{testcase_name}'...")
            t0 = time.time()

            r = requests.post(
                f"{BASE_URL}/testcases/{testcase_name}/select",
                timeout=timeout,
            )

            data = unwrap_json_response(r)
            print(f"Create returned in {time.time() - t0:.1f}s")

            if isinstance(data, dict) and "testid" in data:
                return data["testid"]

            raise RuntimeError(f"Unexpected response: {data}")

        except Exception as e:
            last_err = e
            print(f"Create failed: {e}")

            if attempt < max_retries:
                print("Wait 10s, then retry...")
                time.sleep(10)

    raise RuntimeError(
        f"Failed to create testcase after {max_retries} attempts. Last error: {last_err}"
    )


def stop_testcase(testid):
    try:
        api_put(f"/stop/{testid}", timeout=30)
        print("\nStopped testcase.")
    except Exception as e:
        print("\nWarning: failed to stop testcase:", e)


def get_measurements(testid):
    return api_get(f"/measurements/{testid}")


def get_inputs(testid):
    return api_get(f"/inputs/{testid}")


def get_forecast_points(testid):
    return api_get(f"/forecast_points/{testid}")


def get_forecast(testid, point_names, horizon, interval):
    return api_put(
        f"/forecast/{testid}",
        {
            "point_names": point_names,
            "horizon": horizon,
            "interval": interval,
        },
        timeout=120,
    )


def set_step(testid, step_seconds):
    return api_put(
        f"/step/{testid}",
        {"step": step_seconds},
    )


def initialize(testid):
    return api_put(
        f"/initialize/{testid}",
        {
            "start_time": START_TIME,
            "warmup_period": WARMUP_PERIOD,
        },
    )


def advance(testid, control=None):
    payload = control if control else {}
    return api_post(f"/advance/{testid}", payload)


def find_name(candidates, names):
    for c in candidates:
        for n in names:
            if c.lower() in n.lower():
                return n
    return None


def infer_vars(meas, inputs, forecast_points):
    meas_names = list(meas.keys())
    input_names = list(inputs.keys())
    fcst_names = list(forecast_points.keys())

    Ti = find_name(["TRooAir", "TZon", "reaT"], meas_names)
    P = find_name(["PHeaCoo", "PHea", "PCoo", "power"], meas_names)
    u = find_name(["oveAct_u", "oveHea_u", "oveTSet_u", "_u"], input_names)
    To = find_name(["TDryBul", "TOut", "Out", "wea"], fcst_names)

    return Ti, To, P, u


def build_control(u_name, val):
    if u_name is None:
        return {}

    base = u_name[:-2] if u_name.endswith("_u") else u_name

    return {
        f"{base}_u": float(val),
        f"{base}_activate": 1.0,
    }


def convert_kelvin_to_celsius_if_needed(df):
    if df["Ti"].mean() > 100:
        df["Ti"] = df["Ti"] - 273.15
    if df["To"].mean() > 100:
        df["To"] = df["To"] - 273.15
    return df


def fit_with_auto_heating_cooling(df):
    df = df.copy()

    df["dTi"] = df["Ti_next"] - df["Ti"]

    X_env = (df["To"].values - df["Ti"].values).reshape(-1, 1)
    y = df["dTi"].values

    env_model = LinearRegression(fit_intercept=True).fit(X_env, y)
    a_env = float(env_model.coef_[0])
    c_env = float(env_model.intercept_)

    df["weather_effect"] = a_env * (df["To"] - df["Ti"]) + c_env
    df["residual_effect"] = df["dTi"] - df["weather_effect"]

    df["P_mode"] = np.where(df["residual_effect"] >= 0, "heating", "cooling")
    df["P_signed"] = np.where(df["residual_effect"] >= 0, df["P"], -df["P"])

    X = np.column_stack([
        df["To"].values - df["Ti"].values,
        df["P_signed"].values,
    ])

    model = LinearRegression(fit_intercept=True).fit(X, y)

    a, b = model.coef_
    c = model.intercept_

    dt_mean = df["dt"].mean()

    a_used = abs(float(a))
    b_used = abs(float(b))

    R_est = b_used / a_used
    C_est = dt_mean / b_used

    print("\n===== Auto heating/cooling 1R1C Result =====")
    print("Model: Ti_next - Ti = a*(To - Ti) + b*P_signed + c")
    print("a_raw =", a)
    print("b_raw =", b)
    print("c =", c)
    print("a_used =", a_used)
    print("b_used =", b_used)
    print("R =", R_est)
    print("C =", C_est)

    print("\n===== Mode counts =====")
    print(df["P_mode"].value_counts())

    df.to_csv("boptest_data_with_mode.csv", index=False)

    pd.DataFrame([{
        "a_raw": a,
        "b_raw": b,
        "c": c,
        "a_used": a_used,
        "b_used": b_used,
        "R": R_est,
        "C": C_est,
        "dt_mean": dt_mean,
        "heating_count": int((df["P_mode"] == "heating").sum()),
        "cooling_count": int((df["P_mode"] == "cooling").sum()),
    }]).to_csv("boptest_auto_mode_rc_params.csv", index=False)

    print("\nSaved boptest_data_with_mode.csv")
    print("Saved boptest_auto_mode_rc_params.csv")

    return df, a_used, b_used, c, R_est, C_est


def main():
    print("Listing testcases...")
    print(api_get("/testcases", timeout=30))

    testid = None

    try:
        testid = create_testcase(TESTCASE, max_retries=3, timeout=120)
        print("\nTestid:", testid)

        meas = get_measurements(testid)
        inputs = get_inputs(testid)
        forecast_points = get_forecast_points(testid)

        print("\n=== Measurements ===")
        for k in meas.keys():
            print(k)

        print("\n=== Inputs ===")
        for k in inputs.keys():
            print(k)

        print("\n=== Forecast points ===")
        for k in forecast_points.keys():
            print(k)

        Ti, To, P, u = infer_vars(meas, inputs, forecast_points)

        print("\nDetected:")
        print("Ti:", Ti)
        print("To:", To)
        print("P:", P)
        print("u:", u)

        if Ti is None or To is None:
            raise RuntimeError("Could not identify Ti or To from this testcase.")

        print("\nSetting BOPTEST step...")
        set_step(testid, STEP)

        print("\nInitializing...")
        initialize(testid)

        rows = []
        n_steps = int(SIM_DAYS * 24 * 3600 / STEP)
        current_u = 0.0

        print("\nFetching outdoor temperature forecast once...")
        forecast = get_forecast(
            testid,
            point_names=[To],
            horizon=n_steps * STEP + STEP,
            interval=STEP,
        )
        To_series = forecast[To]

        print(f"\nRunning simulation for {SIM_DAYS * 24} hours...")
        print(f"STEP = {STEP} seconds")
        print(f"Total steps = {n_steps}")

        for k in range(n_steps):
            To_now = To_series[k] if k < len(To_series) else np.nan

            if u:
                if k % 12 == 0:
                    current_u = np.random.uniform(0.0, 1.0)
                ctrl = build_control(u, current_u)
            else:
                ctrl = {}
                current_u = np.nan

            y = advance(testid, ctrl)

            rows.append({
                "time": y.get("time", np.nan),
                "Ti": y.get(Ti, np.nan),
                "To": To_now,
                "P": y.get(P, np.nan) if P else np.nan,
                "u": current_u,
            })

            if (k + 1) % 100 == 0 or (k + 1) == n_steps:
                print(f"step {k + 1}/{n_steps}")

        df = pd.DataFrame(rows)

        if df["P"].isna().all():
            print("No power signal found, using control u as proxy P.")
            df["P"] = df["u"]

        df = df.dropna(subset=["Ti", "To", "P"]).reset_index(drop=True)
        df = convert_kelvin_to_celsius_if_needed(df)

        df["dt"] = df["time"].diff().fillna(STEP)
        df.loc[df["dt"] <= 0, "dt"] = STEP
        df["sim_hour"] = (df["time"] - df["time"].iloc[0]) / 3600.0

        df["Ti_next"] = df["Ti"].shift(-1)
        df = df.dropna().reset_index(drop=True)

        df.to_csv("boptest_data.csv", index=False)
        print("\nSaved boptest_data.csv")

        print("\n===== Time check =====")
        print("mean dt =", df["dt"].mean())
        print("simulation hours =", df["sim_hour"].iloc[-1])

        fit_with_auto_heating_cooling(df)

    finally:
        if testid is not None:
            stop_testcase(testid)


if __name__ == "__main__":
    main()