"""Scenario builders for the formal HVAC experiment suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demo.generate_two_days_temperature import generate_one_day

try:
    import yaml
except ImportError:  # pragma: no cover - fallback is exercised when PyYAML is absent.
    yaml = None


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG_PATH = ROOT_DIR / "config" / "experiment.yaml"

DEFAULT_ROOM = "bedroom"
SUPERVISION_INTERVAL_STEPS = 5
BASE_DATE = "2026-04-02"
DAY2_SEED_OFFSET = 10_000

# The current demo thermal model is calibrated around the existing heating
# setpoint scale used by C0/C1/C2 (roughly 38-41 C), not the conceptual
# 21-22 C wording in the report config.
BASELINE_SP_DAY = 38.0
BASELINE_SP_NIGHT = 40.0
TARGET_SP_DEFAULT_NIGHT = 39.0
TARGET_SP_DAY = 38.0
TARGET_SP_EVENING = 40.0
TARGET_SP_EVENT = 41.0
TARGET_SP_LOW = 38.0

TARIFF_OFFPEAK = 0.12
TARIFF_ONPEAK = 0.20
ONPEAK_HOURS = set(range(16, 21))


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    name: str
    llm_enabled: bool
    supervision_enabled: bool = False
    mode: str | None = None
    model: str | None = None
    supervision_interval_steps: int = SUPERVISION_INTERVAL_STEPS


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    name: str
    command: str
    duration_steps: int
    start_minute: int
    disturbance: str | None = None
    failure_injection: bool = False
    failure_rate: float = 0.0


@dataclass
class ScenarioRun:
    scenario: ScenarioSpec
    seed: int
    room: str
    time_index: pd.DatetimeIndex
    outdoor_temp_c: np.ndarray
    baseline_setpoint_c: np.ndarray
    target_setpoint_c: np.ndarray
    tariff_usd_per_kwh: np.ndarray
    user_objective: str
    forced_rule_steps: set[int] = field(default_factory=set)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSettings:
    runs_per_condition: int
    seeds: tuple[int, ...]
    conditions: dict[str, ConditionSpec]
    scenarios: dict[str, ScenarioSpec]


DEFAULT_CONDITIONS: dict[str, ConditionSpec] = {
    "C0": ConditionSpec(
        id="C0",
        name="Fixed PID Baseline",
        llm_enabled=False,
        supervision_enabled=False,
    ),
    "C1": ConditionSpec(
        id="C1",
        name="LLM Setpoint-Only",
        llm_enabled=True,
        supervision_enabled=False,
    ),
    "C2": ConditionSpec(
        id="C2",
        name="LLM-in-the-Loop (Reactive)",
        llm_enabled=True,
        supervision_enabled=True,
        mode="reactive",
        model="qwen2.5:7b",
    ),
    "C3": ConditionSpec(
        id="C3",
        name="LLM-in-the-Loop (Proactive)",
        llm_enabled=True,
        supervision_enabled=True,
        mode="proactive",
        model="qwen2.5:7b",
    ),
}


DEFAULT_SCENARIOS: dict[str, ScenarioSpec] = {
    "S1": ScenarioSpec(
        id="S1",
        name="Steady-State Tracking",
        command="Set the bedroom to 22C.",
        duration_steps=120,
        start_minute=8 * 60,
    ),
    "S2": ScenarioSpec(
        id="S2",
        name="Weather Disturbance",
        command="Keep the room comfortable.",
        duration_steps=180,
        start_minute=6 * 60,
        disturbance="10C outdoor drop at step 30",
    ),
    "S3": ScenarioSpec(
        id="S3",
        name="Cost Optimization",
        command="Keep it warm but minimize electricity costs.",
        duration_steps=720,
        start_minute=9 * 60,
        disturbance="3x peak tariff mid-period",
    ),
    "S4": ScenarioSpec(
        id="S4",
        name="Occupancy Adaptation",
        command="Guests coming at 6 PM, make it comfortable.",
        duration_steps=360,
        start_minute=15 * 60,
    ),
    "S5": ScenarioSpec(
        id="S5",
        name="Multi-Disturbance Stress",
        command="Balance comfort and cost for 24 hours.",
        duration_steps=1440,
        start_minute=0,
    ),
    "S6": ScenarioSpec(
        id="S6",
        name="LLM Failure Resilience",
        command="Keep the room at 21C.",
        duration_steps=240,
        start_minute=8 * 60,
        failure_injection=True,
        failure_rate=0.20,
    ),
}


def load_experiment_settings(config_path: Path = EXPERIMENT_CONFIG_PATH) -> ExperimentSettings:
    conditions = dict(DEFAULT_CONDITIONS)
    scenarios = dict(DEFAULT_SCENARIOS)
    runs_per_condition = 5
    seeds = (42, 123, 456, 789, 1024)

    if yaml is None or not config_path.exists():
        return ExperimentSettings(
            runs_per_condition=runs_per_condition,
            seeds=seeds,
            conditions=conditions,
            scenarios=scenarios,
        )

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    experiment = data.get("experiment", {})
    runs_per_condition = int(experiment.get("runs_per_condition", runs_per_condition))
    raw_seeds = experiment.get("seeds", seeds)
    seeds = tuple(int(seed) for seed in raw_seeds)

    for key in ("c0", "c1", "c2", "c3"):
        raw_condition = data.get(key, {})
        condition_id = key.upper()
        default = conditions[condition_id]
        conditions[condition_id] = ConditionSpec(
            id=condition_id,
            name=str(raw_condition.get("name", default.name)),
            llm_enabled=bool(raw_condition.get("llm_enabled", default.llm_enabled)),
            supervision_enabled=bool(
                raw_condition.get("supervision_enabled", default.supervision_enabled)
            ),
            mode=raw_condition.get("mode", default.mode),
            model=raw_condition.get("model", default.model),
            supervision_interval_steps=int(
                raw_condition.get(
                    "supervision_interval_steps",
                    default.supervision_interval_steps,
                )
            ),
        )

    raw_scenarios = experiment.get("scenarios", [])
    for raw in raw_scenarios:
        scenario_id = str(raw.get("id", "")).upper()
        if scenario_id not in scenarios:
            continue
        default = scenarios[scenario_id]
        scenarios[scenario_id] = ScenarioSpec(
            id=scenario_id,
            name=str(raw.get("name", default.name)),
            command=str(raw.get("command", default.command)),
            duration_steps=int(raw.get("duration_steps", default.duration_steps)),
            start_minute=default.start_minute,
            disturbance=raw.get("disturbance", default.disturbance),
            failure_injection=bool(raw.get("failure_injection", default.failure_injection)),
            failure_rate=float(raw.get("failure_rate", default.failure_rate)),
        )

    return ExperimentSettings(
        runs_per_condition=runs_per_condition,
        seeds=seeds,
        conditions=conditions,
        scenarios=scenarios,
    )


def default_condition_ids() -> list[str]:
    return ["C0", "C1", "C2", "C3"]


def default_scenario_ids() -> list[str]:
    return ["S1", "S2", "S3", "S4", "S5", "S6"]


def resolve_seed_list(configured_seeds: tuple[int, ...], requested_runs: int) -> list[int]:
    if requested_runs <= len(configured_seeds):
        return [int(seed) for seed in configured_seeds[:requested_runs]]

    resolved = [int(seed) for seed in configured_seeds]
    next_seed = max(resolved) + 1 if resolved else 1
    while len(resolved) < requested_runs:
        resolved.append(next_seed)
        next_seed += 1
    return resolved


def build_tuning_setpoint_schedule(time_index: pd.DatetimeIndex) -> np.ndarray:
    values = []
    for ts in time_index:
        if 9 <= ts.hour < 18:
            values.append(BASELINE_SP_DAY)
        else:
            values.append(BASELINE_SP_NIGHT)
    return np.asarray(values, dtype=float)


def build_baseline_setpoint_schedule(time_index: pd.DatetimeIndex) -> np.ndarray:
    values = []
    for ts in time_index:
        if 9 <= ts.hour < 18:
            values.append(BASELINE_SP_DAY)
        else:
            values.append(BASELINE_SP_NIGHT)
    return np.asarray(values, dtype=float)


def build_scenario_run(
    scenario_id: str,
    seed: int,
    *,
    room: str = DEFAULT_ROOM,
    supervision_interval_steps: int = SUPERVISION_INTERVAL_STEPS,
    settings: ExperimentSettings | None = None,
) -> ScenarioRun:
    if settings is None:
        settings = load_experiment_settings()
    scenario = settings.scenarios[scenario_id.upper()]

    day2_df = generate_one_day(seed + DAY2_SEED_OFFSET, "day_2")
    outdoor_day = np.repeat(day2_df["Ta_C"].to_numpy(dtype=float), 60)

    start_minute = scenario.start_minute
    end_minute = start_minute + scenario.duration_steps
    if end_minute > len(outdoor_day):
        raise ValueError(
            f"Scenario {scenario.id} exceeds the generated 24-hour profile "
            f"({end_minute} > {len(outdoor_day)})."
        )

    outdoor = outdoor_day[start_minute:end_minute].astype(float, copy=True)
    start_hour = start_minute // 60
    start_minute_in_hour = start_minute % 60
    time_index = pd.date_range(
        start=f"{BASE_DATE} {start_hour:02d}:{start_minute_in_hour:02d}:00",
        periods=scenario.duration_steps,
        freq="min",
    )

    notes: dict[str, Any] = {
        "scenario_name": scenario.name,
        "command": scenario.command,
        "duration_steps": scenario.duration_steps,
        "start_minute": start_minute,
        "working_setpoint_scale": "demo_heating_scale_38_to_41C",
    }

    _apply_weather_disturbance(scenario.id, outdoor, notes)
    tariff = _build_tariff_schedule(time_index)
    _apply_tariff_modifiers(scenario.id, tariff, notes)
    baseline_setpoint = build_baseline_setpoint_schedule(time_index)
    target_setpoint = _build_target_setpoint_schedule(scenario.id, time_index, tariff)
    forced_rule_steps = _build_failure_injection_steps(
        scenario=scenario,
        seed=seed,
        supervision_interval_steps=supervision_interval_steps,
    )
    if forced_rule_steps:
        notes["failure_injection_steps"] = sorted(forced_rule_steps)
        notes["failure_injection_count"] = len(forced_rule_steps)

    return ScenarioRun(
        scenario=scenario,
        seed=seed,
        room=room,
        time_index=time_index,
        outdoor_temp_c=outdoor,
        baseline_setpoint_c=baseline_setpoint,
        target_setpoint_c=target_setpoint,
        tariff_usd_per_kwh=tariff,
        user_objective=scenario.command,
        forced_rule_steps=forced_rule_steps,
        notes=notes,
    )


def _build_tariff_schedule(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(
        [
            TARIFF_ONPEAK if ts.hour in ONPEAK_HOURS else TARIFF_OFFPEAK
            for ts in time_index
        ],
        dtype=float,
    )


def _build_target_setpoint_schedule(
    scenario_id: str,
    time_index: pd.DatetimeIndex,
    tariff: np.ndarray,
) -> np.ndarray:
    n = len(time_index)
    scenario_id = scenario_id.upper()

    if scenario_id in {"S1", "S2"}:
        return np.full(n, TARGET_SP_DEFAULT_NIGHT, dtype=float)

    if scenario_id == "S3":
        setpoint = np.full(n, TARGET_SP_DEFAULT_NIGHT, dtype=float)
        expensive = tariff >= (TARIFF_ONPEAK * 2.5)
        setpoint[expensive] = TARGET_SP_LOW
        return setpoint

    if scenario_id == "S4":
        return np.asarray(
            [TARGET_SP_DAY if ts.hour < 18 else TARGET_SP_EVENT for ts in time_index],
            dtype=float,
        )

    if scenario_id == "S5":
        values = []
        for ts in time_index:
            if 0 <= ts.hour < 6:
                values.append(TARGET_SP_DEFAULT_NIGHT)
            elif 6 <= ts.hour < 18:
                values.append(TARGET_SP_DAY)
            elif 18 <= ts.hour < 19:
                values.append(TARGET_SP_EVENT)
            elif 19 <= ts.hour < 22:
                values.append(TARGET_SP_EVENING)
            else:
                values.append(TARGET_SP_DEFAULT_NIGHT)
        return np.asarray(values, dtype=float)

    if scenario_id == "S6":
        return np.full(n, TARGET_SP_LOW, dtype=float)

    raise KeyError(f"Unknown scenario id: {scenario_id}")


def _apply_weather_disturbance(
    scenario_id: str,
    outdoor_temp_c: np.ndarray,
    notes: dict[str, Any],
) -> None:
    scenario_id = scenario_id.upper()

    if scenario_id == "S2":
        outdoor_temp_c[30:] -= 10.0
        np.maximum(outdoor_temp_c, 0.0, out=outdoor_temp_c)
        notes["weather_disturbance"] = "Applied 10C step drop from minute 30 onward."
        return

    if scenario_id == "S5":
        outdoor_temp_c[240:360] -= 6.0
        outdoor_temp_c[720:900] += 3.0
        outdoor_temp_c[1080:1260] -= 4.0
        np.maximum(outdoor_temp_c, 0.0, out=outdoor_temp_c)
        notes["weather_disturbance"] = (
            "Applied cold front, warm rebound, and evening cooldown for the stress test."
        )


def _apply_tariff_modifiers(
    scenario_id: str,
    tariff_usd_per_kwh: np.ndarray,
    notes: dict[str, Any],
) -> None:
    scenario_id = scenario_id.upper()

    if scenario_id == "S3":
        spike_start = len(tariff_usd_per_kwh) // 2 - 90
        spike_end = min(len(tariff_usd_per_kwh), spike_start + 180)
        tariff_usd_per_kwh[spike_start:spike_end] = TARIFF_ONPEAK * 3.0
        notes["tariff_modifier"] = "Applied 3x tariff spike in the middle third of the run."
        return

    if scenario_id == "S5":
        peak_mask = tariff_usd_per_kwh == TARIFF_ONPEAK
        tariff_usd_per_kwh[peak_mask] *= 2.0
        notes["tariff_modifier"] = "Doubled on-peak tariff for the 24-hour stress test."


def _build_failure_injection_steps(
    *,
    scenario: ScenarioSpec,
    seed: int,
    supervision_interval_steps: int,
) -> set[int]:
    if not scenario.failure_injection or scenario.failure_rate <= 0.0:
        return set()

    supervision_points = list(
        range(
            supervision_interval_steps,
            scenario.duration_steps,
            supervision_interval_steps,
        )
    )
    if not supervision_points:
        return set()

    rng = np.random.default_rng(seed + 77_777)
    forced_steps = {
        int(step)
        for step in supervision_points
        if rng.random() < scenario.failure_rate
    }
    return forced_steps
