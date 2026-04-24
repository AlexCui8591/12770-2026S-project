"""Batch runner for the formal C0-C3 HVAC experiment suite."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from agent.supervisor import DEFAULT_MODEL, check_ollama_connection, create_ollama_client
from demo import c0_baseline, c1_llm_setpoint_only, c2_reactive_pid_supervision, c3_proactive_pid_supervision
from demo.generate_two_days_temperature import generate_one_day

from .scenarios import (
    DEFAULT_ROOM,
    ExperimentSettings,
    ScenarioRun,
    build_scenario_run,
    build_tuning_setpoint_schedule,
    default_condition_ids,
    default_scenario_ids,
    load_experiment_settings,
    resolve_seed_list,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "full_experiment"
FALLBACK_RC_PATH = ROOT_DIR / "demo" / "config" / "rc_fit_results.json"
METRIC_COLUMNS = [
    "MAD_C",
    "CVR_fraction",
    "CE_kWh",
    "EC_USD",
    "RT_s",
    "ST_s",
    "OC_count",
    "MO_C",
]


@dataclass(frozen=True)
class PlannedRun:
    condition_id: str
    scenario_id: str
    seed: int
    run_dir: Path
    summary_path: Path

    @property
    def label(self) -> str:
        return f"{self.condition_id}/{self.scenario_id}/seed_{self.seed:04d}"


class ProgressReporter:
    def __init__(self, *, enabled: bool, total_runs: int) -> None:
        self.enabled = enabled
        self.total_runs = total_runs
        self.start_time = time.perf_counter()
        self.completed_runs = 0
        self.completed_durations: list[float] = []

    def info(self, message: str) -> None:
        print(message)

    def log_plan(self, *, planned: int, pending: int, skipped: int) -> None:
        self.info(f"Run plan: planned={planned}, pending={pending}, skipped={skipped}")

    def log_no_pending(self) -> None:
        self.info("No pending runs; all planned runs are already completed or skipped.")

    def run_started(self, run: PlannedRun) -> None:
        if not self.enabled or self.total_runs <= 0:
            return
        index = self.completed_runs + 1
        elapsed = _format_duration(time.perf_counter() - self.start_time)
        eta = self._estimate_eta()
        self.info(
            f"[{index}/{self.total_runs}] Running {run.label} | "
            f"elapsed {elapsed} | eta {eta}"
        )

    def run_completed(self, run: PlannedRun, *, duration_seconds: float) -> None:
        self.completed_runs += 1
        self.completed_durations.append(duration_seconds)
        duration_text = _format_duration(duration_seconds)
        if self.enabled and self.total_runs > 0:
            self.info(
                f"[{self.completed_runs}/{self.total_runs}] "
                f"Completed {run.label} in {duration_text}"
            )
        else:
            self.info(f"Completed {run.label} in {duration_text}")

    def _estimate_eta(self) -> str:
        if not self.completed_durations:
            return "--"
        avg_duration = sum(self.completed_durations) / len(self.completed_durations)
        remaining_runs = max(0, self.total_runs - self.completed_runs)
        return _format_duration(avg_duration * remaining_runs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal C0-C3 HVAC experiments.")
    parser.add_argument(
        "--condition",
        choices=["C0", "C1", "C2", "C3", "all"],
        default="C2",
        help="Condition to run. Use 'all' for the full ablation matrix.",
    )
    parser.add_argument(
        "--scenario",
        choices=["S1", "S2", "S3", "S4", "S5", "S6", "all"],
        default="all",
        help="Scenario to run. Use 'all' for the full suite.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of seeded repetitions per condition/scenario pair.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where full-experiment outputs will be written.",
    )
    parser.add_argument(
        "--supervisor-mode",
        choices=["rules", "agent_auto", "agent_only"],
        default="agent_auto",
        help="Agent supervisor mode for C2/C3.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="LLM model name for C2 reactive supervision.",
    )
    parser.add_argument(
        "--room",
        default=DEFAULT_ROOM,
        help="Room identifier passed to the agent telemetry tool.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose summary.json already exists.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Persist per-run temperature and power plots in addition to CSV/JSON outputs.",
    )
    parser.add_argument(
        "--progress",
        choices=["on", "off"],
        default="on",
        help="Show batch-level progress with ETA in the terminal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_experiment_settings()
    condition_ids = (
        default_condition_ids() if args.condition == "all" else [args.condition]
    )
    scenario_ids = default_scenario_ids() if args.scenario == "all" else [args.scenario]
    requested_runs = args.runs or settings.runs_per_condition
    seeds = resolve_seed_list(settings.seeds, requested_runs)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    planned_runs = _build_planned_runs(
        condition_ids=condition_ids,
        scenario_ids=scenario_ids,
        seeds=seeds,
        output_root=output_root,
    )
    progress = ProgressReporter(
        enabled=(args.progress == "on"),
        total_runs=0,
    )
    existing_records, pending_runs = _partition_runs_for_execution(
        planned_runs=planned_runs,
        resume=args.resume,
        emit=progress.info,
        verbose_skips=not progress.enabled,
    )
    skipped_runs = len(planned_runs) - len(pending_runs)
    progress.total_runs = len(pending_runs)
    progress.log_plan(
        planned=len(planned_runs),
        pending=len(pending_runs),
        skipped=skipped_runs,
    )
    if not pending_runs:
        progress.log_no_pending()
        return

    effective_supervisor_mode, supervisor_client = _prepare_supervisor_resources(
        requested_mode=args.supervisor_mode,
        needs_c2=("C2" in condition_ids or "C3" in condition_ids),
        emit=progress.info,
    )
    tuning_cache = _build_pid_tuning_cache(seeds, output_root=output_root)
    run_records: list[dict[str, Any]] = list(existing_records)

    for planned_run in pending_runs:
        progress.run_started(planned_run)
        run_start = time.perf_counter()
        scenario_run = build_scenario_run(
            planned_run.scenario_id,
            planned_run.seed,
            room=args.room,
            supervision_interval_steps=settings.conditions["C2"].supervision_interval_steps,
            settings=settings,
        )
        base_pid = tuning_cache[planned_run.seed]
        result = _run_condition(
            condition_id=planned_run.condition_id,
            scenario_run=scenario_run,
            base_pid=base_pid,
            requested_supervisor_mode=args.supervisor_mode,
            effective_supervisor_mode=effective_supervisor_mode,
            llm_model=args.model,
            supervisor_client=supervisor_client,
        )
        _save_run_outputs(
            run_dir=planned_run.run_dir,
            scenario_run=scenario_run,
            result=result,
            save_plots=args.save_plots,
        )
        run_records.append(result["manifest_record"])
        progress.run_completed(
            planned_run,
            duration_seconds=time.perf_counter() - run_start,
        )

    if not run_records:
        progress.info("No runs executed.")
        return

    manifest_df = pd.DataFrame(run_records).sort_values(
        ["condition", "scenario_id", "seed"]
    ).reset_index(drop=True)
    manifest_path = output_root / "run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    aggregate_df = _aggregate_manifest(manifest_df)
    aggregate_path = output_root / "aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root.resolve()),
        "conditions": condition_ids,
        "scenarios": scenario_ids,
        "requested_runs_per_condition": requested_runs,
        "resolved_seeds": seeds,
        "requested_supervisor_mode": args.supervisor_mode,
        "effective_supervisor_mode": effective_supervisor_mode,
        "model": args.model if any(condition in {"C2", "C3"} for condition in condition_ids) else None,
        "run_manifest_file": str(manifest_path.resolve()),
        "aggregate_metrics_file": str(aggregate_path.resolve()),
        "num_completed_runs": int(len(manifest_df)),
    }
    (output_root / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    progress.info(f"Saved full-experiment outputs to {output_root.resolve()}")


def _prepare_supervisor_resources(
    *,
    requested_mode: str,
    needs_c2: bool,
    emit: Callable[[str], None] = print,
) -> tuple[str, Any]:
    if not needs_c2 or requested_mode == "rules":
        return requested_mode, None

    if check_ollama_connection():
        return requested_mode, create_ollama_client()

    if requested_mode == "agent_only":
        raise RuntimeError("Ollama is unavailable, so agent_only mode cannot run.")

    emit("Ollama unavailable; downgrading all C2 runs to rule-based supervision.")
    return "rules", None


def _build_planned_runs(
    *,
    condition_ids: list[str],
    scenario_ids: list[str],
    seeds: list[int],
    output_root: Path,
) -> list[PlannedRun]:
    planned_runs: list[PlannedRun] = []
    for condition_id in condition_ids:
        for scenario_id in scenario_ids:
            for seed in seeds:
                run_dir = output_root / condition_id / scenario_id / f"seed_{seed:04d}"
                planned_runs.append(
                    PlannedRun(
                        condition_id=condition_id,
                        scenario_id=scenario_id,
                        seed=seed,
                        run_dir=run_dir,
                        summary_path=run_dir / "summary.json",
                    )
                )
    return planned_runs


def _partition_runs_for_execution(
    *,
    planned_runs: list[PlannedRun],
    resume: bool,
    emit: Callable[[str], None],
    verbose_skips: bool,
) -> tuple[list[dict[str, Any]], list[PlannedRun]]:
    existing_records: list[dict[str, Any]] = []
    pending_runs: list[PlannedRun] = []

    for planned_run in planned_runs:
        if not resume or not planned_run.summary_path.exists():
            pending_runs.append(planned_run)
            continue

        existing_record = _load_manifest_record_from_summary(planned_run.summary_path)
        if existing_record is None:
            emit(f"Unreadable summary; rerunning {planned_run.label}")
            pending_runs.append(planned_run)
            continue

        existing_records.append(existing_record)
        if verbose_skips:
            emit(f"Skipping completed run: {planned_run.label}")

    return existing_records, pending_runs


def _load_manifest_record_from_summary(summary_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        return None

    record = {
        "condition": data.get("condition"),
        "scenario_id": data.get("scenario_id"),
        "scenario_name": data.get("scenario_name"),
        "seed": data.get("seed"),
        "requested_supervisor_mode": data.get("requested_supervisor_mode"),
        "effective_supervisor_mode": data.get("effective_supervisor_mode"),
        "num_supervision_updates": data.get("num_supervision_updates", 0),
    }
    record.update(metrics)
    return record


def _build_pid_tuning_cache(
    seeds: list[int],
    *,
    output_root: Path,
) -> dict[int, dict[str, float]]:
    r_hat, c_hat = _load_rc_params_with_fallback()
    cache: dict[int, dict[str, float]] = {}
    tuning_dir = output_root / "_pid_tuning_cache"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        cache_path = tuning_dir / f"seed_{seed:04d}.json"
        if cache_path.exists():
            cache[seed] = json.loads(cache_path.read_text(encoding="utf-8"))
            continue

        day1_df = generate_one_day(seed, "day_1")
        ta_day1 = c0_baseline.expand_hourly_to_minute(day1_df["Ta_C"].to_numpy(dtype=float))
        time_day1 = c0_baseline.build_minute_time_index("day_1")
        tsp_day1 = build_tuning_setpoint_schedule(time_day1)
        ti0_day1 = float(ta_day1[0])
        best_pid, _, _ = c0_baseline.tune_pid_on_day1(
            time_index=time_day1,
            ta_array=ta_day1,
            tsp_array=tsp_day1,
            r_hat=r_hat,
            c_hat=c_hat,
            ti0=ti0_day1,
        )
        cache[seed] = {
            "KP": float(best_pid["KP"]),
            "KI": float(best_pid["KI"]),
            "KD": float(best_pid["KD"]),
            "R_hat": float(r_hat),
            "C_hat": float(c_hat),
        }
        cache_path.write_text(json.dumps(cache[seed], indent=2), encoding="utf-8")
    return cache


def _load_rc_params_with_fallback() -> tuple[float, float]:
    try:
        return c0_baseline.load_rc_params()
    except FileNotFoundError:
        data = json.loads(FALLBACK_RC_PATH.read_text(encoding="utf-8"))
        return float(data["R_hat_K_per_W"]), float(data["C_hat_J_per_K"])


def _run_condition(
    *,
    condition_id: str,
    scenario_run: ScenarioRun,
    base_pid: dict[str, float],
    requested_supervisor_mode: str,
    effective_supervisor_mode: str,
    llm_model: str,
    supervisor_client: Any,
) -> dict[str, Any]:
    time_index = scenario_run.time_index
    ta_array = scenario_run.outdoor_temp_c
    r_hat = float(base_pid["R_hat"])
    c_hat = float(base_pid["C_hat"])
    kp0 = float(base_pid["KP"])
    ki0 = float(base_pid["KI"])
    kd0 = float(base_pid["KD"])
    ti0 = float(ta_array[0])

    supervision_log = pd.DataFrame()
    plots: dict[str, Any] = {}

    if condition_id == "C0":
        tsp = scenario_run.baseline_setpoint_c
        sim = c0_baseline.simulate_closed_loop_pid_1r1c(
            time_index=time_index,
            ta_array=ta_array,
            tsp_array=tsp,
            r=r_hat,
            c=c_hat,
            ti0=ti0,
            kp=kp0,
            ki=ki0,
            kd=kd0,
        )
        metrics = c0_baseline.compute_evaluation_metrics(
            time_index=time_index,
            ti=sim["Ti"],
            tsp=tsp,
            phea_w=sim["Phea"],
        )
        plots["temperature"] = c0_baseline.save_temperature_plot
        plots["power"] = c0_baseline.save_power_plot
    elif condition_id == "C1":
        tsp = scenario_run.target_setpoint_c
        sim = c1_llm_setpoint_only.simulate_closed_loop_pid_1r1c(
            time_index=time_index,
            ta_array=ta_array,
            tsp_array=tsp,
            r=r_hat,
            c=c_hat,
            ti0=ti0,
            kp=kp0,
            ki=ki0,
            kd=kd0,
        )
        metrics = c1_llm_setpoint_only.compute_evaluation_metrics(
            time_index=time_index,
            ti=sim["Ti"],
            tsp=tsp,
            phea_w=sim["Phea"],
        )
        plots["temperature"] = c1_llm_setpoint_only.save_temperature_plot
        plots["power"] = c1_llm_setpoint_only.save_power_plot
    elif condition_id == "C2":
        tsp = scenario_run.target_setpoint_c
        sim, supervision_log = c2_reactive_pid_supervision.simulate_reactive_pid_supervision(
            time_index=time_index,
            ta_array=ta_array,
            tsp_array=tsp,
            r=r_hat,
            c=c_hat,
            ti0=ti0,
            kp0=kp0,
            ki0=ki0,
            kd0=kd0,
            supervisor_mode=effective_supervisor_mode,
            user_objective=scenario_run.user_objective,
            room=scenario_run.room,
            llm_model=llm_model,
            supervisor_client=supervisor_client,
            forced_rule_steps=scenario_run.forced_rule_steps,
        )
        metrics = c2_reactive_pid_supervision.compute_evaluation_metrics(
            time_index=time_index,
            ti=sim["Ti"],
            tsp=tsp,
            phea_w=sim["Phea"],
        )
        plots["temperature"] = c2_reactive_pid_supervision.save_temperature_plot
        plots["power"] = c2_reactive_pid_supervision.save_power_plot
    elif condition_id == "C3":
        tsp = scenario_run.target_setpoint_c
        sim, supervision_log = c3_proactive_pid_supervision.simulate_proactive_pid_supervision(
            time_index=time_index,
            ta_array=ta_array,
            tsp_array=tsp,
            r=r_hat,
            c=c_hat,
            ti0=ti0,
            kp0=kp0,
            ki0=ki0,
            kd0=kd0,
            supervisor_mode=effective_supervisor_mode,
            user_objective=scenario_run.user_objective,
            room=scenario_run.room,
            llm_model=llm_model,
            supervisor_client=supervisor_client,
        )
        metrics = c3_proactive_pid_supervision.compute_evaluation_metrics(
            time_index=time_index,
            ti=sim["Ti"],
            tsp=tsp,
            phea_w=sim["Phea"],
        )
        plots["temperature"] = c3_proactive_pid_supervision.save_temperature_plot
        plots["power"] = c3_proactive_pid_supervision.save_power_plot
    else:
        raise KeyError(f"Unknown condition id: {condition_id}")

    metrics = dict(metrics)
    metrics["EC_USD"] = _compute_electricity_cost(
        power_w=sim["Phea"],
        tariff_usd_per_kwh=scenario_run.tariff_usd_per_kwh,
    )

    decision_source_counts = (
        Counter(supervision_log["decision_source"])
        if "decision_source" in supervision_log.columns
        else Counter()
    )
    decision_mode_counts = (
        Counter(supervision_log["decision_mode"])
        if "decision_mode" in supervision_log.columns
        else Counter()
    )

    summary = {
        "condition": condition_id,
        "scenario_id": scenario_run.scenario.id,
        "scenario_name": scenario_run.scenario.name,
        "scenario_command": scenario_run.scenario.command,
        "seed": scenario_run.seed,
        "room": scenario_run.room,
        "initial_pid": {"KP": kp0, "KI": ki0, "KD": kd0},
        "requested_supervisor_mode": requested_supervisor_mode if condition_id in {"C2", "C3"} else None,
        "effective_supervisor_mode": effective_supervisor_mode if condition_id in {"C2", "C3"} else None,
        "llm_model": llm_model if condition_id in {"C2", "C3"} and effective_supervisor_mode != "rules" else None,
        "forced_rule_steps": sorted(scenario_run.forced_rule_steps) if condition_id == "C2" else [],
        "num_supervision_updates": int(len(supervision_log)),
        "decision_source_counts": dict(decision_source_counts),
        "decision_mode_counts": dict(decision_mode_counts),
        "metrics": metrics,
        "scenario_notes": scenario_run.notes,
    }

    manifest_record = {
        "condition": condition_id,
        "scenario_id": scenario_run.scenario.id,
        "scenario_name": scenario_run.scenario.name,
        "seed": scenario_run.seed,
        "requested_supervisor_mode": requested_supervisor_mode if condition_id in {"C2", "C3"} else None,
        "effective_supervisor_mode": effective_supervisor_mode if condition_id in {"C2", "C3"} else None,
        "num_supervision_updates": int(len(supervision_log)),
    }
    manifest_record.update(metrics)

    return {
        "time_index": time_index,
        "outdoor_temp_c": ta_array,
        "setpoint_c": tsp,
        "tariff_usd_per_kwh": scenario_run.tariff_usd_per_kwh,
        "sim": sim,
        "supervision_log": supervision_log,
        "summary": summary,
        "manifest_record": manifest_record,
        "plot_fns": plots,
    }


def _compute_electricity_cost(
    *,
    power_w: np.ndarray,
    tariff_usd_per_kwh: np.ndarray,
    dt_seconds: float = 60.0,
) -> float:
    dt_hours = dt_seconds / 3600.0
    return float(np.sum((np.asarray(power_w, dtype=float) / 1000.0) * dt_hours * tariff_usd_per_kwh))


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _save_run_outputs(
    *,
    run_dir: Path,
    scenario_run: ScenarioRun,
    result: dict[str, Any],
    save_plots: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    sim = result["sim"]
    time_index = result["time_index"]
    timeseries = pd.DataFrame(
        {
            "time": time_index,
            "Ta_C": result["outdoor_temp_c"],
            "Tsp_C": result["setpoint_c"],
            "Ti_C": sim["Ti"],
            "Phea_W": sim["Phea"],
            "tariff_USD_per_kWh": result["tariff_usd_per_kwh"],
            "error_C": sim["err"],
            "integral_term": sim["int"],
            "derivative_term": sim["der"],
        }
    )
    if "Kp" in sim:
        timeseries["Kp"] = sim["Kp"]
        timeseries["Ki"] = sim["Ki"]
        timeseries["Kd"] = sim["Kd"]
    timeseries.to_csv(run_dir / "timeseries.csv", index=False)

    supervision_log = result["supervision_log"]
    if not supervision_log.empty:
        supervision_log.to_csv(run_dir / "supervision_log.csv", index=False)

    (run_dir / "scenario_notes.json").write_text(
        json.dumps(scenario_run.notes, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result["summary"]["metrics"], indent=2),
        encoding="utf-8",
    )

    if not save_plots:
        return

    plot_fns = result["plot_fns"]
    plot_fns["temperature"](
        run_dir / "temperature_plot.png",
        result["outdoor_temp_c"],
        result["setpoint_c"],
        sim["Ti"],
        f"{result['summary']['condition']} {result['summary']['scenario_id']} temperature response",
    )
    plot_fns["power"](
        run_dir / "power_plot.png",
        sim["Phea"],
        f"{result['summary']['condition']} {result['summary']['scenario_id']} heater power",
    )


def _aggregate_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        manifest_df.groupby(["condition", "scenario_id"], dropna=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    flattened_columns = []
    for column in aggregated.columns:
        if isinstance(column, tuple):
            left, right = column
            flattened_columns.append(left if not right else f"{left}_{right}")
        else:
            flattened_columns.append(column)
    aggregated.columns = flattened_columns
    return aggregated.sort_values(["condition", "scenario_id"]).reset_index(drop=True)


if __name__ == "__main__":
    main()
