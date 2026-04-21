"""
compare_all_metrics_barplots.py

Load the metric JSON files for C0-C3 and generate:
1. one bar chart per metric
2. a normalized weighted overall score table and bar chart

Each metric chart uses:
- x-axis: C0, C1, C2, C3
- y-axis: metric value

Overall score method
--------------------
All metrics are assumed to be "lower is better".
For each metric j, min-max normalize to a benefit score:

    score_ij = (max_j - x_ij) / (max_j - min_j)

Dimension weights:
- Comfort: 4
- Energy: 3
- Responsiveness: 1
- Stability: 2

Within each dimension, the metrics are averaged first:
- Comfort: MAD_C, CVR_fraction
- Energy: CE_kWh, EC_USD
- Responsiveness: RT_s, ST_s
- Stability: OC_count, MO_C

Then the weighted overall score is:
    overall = (4*comfort + 3*energy + 1*responsiveness + 2*stability) / 10

Expected metric files
---------------------
- outputs/c0_tuned_pid_outputs/c0_day2_metrics.json
- outputs/c1_llm_setpoint_only_outputs/c1_day2_metrics.json
- outputs/c2_reactive_pid_outputs/c2_day2_metrics.json
- outputs/c3_proactive_pid_outputs/c3_day2_metrics.json

Outputs
-------
- outputs/metrics_comparison_plots/metrics_comparison_table.csv
- outputs/metrics_comparison_plots/<metric_name>_bar.png
- outputs/metrics_comparison_plots/normalized_weighted_scores.csv
- outputs/metrics_comparison_plots/overall_weighted_score_bar.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


C0_METRICS = Path("outputs/c0_tuned_pid_outputs/c0_day2_metrics.json")
C1_METRICS = Path("outputs/c1_llm_setpoint_only_outputs/c1_day2_metrics.json")
C2_METRICS = Path("outputs/c2_reactive_pid_outputs/c2_day2_metrics.json")
C3_METRICS = Path("outputs/c3_proactive_pid_outputs/c3_day2_metrics.json")

OUTPUT_DIR = Path("outputs/metrics_comparison_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMENSION_WEIGHTS = {
    "comfort": 4.0,
    "energy": 3.0,
    "responsiveness": 1.0,
    "stability": 2.0,
}

DIMENSION_METRICS = {
    "comfort": ["MAD_C", "CVR_fraction"],
    "energy": ["CE_kWh", "EC_USD"],
    "responsiveness": ["RT_s", "ST_s"],
    "stability": ["OC_count", "MO_C"],
}


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find metrics file: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_value(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def build_metrics_table() -> pd.DataFrame:
    metrics_by_model = {
        "C0": load_metrics(C0_METRICS),
        "C1": load_metrics(C1_METRICS),
        "C2": load_metrics(C2_METRICS),
        "C3": load_metrics(C3_METRICS),
    }

    all_metric_names = []
    for model_metrics in metrics_by_model.values():
        for key in model_metrics.keys():
            if key not in all_metric_names:
                all_metric_names.append(key)

    rows = []
    for metric_name in all_metric_names:
        row = {"metric": metric_name}
        for model_name, model_metrics in metrics_by_model.items():
            row[model_name] = sanitize_value(model_metrics.get(metric_name))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_one_metric(metric_name: str, values: list[float | None], labels: list[str], output_dir: Path) -> None:
    plot_values = [0.0 if v is None else float(v) for v in values]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, plot_values)
    plt.xlabel("Model")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} comparison across C0-C3")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric_name}_bar.png", dpi=200)
    plt.close()


def min_max_benefit_score(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series([None] * len(series), index=series.index, dtype="float64")

    min_val = float(valid.min())
    max_val = float(valid.max())

    if abs(max_val - min_val) < 1e-12:
        return pd.Series([1.0 if pd.notna(v) else None for v in series], index=series.index, dtype="float64")

    return series.apply(lambda x: None if pd.isna(x) else (max_val - float(x)) / (max_val - min_val))


def build_weighted_score_table(metrics_table: pd.DataFrame) -> pd.DataFrame:
    df = metrics_table.set_index("metric")[["C0", "C1", "C2", "C3"]].copy()

    normalized = df.copy()
    for metric_name in normalized.index:
        normalized.loc[metric_name] = min_max_benefit_score(normalized.loc[metric_name])

    dimension_scores = {}
    for dim_name, metric_list in DIMENSION_METRICS.items():
        missing = [m for m in metric_list if m not in normalized.index]
        if missing:
            raise ValueError(f"Missing required metrics for dimension '{dim_name}': {missing}")
        dimension_scores[dim_name] = normalized.loc[metric_list].mean(axis=0, skipna=True)

    score_df = pd.DataFrame(dimension_scores)

    total_weight = sum(DIMENSION_WEIGHTS.values())
    score_df["overall_weighted_score"] = (
        score_df["comfort"] * DIMENSION_WEIGHTS["comfort"]
        + score_df["energy"] * DIMENSION_WEIGHTS["energy"]
        + score_df["responsiveness"] * DIMENSION_WEIGHTS["responsiveness"]
        + score_df["stability"] * DIMENSION_WEIGHTS["stability"]
    ) / total_weight

    score_df = score_df.reset_index().rename(columns={"index": "model"})
    score_df = score_df.sort_values("overall_weighted_score", ascending=False).reset_index(drop=True)
    return score_df


def plot_overall_scores(score_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(score_df["model"], score_df["overall_weighted_score"])
    plt.xlabel("Model")
    plt.ylabel("Overall weighted score")
    plt.title("Normalized weighted overall score across C0-C3")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "overall_weighted_score_bar.png", dpi=200)
    plt.close()


def main() -> None:
    df = build_metrics_table()
    df.to_csv(OUTPUT_DIR / "metrics_comparison_table.csv", index=False)

    labels = ["C0", "C1", "C2", "C3"]
    for _, row in df.iterrows():
        metric_name = str(row["metric"])
        values = [row["C0"], row["C1"], row["C2"], row["C3"]]
        plot_one_metric(metric_name, values, labels, OUTPUT_DIR)

    score_df = build_weighted_score_table(df)
    score_df.to_csv(OUTPUT_DIR / "normalized_weighted_scores.csv", index=False)
    plot_overall_scores(score_df, OUTPUT_DIR)

    best_model = score_df.iloc[0]["model"]
    best_score = score_df.iloc[0]["overall_weighted_score"]

    print("Metric comparison plots generated successfully.")
    print(f"Saved table: {(OUTPUT_DIR / 'metrics_comparison_table.csv').resolve()}")
    print(f"Saved weighted scores: {(OUTPUT_DIR / 'normalized_weighted_scores.csv').resolve()}")
    print(f"Saved plots in: {OUTPUT_DIR.resolve()}")
    print(f"Best overall model by normalized weighted score: {best_model} ({best_score:.4f})")


if __name__ == "__main__":
    main()
