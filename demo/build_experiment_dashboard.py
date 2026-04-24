"""Build a standalone HTML dashboard for the full HVAC experiment outputs.

The generated dashboard embeds the aggregated metrics, sampled timeseries,
and supervisor logs directly into one HTML file so it can be opened from disk
without a local web server.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOTS = {
    "C2": ROOT / "outputs" / "full_experiment_c2",
    "C3": ROOT / "outputs" / "full_experiment_c3_interval10",
}
DESTINATION = ROOT / "demo" / "experiment_control_dashboard.html"

METRICS = [
    "MAD_C",
    "CVR_fraction",
    "CE_kWh",
    "EC_USD",
    "RT_s",
    "ST_s",
    "OC_count",
    "MO_C",
]

LOWER_IS_BETTER = set(METRICS)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _read_aggregate(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {
                "condition": row["condition"],
                "scenario_id": row["scenario_id"],
            }
            for key, value in row.items():
                if key in {"condition", "scenario_id"}:
                    continue
                parsed[key] = _safe_float(value)
            rows.append(parsed)
    return rows


def _compact_timeseries(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    keep = [
        "time",
        "Ta_C",
        "Tsp_base_C",
        "Tsp_C",
        "Ti_C",
        "Phea_W",
        "tariff_USD_per_kWh",
        "Kp",
        "Ki",
        "Kd",
        "comfort_weight",
        "energy_weight",
        "response_weight",
    ]
    df = df[[column for column in keep if column in df.columns]]
    return [
        {
            column: (None if pd.isna(value) else value)
            for column, value in record.items()
        }
        for record in df.to_dict(orient="records")
    ]


def _read_supervision_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    keep = [
        "time",
        "decision_mode",
        "current_ti_C",
        "current_tsp_C",
        "action",
        "decision_source",
        "reason",
        "rationale",
        "kp_after",
        "ki_after",
        "kd_after",
        "setpoint_after_C",
        "comfort_weight_after",
        "energy_weight_after",
        "response_weight_after",
    ]
    df = df[[column for column in keep if column in df.columns]].tail(28)
    return [
        {
            column: (None if pd.isna(value) else value)
            for column, value in record.items()
        }
        for record in df.to_dict(orient="records")
    ]


def build_data() -> dict[str, Any]:
    aggregate: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for condition, output_root in OUTPUT_ROOTS.items():
        if not output_root.exists():
            continue
        aggregate_path = output_root / "aggregate_metrics.csv"
        if aggregate_path.exists():
            aggregate.extend(_read_aggregate(aggregate_path))

        for metrics_path in sorted(output_root.glob(f"{condition}/S*/seed_*/metrics.json")):
            run_dir = metrics_path.parent
            summary = _read_json(run_dir / "summary.json")
            notes = _read_json(run_dir / "scenario_notes.json")
            metrics = _read_json(metrics_path)
            run = {
                "condition": condition,
                "scenario_id": summary["scenario_id"],
                "scenario_name": summary["scenario_name"],
                "scenario_command": summary["scenario_command"],
                "seed": int(summary["seed"]),
                "metrics": metrics,
                "summary": {
                    "num_supervision_updates": summary.get("num_supervision_updates"),
                    "decision_source_counts": summary.get("decision_source_counts", {}),
                    "effective_supervisor_mode": summary.get("effective_supervisor_mode"),
                    "llm_model": summary.get("llm_model"),
                },
                "notes": notes,
                "timeseries": _compact_timeseries(run_dir / "timeseries.csv"),
                "supervision": _read_supervision_log(run_dir / "supervision_log.csv"),
            }
            runs.append(run)
            summaries.append(
                {
                    "condition": condition,
                    "scenario_id": run["scenario_id"],
                    "seed": run["seed"],
                    "num_supervision_updates": run["summary"]["num_supervision_updates"],
                }
            )

    scenarios = {
        run["scenario_id"]: {
            "id": run["scenario_id"],
            "name": run["scenario_name"],
            "command": run["scenario_command"],
            "notes": run["notes"],
        }
        for run in runs
    }
    seeds = sorted({run["seed"] for run in runs})
    conditions = sorted({run["condition"] for run in runs})

    return {
        "generated_from": {key: str(path.relative_to(ROOT)) for key, path in OUTPUT_ROOTS.items()},
        "metrics": METRICS,
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "conditions": conditions,
        "scenarios": [scenarios[key] for key in sorted(scenarios)],
        "seeds": seeds,
        "aggregate": aggregate,
        "runs": runs,
        "run_summaries": summaries,
    }


def render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HVAC Control Experiment Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1b1f23;
      --muted: #68707d;
      --line: #d9dee7;
      --c2: #2f7f95;
      --c3: #c65f3d;
      --accent: #7b61a8;
      --ok: #217a4d;
      --bad: #b33a3a;
      --warn: #a76c13;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 20px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{ margin: 0 0 4px; font-size: 24px; font-weight: 720; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; font-weight: 700; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 18px 24px 28px; }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 12px;
      align-items: end;
      margin-bottom: 16px;
    }}
    label {{ display: grid; gap: 6px; color: var(--muted); font-size: 12px; font-weight: 650; }}
    select {{
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }}
    .span-2 {{ grid-column: 1 / -1; }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .kpi {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 76px;
    }}
    .kpi .label {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    .kpi .value {{ margin-top: 6px; font-size: 22px; font-weight: 760; }}
    .kpi .sub {{ margin-top: 2px; color: var(--muted); font-size: 12px; }}
    svg {{ width: 100%; height: 320px; display: block; }}
    .short svg {{ height: 250px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid #eef1f5; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; background: #fafbfc; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .good {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .note {{
      display: grid;
      gap: 6px;
      padding: 10px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .split {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.7fr);
      gap: 14px;
      align-items: start;
    }}
    .scroll {{ max-height: 330px; overflow: auto; border: 1px solid var(--line); border-radius: 7px; }}
    @media (max-width: 980px) {{
      main {{ padding: 14px; }}
      .controls, .grid, .split, .kpis {{ grid-template-columns: 1fr; }}
      .span-2 {{ grid-column: auto; }}
      header {{ padding: 16px; }}
      svg {{ height: 270px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>HVAC Control Experiment Dashboard</h1>
    <p>C2 reactive supervision vs C3 proactive supervision, built from formal experiment outputs.</p>
  </header>
  <main>
    <section class="controls" aria-label="Dashboard controls">
      <label>Metric<select id="metricSelect"></select></label>
      <label>Scenario<select id="scenarioSelect"></select></label>
      <label>Condition<select id="conditionSelect"></select></label>
      <label>Seed<select id="seedSelect"></select></label>
    </section>

    <section class="kpis" id="kpis"></section>

    <section class="grid">
      <div class="panel">
        <h2>Scenario Metric Comparison</h2>
        <svg id="metricChart" role="img" aria-label="Metric comparison chart"></svg>
        <div class="legend">
          <span><i class="swatch" style="background: var(--c2)"></i>C2 reactive</span>
          <span><i class="swatch" style="background: var(--c3)"></i>C3 proactive</span>
        </div>
      </div>

      <div class="panel">
        <h2>C3 vs C2 Delta</h2>
        <div class="scroll"><table id="deltaTable"></table></div>
        <div class="note" id="scenarioNote"></div>
      </div>

      <div class="panel span-2">
        <div class="split">
          <div>
            <h2>Temperature Tracking</h2>
            <svg id="temperatureChart" role="img" aria-label="Temperature trajectory chart"></svg>
            <div class="legend">
              <span><i class="swatch" style="background: #1f6fb2"></i>Indoor Ti</span>
              <span><i class="swatch" style="background: #b43c5d"></i>Setpoint</span>
              <span><i class="swatch" style="background: #6f7784"></i>Outdoor Ta</span>
            </div>
          </div>
          <div class="short">
            <h2>Power and Tariff</h2>
            <svg id="powerChart" role="img" aria-label="Power and tariff chart"></svg>
            <div class="legend">
              <span><i class="swatch" style="background: #8a6f18"></i>Heating power</span>
              <span><i class="swatch" style="background: #7b61a8"></i>Tariff</span>
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Controller Weights</h2>
        <svg id="weightsChart" role="img" aria-label="Controller weight chart"></svg>
        <div class="legend">
          <span><i class="swatch" style="background: #217a4d"></i>Comfort</span>
          <span><i class="swatch" style="background: #a76c13"></i>Energy</span>
          <span><i class="swatch" style="background: #7b61a8"></i>Response</span>
        </div>
      </div>

      <div class="panel">
        <h2>Recent Supervisor Decisions</h2>
        <div class="scroll"><table id="decisionTable"></table></div>
      </div>
    </section>
  </main>

  <script>
    const DATA = {payload};

    const COLORS = {{
      c2: '#2f7f95',
      c3: '#c65f3d',
      grid: '#e4e8ef',
      axis: '#68707d',
      ti: '#1f6fb2',
      tsp: '#b43c5d',
      ta: '#6f7784',
      power: '#8a6f18',
      tariff: '#7b61a8',
      comfort: '#217a4d',
      energy: '#a76c13',
      response: '#7b61a8'
    }};

    const metricSelect = document.getElementById('metricSelect');
    const scenarioSelect = document.getElementById('scenarioSelect');
    const conditionSelect = document.getElementById('conditionSelect');
    const seedSelect = document.getElementById('seedSelect');

    function fmt(value, digits = 3) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      const num = Number(value);
      if (Math.abs(num) >= 100) return num.toFixed(0);
      if (Math.abs(num) >= 10) return num.toFixed(1);
      return num.toFixed(digits);
    }}

    function timeLabel(value) {{
      const parts = String(value).split(' ');
      return parts.length > 1 ? parts[1].slice(0, 5) : String(value);
    }}

    function option(select, value, label = value) {{
      const node = document.createElement('option');
      node.value = value;
      node.textContent = label;
      select.appendChild(node);
    }}

    function initControls() {{
      DATA.metrics.forEach(metric => option(metricSelect, metric));
      DATA.scenarios.forEach(s => option(scenarioSelect, s.id, `${{s.id}} - ${{s.name}}`));
      DATA.conditions.forEach(condition => option(conditionSelect, condition));
      DATA.seeds.forEach(seed => option(seedSelect, seed));
      metricSelect.value = 'MAD_C';
      scenarioSelect.value = 'S3';
      conditionSelect.value = DATA.conditions.includes('C3') ? 'C3' : DATA.conditions[0];
      seedSelect.value = DATA.seeds.includes(42) ? 42 : DATA.seeds[0];
      [metricSelect, scenarioSelect, conditionSelect, seedSelect].forEach(el => el.addEventListener('change', update));
    }}

    function aggregateRow(condition, scenarioId) {{
      return DATA.aggregate.find(row => row.condition === condition && row.scenario_id === scenarioId);
    }}

    function selectedRun() {{
      const seed = Number(seedSelect.value);
      return DATA.runs.find(run =>
        run.condition === conditionSelect.value &&
        run.scenario_id === scenarioSelect.value &&
        run.seed === seed
      ) || DATA.runs[0];
    }}

    function drawAxes(svg, width, height, margin, yTicks, yScale) {{
      let out = '';
      for (const tick of yTicks) {{
        const y = yScale(tick);
        out += `<line x1="${{margin.left}}" y1="${{y}}" x2="${{width - margin.right}}" y2="${{y}}" stroke="${{COLORS.grid}}" />`;
        out += `<text x="${{margin.left - 8}}" y="${{y + 4}}" text-anchor="end" fill="${{COLORS.axis}}" font-size="11">${{fmt(tick, 2)}}</text>`;
      }}
      out += `<line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="${{COLORS.axis}}" />`;
      out += `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="${{COLORS.axis}}" />`;
      return out;
    }}

    function extent(values) {{
      const nums = values.filter(v => v !== null && v !== undefined && Number.isFinite(Number(v))).map(Number);
      if (!nums.length) return [0, 1];
      let min = Math.min(...nums);
      let max = Math.max(...nums);
      if (Math.abs(max - min) < 1e-9) {{
        min -= 1;
        max += 1;
      }}
      const pad = (max - min) * 0.08;
      return [min - pad, max + pad];
    }}

    function ticks(min, max, count = 5) {{
      const step = (max - min) / Math.max(1, count - 1);
      return Array.from({{length: count}}, (_, i) => min + step * i);
    }}

    function drawLineChart(id, series, yLabel) {{
      const svg = document.getElementById(id);
      const width = 900;
      const height = 320;
      const margin = {{top: 18, right: 22, bottom: 34, left: 54}};
      const points = series.flatMap(s => s.values.map((v, i) => ({{x: i, y: v}}))).filter(p => p.y !== null && p.y !== undefined);
      const maxX = Math.max(1, ...series.map(s => s.values.length - 1));
      const [minY, maxY] = extent(points.map(p => p.y));
      const xScale = x => margin.left + (x / maxX) * (width - margin.left - margin.right);
      const yScale = y => height - margin.bottom - ((Number(y) - minY) / (maxY - minY)) * (height - margin.top - margin.bottom);
      let out = drawAxes(svg, width, height, margin, ticks(minY, maxY), yScale);
      for (const s of series) {{
        const d = s.values.map((v, i) => v === null || v === undefined ? null : `${{i === 0 ? 'M' : 'L'}} ${{xScale(i).toFixed(2)}} ${{yScale(v).toFixed(2)}}`).filter(Boolean).join(' ');
        out += `<path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="${{s.width || 2}}" />`;
      }}
      const first = series[0]?.labels?.[0] || '';
      const last = series[0]?.labels?.[series[0].labels.length - 1] || '';
      out += `<text x="${{margin.left}}" y="${{height - 10}}" fill="${{COLORS.axis}}" font-size="11">${{timeLabel(first)}}</text>`;
      out += `<text x="${{width - margin.right}}" y="${{height - 10}}" text-anchor="end" fill="${{COLORS.axis}}" font-size="11">${{timeLabel(last)}}</text>`;
      out += `<text x="12" y="18" fill="${{COLORS.axis}}" font-size="11">${{yLabel}}</text>`;
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = out;
    }}

    function drawMetricChart(metric) {{
      const svg = document.getElementById('metricChart');
      const width = 900;
      const height = 320;
      const margin = {{top: 18, right: 18, bottom: 42, left: 58}};
      const scenarios = DATA.scenarios.map(s => s.id);
      const values = [];
      scenarios.forEach(id => {{
        ['C2', 'C3'].forEach(condition => {{
          const row = aggregateRow(condition, id);
          if (row) values.push(row[`${{metric}}_mean`]);
        }});
      }});
      const [minRaw, maxY] = extent(values.concat([0]));
      const minY = Math.min(0, minRaw);
      const yScale = y => height - margin.bottom - ((Number(y) - minY) / (maxY - minY)) * (height - margin.top - margin.bottom);
      const plotW = width - margin.left - margin.right;
      const groupW = plotW / scenarios.length;
      const barW = Math.min(32, groupW * 0.28);
      let out = drawAxes(svg, width, height, margin, ticks(minY, maxY), yScale);
      scenarios.forEach((scenario, i) => {{
        const center = margin.left + groupW * i + groupW / 2;
        ['C2', 'C3'].forEach((condition, j) => {{
          const row = aggregateRow(condition, scenario);
          const value = row ? row[`${{metric}}_mean`] : null;
          if (value === null || value === undefined) return;
          const x = center + (j === 0 ? -barW - 3 : 3);
          const y = yScale(value);
          const h = yScale(0) - y;
          out += `<rect x="${{x}}" y="${{Math.min(y, yScale(0))}}" width="${{barW}}" height="${{Math.abs(h)}}" fill="${{condition === 'C2' ? COLORS.c2 : COLORS.c3}}" rx="3" />`;
          out += `<text x="${{x + barW / 2}}" y="${{y - 5}}" text-anchor="middle" fill="${{COLORS.axis}}" font-size="10">${{fmt(value, 2)}}</text>`;
        }});
        out += `<text x="${{center}}" y="${{height - 16}}" text-anchor="middle" fill="${{COLORS.axis}}" font-size="12">${{scenario}}</text>`;
      }});
      svg.setAttribute('viewBox', `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = out;
    }}

    function renderDeltaTable(metric) {{
      const rows = DATA.scenarios.map(s => {{
        const c2 = aggregateRow('C2', s.id);
        const c3 = aggregateRow('C3', s.id);
        const c2v = c2 ? c2[`${{metric}}_mean`] : null;
        const c3v = c3 ? c3[`${{metric}}_mean`] : null;
        const delta = c2v === null || c3v === null ? null : c3v - c2v;
        const improvement = delta === null ? null : -delta;
        return {{scenario: s.id, name: s.name, c2v, c3v, delta, improvement}};
      }});
      document.getElementById('deltaTable').innerHTML = `
        <thead><tr><th>Scenario</th><th class="num">C2</th><th class="num">C3</th><th class="num">C3-C2</th></tr></thead>
        <tbody>${{rows.map(row => {{
          const cls = row.delta === null ? '' : (row.improvement >= 0 ? 'good' : 'bad');
          return `<tr><td><strong>${{row.scenario}}</strong><br><span class="muted">${{row.name}}</span></td><td class="num">${{fmt(row.c2v)}}</td><td class="num">${{fmt(row.c3v)}}</td><td class="num ${{cls}}">${{fmt(row.delta)}}</td></tr>`;
        }}).join('')}}</tbody>`;
    }}

    function renderKpis(run, metric) {{
      const row = aggregateRow(run.condition, run.scenario_id);
      const selectedValue = row ? row[`${{metric}}_mean`] : run.metrics[metric];
      const totalRuns = DATA.runs.length;
      const updates = run.summary.num_supervision_updates ?? 0;
      const energy = run.metrics.CE_kWh;
      const mad = run.metrics.MAD_C;
      const stability = run.metrics.OC_count;
      document.getElementById('kpis').innerHTML = [
        ['Formal runs', totalRuns, `${{DATA.conditions.join(' / ')}} across ${{DATA.scenarios.length}} scenarios`],
        ['Selected metric mean', fmt(selectedValue), `${{run.condition}} ${{run.scenario_id}} · ${{metric}}`],
        ['Run MAD', `${{fmt(mad)}} C`, 'Mean absolute deviation'],
        ['Run energy', `${{fmt(energy)}} kWh`, 'Cumulative heater energy'],
        ['Overshoot count', fmt(stability, 0), 'Per-run control stability'],
        ['Supervisor updates', fmt(updates, 0), run.summary.effective_supervisor_mode || 'agent_auto']
      ].map(([label, value, sub]) => `<div class="kpi"><div class="label">${{label}}</div><div class="value">${{value}}</div><div class="sub">${{sub}}</div></div>`).join('');
    }}

    function renderScenarioNote(run) {{
      const notes = run.notes || {{}};
      const lines = [
        `<strong>${{run.scenario_id}} - ${{run.scenario_name}}</strong>`,
        `Command: ${{run.scenario_command}}`,
        notes.disturbance ? `Disturbance: ${{notes.disturbance}}` : null,
        notes.tariff_modifier ? `Tariff: ${{notes.tariff_modifier}}` : null,
        notes.failure_injection ? `Failure injection: ${{notes.failure_injection}}` : null,
        `Source: ${{DATA.generated_from[run.condition]}}`
      ].filter(Boolean);
      document.getElementById('scenarioNote').innerHTML = lines.map(line => `<div>${{line}}</div>`).join('');
    }}

    function renderDecisionTable(run) {{
      const rows = (run.supervision || []).slice().reverse();
      document.getElementById('decisionTable').innerHTML = `
        <thead><tr><th>Time</th><th>Action</th><th>Reason</th><th class="num">Weights</th></tr></thead>
        <tbody>${{rows.map(row => `
          <tr>
            <td>${{timeLabel(row.time)}}<br><span class="muted">${{row.decision_source || ''}} ${{row.decision_mode || ''}}</span></td>
            <td>${{row.action || 'n/a'}}</td>
            <td>${{row.reason || row.rationale || 'n/a'}}</td>
            <td class="num">${{fmt(row.comfort_weight_after, 2)}} / ${{fmt(row.energy_weight_after, 2)}} / ${{fmt(row.response_weight_after, 2)}}</td>
          </tr>`).join('')}}</tbody>`;
    }}

    function update() {{
      const metric = metricSelect.value;
      const run = selectedRun();
      const ts = run.timeseries || [];
      const labels = ts.map(row => row.time);
      renderKpis(run, metric);
      drawMetricChart(metric);
      renderDeltaTable(metric);
      renderScenarioNote(run);
      renderDecisionTable(run);
      drawLineChart('temperatureChart', [
        {{labels, values: ts.map(row => row.Ti_C), color: COLORS.ti, width: 2.4}},
        {{labels, values: ts.map(row => row.Tsp_C), color: COLORS.tsp, width: 2}},
        {{labels, values: ts.map(row => row.Ta_C), color: COLORS.ta, width: 1.8}}
      ], 'Temperature C');
      drawLineChart('powerChart', [
        {{labels, values: ts.map(row => row.Phea_W), color: COLORS.power, width: 2.2}},
        {{labels, values: ts.map(row => row.tariff_USD_per_kWh ? row.tariff_USD_per_kWh * 100 : null), color: COLORS.tariff, width: 2}}
      ], 'W / cents per kWh');
      drawLineChart('weightsChart', [
        {{labels, values: ts.map(row => row.comfort_weight), color: COLORS.comfort, width: 2.2}},
        {{labels, values: ts.map(row => row.energy_weight), color: COLORS.energy, width: 2.2}},
        {{labels, values: ts.map(row => row.response_weight), color: COLORS.response, width: 2.2}}
      ], 'Weight');
    }}

    initControls();
    update();
  </script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    DESTINATION.write_text(render_html(data), encoding="utf-8")
    print(f"Dashboard written to {DESTINATION}")


if __name__ == "__main__":
    main()
