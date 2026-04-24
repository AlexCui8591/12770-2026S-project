# C2 / C3 Agent-PID Experiment Design

本文档整理当前仓库中 C2 与 C3 的真实实验实现。重点说明 agent 设计、数据生成、PID 调用链、日志输出和两组实验的区别。

## 1. 实验总入口

C2 和 C3 都通过 `simulation.runner` 批量运行：

```powershell
# C2 full experiment: 6 scenarios x 5 seeds = 30 runs
.\.venv\Scripts\python.exe -m simulation.runner --condition C2 --scenario all --runs 5 --supervisor-mode agent_auto --output-dir outputs\full_experiment_c2

# C3 full experiment: 6 scenarios x 5 seeds = 30 runs
.\.venv\Scripts\python.exe -m simulation.runner --condition C3 --scenario all --runs 5 --supervisor-mode agent_auto --output-dir outputs\full_experiment_c3
```

如需严格要求 agent 可用，不允许降级到规则 fallback：

```powershell
.\.venv\Scripts\python.exe -m simulation.runner --condition C2 --scenario all --runs 5 --supervisor-mode agent_only --output-dir outputs\full_experiment_c2
.\.venv\Scripts\python.exe -m simulation.runner --condition C3 --scenario all --runs 5 --supervisor-mode agent_only --output-dir outputs\full_experiment_c3
```

runner 的核心流程：

```text
simulation.runner
  -> load_experiment_settings()
  -> build planned runs
  -> build_scenario_run()
  -> build PID tuning cache from Day 1
  -> run C2 or C3 simulator
  -> save timeseries, supervision log, metrics, summary
```

`--resume` 会跳过已有 `summary.json` 的 run。`--progress on` 默认开启，会显示总进度、当前 run、elapsed 和 ETA。

## 2. 共同实验设置

### 2.1 Run matrix

正式配置来自 `config/experiment.yaml`：

| Item | Value |
|---|---:|
| Conditions | C0, C1, C2, C3 |
| C2/C3 seeds | 42, 123, 456, 789, 1024 |
| Scenarios | S1-S6 |
| Runs per condition | 5 |
| C2 full runs | 30 |
| C3 full runs | 30 |

六个 scenario：

| Scenario | Name | Duration | Main disturbance |
|---|---|---:|---|
| S1 | Steady-State Tracking | 120 min | None |
| S2 | Weather Disturbance | 180 min | Outdoor temperature drops by 10C after minute 30 |
| S3 | Cost Optimization | 720 min | 3x tariff spike in the middle window |
| S4 | Occupancy Adaptation | 360 min | Evening comfort demand |
| S5 | Multi-Disturbance Stress | 1440 min | Multi-period weather and tariff stress |
| S6 | LLM Failure Resilience | 240 min | 20% forced fallback points |

### 2.2 Scenario data

Scenario construction lives in `simulation/scenarios.py`.

For each seed:

1. Generate a synthetic 24-hour Day 2 outdoor temperature profile.
2. Expand hourly outdoor temperatures to 1-minute resolution.
3. Slice the scenario-specific time window.
4. Build scenario setpoint schedule.
5. Build tariff schedule.
6. Apply scenario disturbances.
7. For S6, generate forced fallback steps.

Important implementation detail: the scenario command text still says values such as 21C or 22C, but the simulation uses the existing demo heating scale around 38-41C. This is intentional because the current 1R1C thermal model is heating-only; using 21-22C directly would make the heater mostly idle and weaken the experiment.

### 2.3 PID initialization

C2 and C3 do not start from arbitrary PID gains. For each seed, runner first builds a Day 1 PID tuning cache:

```text
seed -> synthetic Day 1 temperature -> grid search -> best Kp/Ki/Kd
```

The resulting PID gains are reused as the initial gains for all scenarios with that seed.

Each run therefore has:

```json
{
  "initial_pid": {
    "KP": "...",
    "KI": "...",
    "KD": "..."
  }
}
```

This makes C2 and C3 comparable because they inherit the same tuned PID starting point.

### 2.4 Thermal model and PID loop

Both C2 and C3 use the same 1R1C building model and PID control style:

```text
every 1 minute:
  read current indoor temperature Ti
  read current setpoint Tsp
  compute PID error = Tsp - Ti
  compute heater power Phea
  saturate Phea to [0, PMAX]
  update 1R1C building temperature
```

The supervised variables are only PID gains:

```text
Kp, Ki, Kd
```

The supervised variables are not:

```text
setpoint
cost weights
thermal model parameters
tariff schedule
```

## 3. C2 Design: Reactive Agent-PID Supervision

### 3.1 Goal

C2 tests whether an LLM agent can improve PID behavior using only recent closed-loop telemetry.

C2 is reactive only:

```text
No forecast
No user schedule tools
No proactive reasoning
No online setpoint changes
No online cost-weight changes
```

### 3.2 Agent interface

C2 agent implementation:

```text
agent/supervisor.py
agent/supervisor_prompt.txt
```

The C2 prompt constrains the model to:

```json
{
  "action": "hold | set_pid",
  "kp": "number | null",
  "ki": "number | null",
  "kd": "number | null",
  "rationale": "short explanation"
}
```

Allowed actions:

| Action | Meaning |
|---|---|
| hold | Keep current PID gains |
| set_pid | Request one or more PID gain changes |

C2 exposes only one tool to the agent:

```text
get_pid_telemetry(room)
```

The supervisor agent uses Ollama through an OpenAI-compatible endpoint:

```text
base_url = http://localhost:11434/v1
model = qwen2.5:7b
```

### 3.3 Telemetry window

Every 5 minutes, C2 summarizes the most recent 5-minute control window.

Telemetry sent to the agent includes:

| Field | Meaning |
|---|---|
| mean_abs_error | Mean absolute tracking error |
| mean_error | Signed mean error |
| max_abs_error | Maximum absolute tracking error |
| power_mean_W | Mean heater power |
| power_std_W | Heater power variability |
| last_power_W | Last heater power in the window |
| energy_kWh | Energy used in the window |
| error_sign_changes | Oscillation proxy |
| last_error | Latest tracking error |
| current_ti | Current indoor temperature |
| current_tsp | Current setpoint |

Before calling the LLM, the simulator also writes the current state into `agent.mock_pid.DEFAULT_CONTROLLER`. This makes `get_pid_telemetry(room)` return the current simulator snapshot instead of stale mock data.

### 3.4 C2 supervision loop

C2 control flow:

```text
for each simulation minute:
  if minute % 5 == 0:
    summarize last 5 minutes
    sync simulator state into mock_pid.DEFAULT_CONTROLLER
    call C2 reactive supervisor agent
    validate JSON decision
    clip gain changes
    write new Kp/Ki/Kd into PID loop

  run one PID step
  update building temperature
```

PID safety constraints:

| Gain | Range | Max single update |
|---|---:|---:|
| Kp | 0.05 to 15.0 | 0.30 |
| Ki | 0.0 to 5.0 | 0.20 |
| Kd | 0.0 to 3.0 | 0.20 |

The agent can request larger changes, but the simulator enforces these bounds before applying them.

### 3.5 Fallback behavior

C2 supports three modes:

| Mode | Behavior |
|---|---|
| rules | Always use rule-based supervisor |
| agent_auto | Use agent, fall back to rules if agent fails |
| agent_only | Use agent only; fail the run if agent is unavailable |

Rule fallback is also used in S6 failure injection. In that case the log marks:

```text
decision_source = injected_rule_failure
```

Normal agent decisions are logged as:

```text
decision_source = agent
```

Agent prompt-injection fallback decisions are logged as:

```text
decision_source = agent_prompt_injection
```

Rule fallback due to agent error is logged as:

```text
decision_source = rules_fallback
```

### 3.6 C2 outputs

Each C2 run writes:

```text
timeseries.csv
supervision_log.csv
metrics.json
summary.json
scenario_notes.json
```

Key `supervision_log.csv` fields:

| Field | Meaning |
|---|---|
| time | Supervision timestamp |
| current_ti_C | Current indoor temperature |
| current_tsp_C | Current setpoint |
| mean_abs_error_C | Recent tracking error |
| power_mean_W | Recent heater use |
| kp_before / ki_before / kd_before | Gains before decision |
| action | hold, set_pid, or rule action label |
| decision_source | agent, rules, fallback, injected failure |
| rationale | Agent or fallback explanation |
| kp_after / ki_after / kd_after | Applied gains after clipping |

Metrics:

| Metric | Meaning |
|---|---|
| MAD_C | Mean absolute deviation |
| CVR_fraction | Comfort violation ratio |
| CE_kWh | Energy consumption |
| EC_USD | Electricity cost |
| RT_s | Rise time |
| ST_s | Settling time |
| OC_count | Oscillation count |
| MO_C | Maximum overshoot after setpoint changes |

## 4. C3 Design: Proactive Agent-PID Supervision

### 4.1 Goal

C3 extends C2 by giving the agent forward-looking context. The goal is to test whether the LLM can tune PID gains before expected disturbances or user comfort events occur.

C3 still only controls PID gains:

```text
Kp, Ki, Kd
```

C3 still does not change:

```text
setpoint
cost weights
thermal model
tariff data
```

### 4.2 Agent interface

C3 agent implementation:

```text
agent/proactive_supervisor.py
agent/proactive_supervisor_prompt.txt
```

C3 prompt constrains the model to:

```json
{
  "mode": "hold | proactive | reactive",
  "action": "hold | set_pid",
  "kp": "number | null",
  "ki": "number | null",
  "kd": "number | null",
  "rationale": "short explanation"
}
```

The extra `mode` field is important:

| Mode | Meaning |
|---|---|
| hold | No gain change |
| proactive | Gain change based on future context |
| reactive | Gain change based on recent telemetry |

Allowed tools:

```text
get_pid_telemetry(room)
get_weather_forecast(city, hours_ahead)
get_schedule(user_id, date)
```

Compared with C2, C3 can inspect future weather and upcoming schedule events.

### 4.3 Proactive data context

C3 builds two layers of proactive context.

First, the local simulator computes a compact forecast summary:

| Field | Meaning |
|---|---|
| Ta_now_C | Current outdoor temperature |
| Ta_future_C | Outdoor temperature at horizon end |
| Ta_mean_future_C | Mean future outdoor temperature |
| Ta_delta_next_1h_C | Future temperature change over 1 hour |
| next_habit_label | Next synthetic habit event |
| next_habit_type | cooler, warmer, stability, or none |
| minutes_to_next_habit | Minutes until the next event |

Second, the agent tool layer receives temporary synthetic overrides:

```text
get_weather_forecast -> scenario future outdoor temperature
get_schedule -> synthetic upcoming habit events
```

This is important because the agent should reason over the simulated experiment context, not live internet weather or unrelated calendar data.

### 4.4 C3 supervision loop

C3 control flow:

```text
for each simulation minute:
  if minute % 5 == 0:
    summarize last 5 minutes
    summarize next 1 hour forecast
    sync simulator state into mock_pid.DEFAULT_CONTROLLER
    install synthetic tool overrides for forecast and schedule
    call C3 proactive supervisor agent
    validate JSON decision
    clip gain changes
    clear tool overrides
    write new Kp/Ki/Kd into PID loop

  run one PID step
  update building temperature
```

The gain bounds and max-step limits are the same as C2.

### 4.5 C3 fallback behavior

C3 supports the same `rules`, `agent_auto`, and `agent_only` modes.

When using rules, C3 first applies proactive rules:

| Trigger | Rule behavior |
|---|---|
| Future outdoor cooling | Increase Kp/Ki to prepare heating response |
| Future outdoor warming | Reduce Kp/Ki to avoid overshoot |
| Warmer habit event | Increase Kp/Ki |
| Stability habit event | Reduce Kp and increase Kd |
| Cooler habit event | Reduce Kp/Ki |

If no proactive rule triggers, it falls back to the same reactive logic used by C2.

When using the agent:

```text
agent decision succeeds -> decision_source = agent
tool-calling parse fails but prompt fallback works -> decision_source = agent_prompt_injection
agent fails under agent_auto -> decision_source = rules_fallback
agent fails under agent_only -> run fails
```

### 4.6 C3 outputs

C3 writes the same core files as C2:

```text
timeseries.csv
supervision_log.csv
metrics.json
summary.json
scenario_notes.json
```

C3 `supervision_log.csv` adds proactive fields:

| Field | Meaning |
|---|---|
| decision_mode | proactive, reactive, or hold |
| Ta_now_C | Current outdoor temperature |
| Ta_future_C | Future outdoor temperature |
| Ta_mean_future_C | Mean future outdoor temperature |
| Ta_delta_next_1h_C | 1-hour forecast delta |
| next_habit_label | Upcoming habit label |
| next_habit_type | warmer, cooler, stability, or none |
| minutes_to_next_habit | Time to event |
| decision_source | agent, rules, fallback |
| rationale | Agent or fallback explanation |

`summary.json` includes:

```text
decision_source_counts
decision_mode_counts
num_proactive_updates
num_reactive_updates
num_hold_updates
```

These fields are the main evidence for whether C3 is actually using proactive reasoning.

## 5. C2 vs C3 Comparison

| Dimension | C2 | C3 |
|---|---|---|
| Agent type | Reactive PID supervisor | Proactive + reactive PID supervisor |
| Main file | `agent/supervisor.py` | `agent/proactive_supervisor.py` |
| Prompt | `agent/supervisor_prompt.txt` | `agent/proactive_supervisor_prompt.txt` |
| Tools | `get_pid_telemetry` | `get_pid_telemetry`, `get_weather_forecast`, `get_schedule` |
| Inputs | Recent 5-minute telemetry | Recent telemetry + future weather + schedule |
| Output action | `hold` or `set_pid` | `hold` or `set_pid` |
| Output mode | None | `hold`, `reactive`, `proactive` |
| Online setpoint change | No | No |
| Online cost-weight change | No | No |
| Gain safety bounds | Yes | Yes |
| Rule fallback | Yes | Yes |
| S6 failure injection | Yes | Not currently used in runner summary as forced C3 fallback |

The main scientific comparison:

```text
C2 tests whether recent telemetry alone is enough for useful online PID tuning.
C3 tests whether adding forecast and schedule context improves anticipation and stability.
```

## 6. Interpretation Strategy

When reading results, check these in order:

1. `summary.json`
   - Confirm `effective_supervisor_mode`.
   - Confirm `decision_source_counts`.
   - For C3, inspect `decision_mode_counts`.

2. `supervision_log.csv`
   - For C2, inspect whether agent decisions mostly `hold` or repeatedly tune gains.
   - For C3, inspect whether `proactive` decisions occur near future weather or habit events.

3. `timeseries.csv`
   - Check whether `Kp/Ki/Kd` drift continuously or stabilize.
   - Check whether temperature tracking improves after supervision updates.

4. `aggregate_metrics.csv`
   - Compare C2 vs C3 over all seeds and scenarios.
   - Do not over-interpret a single seed.

Expected result pattern:

```text
If C3 is useful:
  lower MAD_C / CVR_fraction under disturbance scenarios
  comparable or lower CE_kWh / EC_USD
  fewer large overshoots or oscillations

If C3 is too aggressive:
  higher MO_C
  worse CVR_fraction
  too many proactive updates
  Kp/Ki drift across the run
```

## 7. Known Limitations

1. The current plant model is simplified 1R1C and heating-only.
2. The working temperature scale is the demo scale around 38-41C.
3. C2 and C3 only tune PID gains; setpoint and cost weights stay fixed online.
4. LLM outputs are constrained by JSON schema, then clipped by deterministic safety bounds.
5. `agent_auto` can hide occasional LLM failures by falling back to rules; use `agent_only` when measuring pure agent reliability.
6. C3 proactive quality depends heavily on whether the agent uses forecast/schedule context conservatively.

## 8. Recommended Reporting Language

C2 can be described as:

> C2 is a reactive LLM-in-the-loop PID supervision condition. The agent observes recent closed-loop telemetry every 5 minutes and may adjust only PID gains, while the setpoint schedule and cost weights remain fixed.

C3 can be described as:

> C3 extends C2 with proactive context. In addition to recent PID telemetry, the agent can inspect synthetic forecast and schedule tools derived from the current simulation scenario, allowing it to adjust PID gains ahead of expected disturbances or comfort events.

The clean experimental contrast is:

> C2 measures reactive online gain tuning from telemetry alone, while C3 measures whether forecast-aware and schedule-aware supervision provides additional benefit over reactive tuning.

