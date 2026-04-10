# PID Controller Specification

PID 控制器负责温度的闭环自动控制。它读取 Agent 输出的目标温度配置，结合传感器实时数据，计算控制量并映射为 Home Assistant 动作。

**控制器做什么：**
- 读取当前温度与目标温度
- 按 PID 公式计算控制量 `u`
- 将 `u` 映射为 HA climate 服务调用
- 执行安全检查（温度限幅、防短循环、窗户联锁）

**控制器不做什么：**
- 不解析自然语言
- 不调用 LLM
- 不修改 Agent 输出的 `target_temperature`

---

## Inputs

每个控制周期，`pid.py` 接收以下输入：

| 字段 | 类型 | 单位 | 来源 |
|------|------|------|------|
| `current_temperature` | float | °C | HA 传感器（`sensor.*_temperature`） |
| `target_temperature` | float | °C | Agent 输出 JSON |
| `deadband` | float | °C | Agent 输出 JSON（默认 `0.5`） |
| `solar_radiation` | object | `W/m^2` | Agent 输出 JSON，用于日照前馈或负载补偿 |
| `dt` | float | 秒 | 控制周期，来自 `config/pid.yaml` |
| `prev_error` | float | °C | 上一周期误差（控制器内部状态） |
| `integral` | float | °C·s | 积分累积值（控制器内部状态） |
| `Kp` | float | — | `config/pid.yaml` |
| `Ki` | float | — | `config/pid.yaml` |
| `Kd` | float | — | `config/pid.yaml` |

---

## PID Formula

除标准温差闭环外，控制器可读取 Agent 输出中的 `solar_radiation` 作为前馈量，补偿太阳得热带来的额外热负荷。

### 误差定义

```text
error = target_temperature - current_temperature
```

正误差 → 当前温度低于目标（需要加热）
负误差 → 当前温度高于目标（需要制冷）

### 标准 PID 公式

```text
u = Kp * error + Ki * integral + Kd * derivative
```

其中：

```text
integral   = integral + error * dt
derivative = (error - prev_error) / dt
```

### 各项物理含义

| 项 | 含义 |
|----|------|
| `Kp * error` | **比例项**：误差越大，控制量越强；即时响应，但单独使用有稳态误差 |
| `Ki * integral` | **积分项**：消除长期稳态误差；过大会导致超调和振荡 |
| `Kd * derivative` | **微分项**：预测误差变化趋势，抑制超调；对噪声敏感 |

---

## Deadband

Deadband（死区）来自 Agent 输出的 `deadband` 字段，默认 `0.5°C`。

当 `|error| <= deadband` 时，控制器**不发出控制动作**，维持当前状态。

**作用：**
- 避免温度在目标值附近反复微小波动导致 HA API 频繁调用
- 减少空调启停次数，延长设备寿命

```python
if abs(error) <= deadband:
    return  # 保持当前状态，不调用 HA
```

---

## Control Output Mapping

`mapper.py` 将控制量 `u` 映射为 HA 动作：

| 控制量范围 | 语义 | HA 动作 |
|-----------|------|---------|
| `u < -2.0` | 强制制冷（误差大，温度远高于目标） | `set_hvac_mode(cool)` + 将设定温度下调 1°C |
| `-2.0 <= u < -0.5` | 普通制冷 | `set_hvac_mode(cool)` + `set_temperature(target)` |
| `-0.5 <= u <= 0.5` | 在 deadband 内，保持 | 无动作 |
| `u > 0.5` | 停止制冷或切换加热 | `set_hvac_mode(off)` 或 `set_hvac_mode(heat)` |

> **注意：** 具体阈值（`-2.0` / `-0.5` / `0.5`）可在 `config/pid.yaml` 中通过 `mapping_thresholds` 字段覆盖。

---

## Safety Module

`safety.py` 在每次控制动作执行前进行检查，对应 [需求.md](../需求.md) §16 的安全建议：

### 1. 温度限幅

`target_temperature` 超出 `[temp_min, temp_max]`（来自 `config/limits.yaml`）时，截断到边界值，记录警告日志。

### 2. 防短循环（Anti-Short-Cycle）

空调在 `anti_cycle_seconds`（默认 180 秒）内不得重复启动或切换模式。控制器维护 `last_action_time`，若间隔不足则跳过本次动作。

### 3. 窗户联锁

`window_open=true` 时，禁止执行制冷动作（`hvac_mode=cool`）。若 Agent 输出 `hvac_mode=cool` 但上下文显示窗户开着，`safety.py` 将动作降级为 `hold`（保持当前状态）并记录日志。

### 4. 手动接管

系统维护一个 `manual_override` 标志位（可通过 HA 自动化或 CLI 设置）。当 `manual_override=true` 时，控制器完全停止自动动作，等待人工恢复。

---

## PID Tuning Guide

### 推荐初始值（典型分体式空调）

| 参数 | 推荐初始值 | 说明 |
|------|-----------|------|
| `Kp` | `1.0` | 每 1°C 误差产生 1 个单位控制量 |
| `Ki` | `0.1` | 积分系数，避免过大导致超调 |
| `Kd` | `0.05` | 微分系数，轻度抑制超调 |
| `deadband` | `0.5` | 死区 0.5°C，减少频繁启停 |
| `dt` | `30` | 控制周期 30 秒 |

### 调参症状对照

| 现象 | 诊断 | 调整方向 |
|------|------|----------|
| 温度长期偏离目标，不收敛 | `Kp` 太小 | 增大 `Kp` |
| 温度剧烈振荡 | `Kp` 太大 或 `Kd` 太小 | 减小 `Kp`，增大 `Kd` |
| 温度缓慢趋近目标，有稳态偏差 | `Ki` 太小 | 增大 `Ki` |
| 温度超调后缓慢回调 | `Ki` 太大 | 减小 `Ki` |
| 启停过于频繁 | `deadband` 太小 | 增大 `deadband` |

---

## State Machine

控制器运行时处于以下四种状态之一：

```text
         manual_override=true
              ┌─────────┐
              ▼         │
┌─────────────────────────────────────┐
│         manual_override             │
└─────────────────────────────────────┘
              │ manual_override=false
              ▼
         ┌─────────┐
    ─────►  idle   ◄──────────────────────┐
         └────┬────┘                      │
              │ |error| > deadband        │
              ▼                           │
         ┌─────────┐     u > 0.5          │
    ─────► cooling │ ──────────────────── ┤
         └─────────┘                      │
                         u < -0.5         │
         ┌─────────┐                      │
    ─────► heating │ ──────────────────── ┘
         └─────────┘
```

| 状态 | 触发条件 | 行为 |
|------|----------|------|
| `idle` | `|error| <= deadband` | 不发出 HA 动作 |
| `cooling` | `u < -0.5` | 执行制冷相关 HA 动作 |
| `heating` | `u > 0.5` | 执行加热相关 HA 动作 |
| `manual_override` | 手动接管标志位置位 | 停止所有自动动作 |
