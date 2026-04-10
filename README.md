# Home Temperature Control Agent

一个面向 Home Assistant 的智能温控系统，通过自然语言指令驱动 PID 控制器自动调节空调/暖气。

---

## Overview

```text
用户自然语言
    ↓
Agent（自然语言理解 + Tool Use）
    ├── get_weather()        → 室外天气
    ├── get_user_habits()    → 用户偏好
    ├── get_room_status()    → 房间传感器
    ├── get_energy_price()   → 峰谷电价
    ├── get_schedule()       → 用户日程
    └── get_solar_radiation() → 本地太阳辐射
    ↓
结构化目标配置（JSON）
    ↓
PID Controller（闭环自动控制）
    ↓
Home Assistant API
    ↓
空调 / 暖气 / 风扇
    ↓
温度传感器反馈
    ↺
```

| 层级 | 职责 |
|------|------|
| **Agent** | 解析用户自然语言，通过 tool use 获取天气/电价/习惯等外部信息，输出结构化控制配置 JSON |
| **PID Controller** | 读取目标温度与当前温度，计算控制量，映射为 HA 动作 |
| **Home Assistant** | 提供设备状态，执行 climate 服务调用 |

---

## Quick Start

**前置条件**

- Python 3.10+
- 运行中的 Home Assistant 实例（本地或远程）
- HA 长期访问令牌（Settings → Profile → Long-Lived Access Tokens）

**安装**

```bash
pip install -r requirements.txt
```

**最小运行示例**

```bash
python main.py \
  --command "卧室凉一点" \
  --room bedroom \
  --ha-url http://homeassistant.local:8123 \
  --ha-token YOUR_TOKEN
```

---

## Project Structure

```text
project/
├── README.md                  # 本文件
├── 需求.md                    # 原始需求文档（中文）
├── docs/
│   ├── agent.md               # Agent 输入/输出规范、System Prompt
│   ├── controller.md          # PID 公式、控制输出映射、安全模块
│   ├── ha_integration.md      # Home Assistant 集成与 API 参考
│   └── config_reference.md   # 配置文件字段完整说明
├── agent/
│   ├── prompt.txt             # Agent System Prompt
│   ├── parser.py              # 自然语言解析，调用 LLM，返回 JSON
│   ├── schema.py              # Agent 输入/输出 schema 定义与校验
│   └── tools.py               # Tool Use 实现（天气、电价、用户习惯等）
├── controller/
│   ├── pid.py                 # PID 算法实现
│   ├── mapper.py              # 控制量 u → HA 动作映射
│   └── safety.py             # 温度限幅、防短循环、窗户联锁
├── ha/
│   ├── client.py              # HA REST API 客户端（含冷却限速）
│   └── services.py            # climate 服务封装
├── config/
│   ├── rooms.yaml             # 房间与 entity_id 映射
│   ├── limits.yaml            # 安全温度范围、冷却时间等
│   └── pid.yaml               # Kp / Ki / Kd / deadband / dt
└── logs/                      # 运行日志（第四阶段启用）
```

---

## How It Works

完整数据流（以卧室示例为例）：

**1. 用户输入**

```text
"今晚卧室凉一点，但不要太费电"
```

**2. Agent 调用 Tools 获取外部信息**

```text
get_weather("Pittsburgh")       → {"outdoor_temp": 5.2, "humidity": 72, ...}
get_user_habits("user1", "night") → {"preferred_sleep_temp": 23.0, ...}
get_energy_price("2026-03-29T22:30:00") → {"tier": "peak", ...}
get_solar_radiation("2026-03-29T14:00:00") → {"ghi": 512.3, "dni": 621.8, "dhi": 118.4, "source": "Open-Meteo Forecast API", ...}
```

**3. Agent 综合推理，输出结构化配置**

```json
{
  "room": "bedroom",
  "target_temperature": 24.0,
  "hvac_mode": "cool",
  "preset_mode": "eco",
  "fan_mode": "low",
  "deadband": 0.5,
  "valid_until": null,
  "reason": "室外 5°C 散热快无需过度制冷；用户睡眠偏好 23°C；峰时电价优先节能，取 24°C 平衡舒适与省电"
}
```

**4. PID 控制器计算**

```text
current_temperature = 27.2
target_temperature  = 24.0
error               = -3.2  →  触发强制制冷区间
```

**5. Home Assistant 执行**

```text
climate.set_hvac_mode(cool)
climate.set_temperature(24)
climate.set_fan_mode(low)
```

**6. 传感器反馈，PID 持续闭环调节**

---

## Configuration

| 文件 | 作用 |
|------|------|
| `config/rooms.yaml` | 定义房间名称与 HA entity_id 的映射 |
| `config/limits.yaml` | 安全温度上下限、API 冷却时间、防短循环时间 |
| `config/pid.yaml` | PID 参数（Kp / Ki / Kd）、deadband、控制周期 dt |

详细字段说明见 [docs/config_reference.md](docs/config_reference.md)。

---

## Development Phases

| 阶段 | 功能 | 关键交付物 |
|------|------|-----------|
| **第一阶段** | 单房间、仅目标温度、基础 PID | `parser.py` + `pid.py` + `set_temperature` |
| **第二阶段** | 加入 `preset_mode` / `fan_mode` / `deadband` | `mapper.py` + `schema.py` 完善 |
| **第三阶段** | 上下文感知（占用状态、窗户、时间段） | `safety.py` + 上下文字段扩展 |
| **第四阶段** | 日志与评估 | `logs/` 完整记录链路 |

---

## Safety Constraints

1. 目标温度必须在 `[20, 28]°C` 范围内
2. Agent 禁止直接输出 PID 参数（Kp / Ki / Kd）
3. 支持手动接管开关，优先级高于自动控制
4. 窗户打开时暂停自动制冷
5. 防止频繁启停（最小开/关间隔）
6. HA API 调用设有冷却时间，避免频繁请求
7. 所有控制行为记录日志，便于审计

---

## Further Reading

- [docs/agent.md](docs/agent.md) — Agent 输入/输出规范、System Prompt、解析规则
- [docs/controller.md](docs/controller.md) — PID 公式、控制输出映射、安全模块
- [docs/ha_integration.md](docs/ha_integration.md) — HA 集成、API 调用示例
- [docs/config_reference.md](docs/config_reference.md) — 配置文件完整字段参考
