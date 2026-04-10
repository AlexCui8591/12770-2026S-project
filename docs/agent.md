# Agent Specification

Agent 负责将用户的自然语言温控指令转换为结构化 JSON 配置，供下游 PID 控制器消费。

**Agent 做什么：**
- 理解用户意图（目标温度、房间、模式、约束）
- 通过 tool use 主动获取天气、用户习惯、电价等外部信息
- 综合上下文与 tool 数据，生成保守、合法的控制配置
- 输出纯 JSON，字段定义见下文

**Agent 不做什么：**
- 不直接调用 Home Assistant API
- 不输出 PID 参数（Kp / Ki / Kd）
- 不进行闭环控制或实时温度调节

---

## Input Schema

Agent 的每次调用包含两个顶层字段：

```json
{
  "user_command": "今晚卧室凉一点，但不要太费电",
  "current_context": {
    "room": "bedroom",
    "current_temperature": 27.2,
    "current_hvac_mode": "cool",
    "current_fan_mode": "medium",
    "occupied": true,
    "window_open": false,
    "time": "2026-03-29T22:30:00"
  }
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_command` | string | 是 | 用户原始自然语言指令 |
| `current_context.room` | string | 是 | 目标房间 ID，需与 `rooms.yaml` 一致 |
| `current_context.current_temperature` | float | 是 | 当前室温（°C） |
| `current_context.current_hvac_mode` | string | 是 | 当前 HVAC 模式：`cool` / `heat` / `off` / `auto` |
| `current_context.current_fan_mode` | string | 是 | 当前风速：`low` / `medium` / `high` / `auto` |
| `current_context.occupied` | bool | 是 | 房间是否有人 |
| `current_context.window_open` | bool | 是 | 窗户是否打开 |
| `current_context.time` | string | 是 | ISO 8601 当前时间，用于判断时段 |

---

## Output Schema

Agent 输出必须是合法 JSON，不含 markdown 代码块，不含注释。

```json
{
  "room": "bedroom",
  "target_temperature": 24.5,
  "hvac_mode": "cool",
  "preset_mode": "eco",
  "fan_mode": "low",
  "deadband": 0.5,
  "valid_until": null,
  "solar_radiation": {
    "ghi": 312.0,
    "ghi_unit": "W/m^2",
    "dni": 87.53,
    "dni_unit": "W/m^2",
    "dhi": 242.61,
    "dhi_unit": "W/m^2",
    "data_status": "ok"
  },
  "reason": "夜间希望更凉爽，同时兼顾节能"
}
```

### 字段说明

| 字段 | 类型 | 说明 | 允许值 |
|------|------|------|--------|
| `room` | string | 目标房间 ID | 与输入 `room` 一致 |
| `target_temperature` | float | 目标温度（°C） | `[20.0, 28.0]` |
| `hvac_mode` | string | HVAC 模式 | `cool` / `heat` / `off` / `auto` |
| `preset_mode` | string | 高层控制策略 | `eco` / `sleep` / `comfort` / `none` |
| `fan_mode` | string | 风速模式 | `low` / `medium` / `high` / `auto` |
| `deadband` | float | 控制死区（°C），减少频繁启停 | `0.0` – `2.0`，推荐 `0.5` |
| `valid_until` | string / null | 配置有效截止时间（ISO 8601），`null` 表示持续有效 | ISO 8601 或 `null` |
| `solar_radiation` | object | 下发给 PID 的太阳辐射观测值 | 包含 `ghi/dni/dhi/data_status` |
| `reason` | string | 简短解释，便于调试与日志追踪 | 任意字符串 |

---

## Output Constraints

Agent 生成输出时必须遵守以下规则：

1. **只输出合法 JSON**，不允许 markdown 包裹，不允许行内注释
2. **不直接输出 HA API 调用**（如 `climate.set_temperature`）
3. **不输出 PID 参数**（Kp / Ki / Kd 由 `config/pid.yaml` 静态配置）
4. **温度必须在 `[20, 28]°C`**，超出范围时截断到边界值
5. **指令模糊时优先保守配置**：小幅调整，不激进
6. **窗户打开（`window_open=true`）时**，建议 `hvac_mode: off` 或不降温
7. **用户要求与上下文冲突时**，输出最安全方案并在 `reason` 字段说明

---

## System Prompt

以下是 `agent/prompt.txt` 的完整内容，用于初始化 LLM：

```text
你是一个智能家居温控 Agent。

你的任务是：
1. 解析用户的自然语言温控需求
2. 结合当前上下文，输出结构化温控配置
3. 输出必须为 JSON
4. 不要输出额外解释文字
5. 不要直接控制设备
6. 不要输出 PID 参数
7. 你的输出将被下游 PID 控制器使用

你可以使用以下 tools 获取额外信息：
- get_weather(city): 查询室外天气（温度、湿度、天气状况）
- get_user_habits(user_id, time_of_day): 查询用户历史温控偏好
- get_room_status(room): 查询房间实时传感器状态
- get_energy_price(time): 查询当前电价（峰谷）
- get_schedule(user_id, date): 查询用户日程安排
- get_solar_radiation(time, latitude?, longitude?, timezone?): 查询本地太阳辐射（默认匹兹堡）

Tool 使用规则：
- 按需调用，不强制使用
- 每次推理最多调用 3 次 tool
- tool 调用失败时，基于已有上下文继续推理
- 在 reason 字段中注明参考了哪些 tool 数据

输出字段包括：
- room
- target_temperature
- hvac_mode
- preset_mode
- fan_mode
- deadband
- valid_until
- solar_radiation
- reason

规则：
- target_temperature 必须在 20 到 28 摄氏度之间
- 用户说"凉一点"时，可在当前基础上降低 0.5 到 1.5 度
- 用户说"省电"时，preset_mode 优先设为 eco
- 用户说"睡觉"时，preset_mode 优先设为 sleep，fan_mode 不宜过高
- 当上下文显示 window_open=true 时，优先输出保守策略
- 室外温度低时，不需要过度制冷
- 峰时电价时，优先 eco 模式
- 用户即将外出时，可提前切换 eco
- 输出必须是合法 JSON，不允许输出 markdown，不允许输出注释
```

**扩展提示词的建议：**
- 新增房间时，在规则段追加 `room` 相关描述
- 新增 `preset_mode` 枚举值时，同步更新"输出字段"说明
- 不要在 System Prompt 中硬编码具体温度阈值，改用动态注入上下文

---

## Tool Use

Agent 支持通过 tool use（LLM function calling）在推理过程中主动获取外部信息。

### 可用 Tools

#### `get_weather(city: string)`

查询指定城市的室外天气。

```json
// 返回值
{
  "outdoor_temp": 5.2,
  "humidity": 72,
  "condition": "rain",
  "wind_speed_kmh": 15
}
```

**决策影响：** 室外低温 → 散热快，不需过度制冷；室外高温高湿 → 需更积极制冷。

#### `get_user_habits(user_id: string, time_of_day: string)`

查询用户历史温控偏好。`time_of_day` 取值：`"morning"` / `"afternoon"` / `"night"`。

```json
// 返回值
{
  "preferred_sleep_temp": 23.0,
  "preferred_day_temp": 25.0,
  "bedtime": "23:00",
  "wake_time": "07:30",
  "fan_preference": "low"
}
```

**决策影响：** 用户说"凉一点"时，参考历史偏好而非机械降 1°C。

#### `get_room_status(room: string)`

查询房间实时传感器状态，可替代调用方手动组装 `current_context`。

```json
// 返回值
{
  "current_temperature": 27.2,
  "current_humidity": 65,
  "current_hvac_mode": "cool",
  "current_fan_mode": "medium",
  "occupied": true,
  "window_open": false
}
```

**决策影响：** Agent 可自主获取最新房间状态，不依赖调用方的上下文时效性。

#### `get_energy_price(time: string)`

查询指定时间的电价（ISO 8601 格式）。

```json
// 返回值
{
  "price_per_kwh": 0.85,
  "tier": "peak",
  "next_off_peak": "2026-03-30T00:00:00"
}
```

**决策影响：** 峰时 → 倾向 `preset_mode: eco`；谷时 → 可适度提升舒适度。

#### `get_schedule(user_id: string, date: string)`

查询用户当天日程安排。

```json
// 返回值
{
  "events": [
    {"type": "leave_home", "at": "08:30"},
    {"type": "return_home", "at": "18:00"}
  ]
}
```

**决策影响：** 即将外出 → 提前切 eco；即将到家 → 提前预冷/预热。

#### `get_solar_radiation(time: string, latitude?: float, longitude?: float, timezone?: string)`

查询指定时间和地点的太阳辐射，默认使用匹兹堡（`40.4406, -79.9959`）并按 `America/New_York` 解释输入时间，再转换为 UTC 请求 Open-Meteo 小时级 API。

```json
// 返回值
{
  "source": "Open-Meteo Forecast API",
  "time_standard": "UTC",
  "requested_time": "2026-03-29T14:00:00",
  "resolved_time_utc": "2026-03-29T18:00:00Z",
  "latitude": 40.4406,
  "longitude": -79.9959,
  "ghi": 512.3,
  "ghi_unit": "W/m^2",
  "dni": 621.8,
  "dni_unit": "W/m^2",
  "dhi": 118.4,
  "dhi_unit": "W/m^2",
  "data_status": "ok"
}
```

**决策影响：** 白天高太阳辐射意味着更强日照得热，可减少制热倾向；如果房间日照强，也可提前保守降温。

### Tool Use 规则

1. **按需调用**：不强制每次都调 tool，简单指令（如"关空调"）无需外部信息
2. **上限 3 次/推理**：避免延迟过高，优先选最有价值的 tool
3. **容错**：tool 调用失败时，基于已有 `current_context` 继续推理，不得拒绝输出
4. **透明**：在 `reason` 字段中注明参考了哪些 tool 的数据
5. **不缓存**：每次推理独立调用，不使用上次推理的 tool 结果

### Tool Use 示例

用户说 *"凉一点"*，Agent 调用了 `get_weather` 和 `get_user_habits`：

```json
{
  "room": "bedroom",
  "target_temperature": 24.0,
  "hvac_mode": "cool",
  "preset_mode": "eco",
  "fan_mode": "low",
  "deadband": 0.5,
  "valid_until": null,
  "reason": "室外温度 5°C（get_weather），散热快无需过度制冷；用户睡眠偏好 23°C（get_user_habits），取 24°C 兼顾舒适与节能"
}
```

### 错误处理

| 异常情况 | 处理方式 |
|----------|----------|
| tool 调用超时（>5s） | 放弃该 tool，基于已有上下文推理 |
| tool 返回格式异常 | 忽略该 tool 返回值，记录警告日志 |
| tool 服务不可用 | 同超时处理，在 `reason` 中注明 |

---

## Parsing Rules

模糊语言到结构化字段的映射规则（供 `parser.py` 实现参考）：

| 用户表达 | 对应调整 |
|----------|----------|
| "凉一点" / "凉快点" | `target_temperature` 在当前基础上 −0.5 ~ −1.5°C |
| "暖一点" / "热一点" | `target_temperature` 在当前基础上 +0.5 ~ +1.5°C，`hvac_mode: heat` |
| "省电" / "节能" | `preset_mode: eco` |
| "舒适" / "舒服" | `preset_mode: comfort` |
| "睡觉" / "睡眠" / "晚上" | `preset_mode: sleep`，`fan_mode: low` |
| "风大一点" | `fan_mode: high` |
| "风小一点" / "不要风太大" | `fan_mode: low` |
| "关掉" / "关空调" | `hvac_mode: off` |
| "没人" / "出门了" | `preset_mode: eco`，温度可放宽到安全上限 |
| 指定具体温度（如"24度"）| 直接设为 `target_temperature: 24.0` |
| 时间限定（如"今晚"）| 设置 `valid_until` 为当天结束时间（如 `T23:59:59`） |

---

## Error Handling

`parser.py` 在以下情况需要做防御处理：

| 异常情况 | 处理方式 |
|----------|----------|
| LLM 返回非 JSON 文本 | 记录原始输出，抛出 `ParseError`，不传递给控制器 |
| `target_temperature` 超出 `[20, 28]` | 截断到边界值，在 `reason` 中注明 |
| 必填字段缺失（如 `room`）| 用输入的 `current_context.room` 填充；仍缺失则抛出 `SchemaError` |
| `hvac_mode` 为非法枚举值 | 回退到 `current_hvac_mode`，在 `reason` 中注明 |
| LLM 超时或服务不可用 | 向上抛出，由调用方决定是否使用上一次有效配置 |

---

## Example Call

**输入**

```json
{
  "user_command": "今晚卧室凉一点，但不要太费电",
  "current_context": {
    "room": "bedroom",
    "current_temperature": 27.2,
    "current_hvac_mode": "cool",
    "current_fan_mode": "medium",
    "occupied": true,
    "window_open": false,
    "time": "2026-03-29T22:30:00"
  }
}
```

**System Prompt 注入上述上下文后，LLM 输出**

```json
{
  "room": "bedroom",
  "target_temperature": 24.5,
  "hvac_mode": "cool",
  "preset_mode": "eco",
  "fan_mode": "low",
  "deadband": 0.5,
  "valid_until": null,
  "reason": "夜间希望更凉爽，同时兼顾节能"
}
```

**解析逻辑说明：**
- "凉一点" → 27.2 − 2.7 = 24.5（在 −0.5 ~ −1.5 范围内取中间值）
- "省电" → `preset_mode: eco`
- "今晚" → `valid_until: null`（用户未指定截止，持续有效）
- 夜间睡眠场景 → `fan_mode: low`
