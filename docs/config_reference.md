# Configuration Reference

本项目使用三个 YAML 配置文件，位于 `config/` 目录下。启动时系统自动加载并校验所有字段。

| 文件 | 作用 | 读取方 |
|------|------|--------|
| `rooms.yaml` | 定义房间与 HA 实体的映射 | `ha/client.py`，`controller/pid.py` |
| `limits.yaml` | 安全约束、冷却时间、HA 连接信息 | `controller/safety.py`，`ha/client.py` |
| `pid.yaml` | PID 参数与控制周期 | `controller/pid.py` |

---

## rooms.yaml

定义系统管理的房间列表及其对应的 HA 实体。

**带注释示例：**

```yaml
rooms:
  - room_id: bedroom           # 房间唯一 ID，与 Agent 输出的 room 字段一致
    entity_id: climate.bedroom_ac          # HA climate 实体 ID
    display_name: 卧室空调                  # 人类可读名称，用于日志
    sensor_entity_id: sensor.bedroom_temperature  # 温度传感器实体 ID

  - room_id: living_room
    entity_id: climate.living_room_ac
    display_name: 客厅空调
    sensor_entity_id: sensor.living_room_temperature
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `room_id` | string | 是 | 房间唯一标识符，需与 Agent 输出的 `room` 字段完全匹配 |
| `entity_id` | string | 是 | HA climate 实体 ID，格式为 `climate.<name>` |
| `display_name` | string | 否 | 日志和调试信息中使用的显示名称 |
| `sensor_entity_id` | string | 是 | HA 温度传感器实体 ID，格式为 `sensor.<name>` |

---

## limits.yaml

定义安全约束、API 连接参数和时间限制。

**带注释示例：**

```yaml
# HA 连接配置
ha_url: "http://homeassistant.local:8123"
ha_token: "YOUR_LONG_LIVED_TOKEN"          # 建议改用环境变量 HA_TOKEN

# 温度安全范围
temp_min: 20.0          # 目标温度下限（°C），低于此值时截断
temp_max: 28.0          # 目标温度上限（°C），高于此值时截断

# 防短循环：空调两次启停之间的最小间隔（秒）
anti_cycle_seconds: 180

# API 调用冷却：两次 HA HTTP 请求之间的最小间隔（秒）
api_cooldown_seconds: 10

# 控制循环超时：传感器读取超时时间（秒）
sensor_timeout_seconds: 5
```

**字段说明：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `ha_url` | string | 是 | — | Home Assistant 的基础 URL |
| `ha_token` | string | 是* | — | HA 长期访问令牌；推荐使用环境变量 `HA_TOKEN` 替代 |
| `temp_min` | float | 是 | `20.0` | 目标温度下限（°C） |
| `temp_max` | float | 是 | `28.0` | 目标温度上限（°C） |
| `anti_cycle_seconds` | int | 否 | `180` | 防短循环最小间隔（秒） |
| `api_cooldown_seconds` | int | 否 | `10` | API 调用冷却时间（秒） |
| `sensor_timeout_seconds` | int | 否 | `5` | 传感器读取超时（秒） |

> *`ha_token` 若设置了环境变量 `HA_TOKEN`，配置文件中的值将被覆盖。

---

## pid.yaml

定义 PID 控制器参数和控制周期。字段含义见 [docs/controller.md](controller.md#pid-tuning-guide)。

**带注释示例：**

```yaml
# PID 增益参数
Kp: 1.0      # 比例增益：误差每 1°C 产生 1 个单位控制量
Ki: 0.1      # 积分增益：消除稳态误差，避免过大
Kd: 0.05     # 微分增益：抑制超调，对噪声敏感

# 控制死区：误差在此范围内不发出动作
deadband_default: 0.5    # 单位：°C；Agent 未指定 deadband 时使用此值

# 控制周期
dt: 30       # 单位：秒；每隔 dt 秒执行一次 PID 计算

# 控制量映射阈值（对应 controller.md §Control Output Mapping）
mapping_thresholds:
  strong_cool: -2.0    # u < strong_cool → 强制制冷
  cool: -0.5           # strong_cool <= u < cool → 普通制冷
  heat: 0.5            # u > heat → 停止制冷或加热
```

**字段说明：**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `Kp` | float | 是 | `1.0` | 比例增益 |
| `Ki` | float | 是 | `0.1` | 积分增益 |
| `Kd` | float | 是 | `0.05` | 微分增益 |
| `deadband_default` | float | 否 | `0.5` | Agent 未提供 deadband 时的默认死区（°C） |
| `dt` | int | 是 | `30` | 控制周期（秒） |
| `mapping_thresholds.strong_cool` | float | 否 | `-2.0` | 强制制冷阈值 |
| `mapping_thresholds.cool` | float | 否 | `-0.5` | 普通制冷阈值 |
| `mapping_thresholds.heat` | float | 否 | `0.5` | 加热/停止制冷阈值 |

---

## Validation Rules

系统启动时（`main.py`）自动校验配置，失败时打印错误并退出：

| 检查项 | 规则 | 报错信息示例 |
|--------|------|-------------|
| `rooms` 非空 | 至少定义一个房间 | `Config error: rooms list is empty` |
| `room_id` 唯一 | 不允许重复 | `Config error: duplicate room_id 'bedroom'` |
| `entity_id` 格式 | 必须以 `climate.` 开头 | `Config error: invalid entity_id 'ac_bedroom'` |
| `temp_min < temp_max` | 下限必须小于上限 | `Config error: temp_min >= temp_max` |
| `Kp`, `Ki`, `Kd` > 0 | 增益必须为正数 | `Config error: Kp must be positive` |
| `dt` >= 1 | 控制周期至少 1 秒 | `Config error: dt must be >= 1` |
| `ha_url` 格式 | 必须以 `http://` 或 `https://` 开头 | `Config error: invalid ha_url` |

---

## Full Example

以下是一套完整的双房间（卧室 + 客厅）配置示例：

**`config/rooms.yaml`**

```yaml
rooms:
  - room_id: bedroom
    entity_id: climate.bedroom_ac
    display_name: 卧室空调
    sensor_entity_id: sensor.bedroom_temperature

  - room_id: living_room
    entity_id: climate.living_room_ac
    display_name: 客厅空调
    sensor_entity_id: sensor.living_room_temperature
```

**`config/limits.yaml`**

```yaml
ha_url: "http://192.168.1.100:8123"
ha_token: ""        # 留空，使用环境变量 HA_TOKEN

temp_min: 20.0
temp_max: 28.0
anti_cycle_seconds: 180
api_cooldown_seconds: 10
sensor_timeout_seconds: 5
```

**`config/pid.yaml`**

```yaml
Kp: 1.0
Ki: 0.1
Kd: 0.05
deadband_default: 0.5
dt: 30

mapping_thresholds:
  strong_cool: -2.0
  cool: -0.5
  heat: 0.5
```
