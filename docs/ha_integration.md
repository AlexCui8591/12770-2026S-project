# Home Assistant Integration

本文档描述 `ha/` 模块如何与 Home Assistant 通信，包括认证方式、entity 命名约定、支持的 climate 服务、以及测试方法。

---

## Prerequisites

| 要求 | 说明 |
|------|------|
| Home Assistant 版本 | 2023.1 或更高（climate 实体 API 稳定） |
| 网络可达性 | 运行本项目的主机需能访问 HA 的 HTTP 端口（默认 8123） |
| 长期访问令牌 | 用于 API 认证，不会过期 |

**获取长期访问令牌：**

1. 登录 Home Assistant Web UI
2. 点击左下角头像 → **Profile**
3. 滚动到页面底部 → **Long-Lived Access Tokens** → **Create Token**
4. 复制 Token，存入环境变量或 `config/limits.yaml` 的 `ha_token` 字段

> 令牌只显示一次，请立即保存。

---

## Entity Naming Convention

HA climate 实体 ID 由 `config/rooms.yaml` 中的 `entity_id` 字段直接指定，不做自动推导。

示例（`rooms.yaml`）：

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

`ha/client.py` 通过 `room_id` 查找对应的 `entity_id`，若找不到则抛出 `RoomNotFoundError`。

---

## Supported Services

本系统仅使用以下五个 HA climate 服务：

| 服务 | 触发条件 | 关键参数 |
|------|----------|---------|
| `climate.set_temperature` | 目标温度变化 | `entity_id`, `temperature` |
| `climate.set_hvac_mode` | HVAC 模式变化（cool/heat/off） | `entity_id`, `hvac_mode` |
| `climate.set_fan_mode` | 风速模式变化 | `entity_id`, `fan_mode` |
| `climate.turn_on` | 从 off 状态启动 | `entity_id` |
| `climate.turn_off` | 关闭设备 | `entity_id` |

---

## API Call Examples

HA REST API 的调用格式（`ha/services.py` 封装了以下调用）：

**Base URL:** `http://<HA_HOST>:8123/api/services/<domain>/<service>`

**Headers:**

```
Authorization: Bearer <YOUR_TOKEN>
Content-Type: application/json
```

### set_temperature

```json
POST /api/services/climate/set_temperature
{
  "entity_id": "climate.bedroom_ac",
  "temperature": 24
}
```

### set_hvac_mode

```json
POST /api/services/climate/set_hvac_mode
{
  "entity_id": "climate.bedroom_ac",
  "hvac_mode": "cool"
}
```

### set_fan_mode

```json
POST /api/services/climate/set_fan_mode
{
  "entity_id": "climate.bedroom_ac",
  "fan_mode": "low"
}
```

### turn_on / turn_off

```json
POST /api/services/climate/turn_on
{
  "entity_id": "climate.bedroom_ac"
}
```

**curl 等效命令（用于手动测试）：**

```bash
curl -X POST http://homeassistant.local:8123/api/services/climate/set_temperature \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "climate.bedroom_ac", "temperature": 24}'
```

---

## Rate Limiting & Cooldown

`ha/client.py` 在每次 API 调用后记录时间戳，若距上次调用未超过 `api_cooldown_seconds`（来自 `config/limits.yaml`，推荐 `10` 秒），则跳过本次调用并记录 DEBUG 日志。

```python
if time.time() - self.last_call_time < self.cooldown:
    logger.debug("API call skipped: cooldown not elapsed")
    return
```

这与 PID 的防短循环（`anti_cycle_seconds`）是两层独立保护：
- `api_cooldown_seconds`：限制 HTTP 请求频率
- `anti_cycle_seconds`：限制空调启停频率（更长，通常 180 秒）

---

## Error Handling

| HTTP 状态码 | 含义 | 处理方式 |
|-------------|------|----------|
| `200` / `201` | 成功 | 记录 INFO 日志，继续 |
| `401` | Token 无效或过期 | 抛出 `AuthError`，停止控制循环 |
| `404` | entity_id 不存在 | 抛出 `EntityNotFoundError`，检查 `rooms.yaml` |
| `400` | 参数错误（如非法 hvac_mode） | 记录 ERROR，跳过本次动作 |
| `5xx` | HA 服务端错误 | 重试 1 次（间隔 5 秒），失败后记录 ERROR |
| 连接超时 | HA 不可达 | 记录 ERROR，等待下一个控制周期 |

---

## Testing Without Hardware

在没有真实空调的情况下，可以用以下方式测试：

### 方法 1：HA 虚拟 Climate 实体

在 HA 的 `configuration.yaml` 中添加：

```yaml
climate:
  - platform: generic_thermostat
    name: bedroom_ac
    heater: switch.bedroom_heater_mock
    target_sensor: sensor.bedroom_temperature
    min_temp: 16
    max_temp: 30
```

重启 HA 后，`climate.bedroom_ac` 会出现在实体列表中，可接受所有 climate 服务调用。

### 方法 2：HA Developer Tools

1. 打开 HA Web UI → **Developer Tools** → **Services**
2. 选择 `climate.set_temperature`，填写 entity_id 和 temperature
3. 点击 **Call Service** 验证 API 可达性

### 方法 3：Mock HA Client

在单元测试中，用 `ha/client.py` 的 Mock 替代真实调用：

```python
from unittest.mock import MagicMock
ha_client = MagicMock()
ha_client.set_temperature.return_value = {"result": "ok"}
```
