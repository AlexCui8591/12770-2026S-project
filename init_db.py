"""初始化 SQLite 数据库：从现有 JSON/YAML 文件迁移数据到 data/smart_home.db。

可重复执行（先清空表再插入）。

使用方式：
    python init_db.py
"""

import json
import sqlite3
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
DB_PATH = DATA_DIR / "smart_home.db"

# ============================================================
# 建表 DDL
# ============================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS user_habits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT    NOT NULL,
    time_of_day     TEXT    NOT NULL CHECK(time_of_day IN ('morning','afternoon','night')),
    preferred_temp  REAL    NOT NULL DEFAULT 25.0,
    preferred_sleep_temp REAL DEFAULT NULL,
    bedtime         TEXT    DEFAULT NULL,
    wake_time       TEXT    DEFAULT NULL,
    fan_preference  TEXT    NOT NULL DEFAULT 'auto',
    UNIQUE(user_id, time_of_day)
);

CREATE TABLE IF NOT EXISTS user_schedule (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    day_type    TEXT    NOT NULL CHECK(day_type IN ('weekday','weekend')),
    event_type  TEXT    NOT NULL,
    event_time  TEXT    NOT NULL,
    note        TEXT    DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS rooms (
    room_id          TEXT PRIMARY KEY,
    entity_id        TEXT NOT NULL,
    display_name     TEXT NOT NULL,
    sensor_entity_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS energy_price (
    hour          INTEGER NOT NULL,
    tier          TEXT    NOT NULL CHECK(tier IN ('peak','off_peak')),
    price_per_kwh REAL    NOT NULL,
    PRIMARY KEY (hour)
);

CREATE TABLE IF NOT EXISTS limits (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# 数据迁移函数
# ============================================================

def _migrate_user_habits(cur: sqlite3.Cursor) -> int:
    """迁移 data/user_habits.json → user_habits 表。"""
    cur.execute("DELETE FROM user_habits")
    data = _load_json(DATA_DIR / "user_habits.json")
    count = 0
    for user_id, periods in data.items():
        for time_of_day, prefs in periods.items():
            cur.execute(
                """INSERT INTO user_habits
                   (user_id, time_of_day, preferred_temp, preferred_sleep_temp,
                    bedtime, wake_time, fan_preference)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    time_of_day,
                    prefs.get("preferred_temp", 25.0),
                    prefs.get("preferred_sleep_temp"),
                    prefs.get("bedtime"),
                    prefs.get("wake_time"),
                    prefs.get("fan_preference", "auto"),
                ),
            )
            count += 1
    return count


def _migrate_user_schedule(cur: sqlite3.Cursor) -> int:
    """迁移 data/user_schedule.json → user_schedule 表。"""
    cur.execute("DELETE FROM user_schedule")
    data = _load_json(DATA_DIR / "user_schedule.json")
    count = 0
    for user_id, day_types in data.items():
        for day_type, events in day_types.items():
            for event in events:
                cur.execute(
                    """INSERT INTO user_schedule
                       (user_id, day_type, event_type, event_time, note)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        day_type,
                        event["type"],
                        event["at"],
                        event.get("note"),
                    ),
                )
                count += 1
    return count


def _migrate_rooms(cur: sqlite3.Cursor) -> int:
    """迁移 config/rooms.yaml → rooms 表。"""
    cur.execute("DELETE FROM rooms")
    data = _load_yaml(CONFIG_DIR / "rooms.yaml")
    count = 0
    for room in data.get("rooms", []):
        cur.execute(
            """INSERT INTO rooms (room_id, entity_id, display_name, sensor_entity_id)
               VALUES (?, ?, ?, ?)""",
            (room["room_id"], room["entity_id"], room["display_name"], room["sensor_entity_id"]),
        )
        count += 1
    return count


def _migrate_energy_price(cur: sqlite3.Cursor) -> int:
    """迁移 config/energy.yaml → energy_price 表（逐小时展开）。"""
    cur.execute("DELETE FROM energy_price")
    data = _load_yaml(CONFIG_DIR / "energy.yaml")
    count = 0
    for tier_key in ("peak", "off_peak"):
        tier_data = data.get(tier_key, {})
        tier_name = tier_data.get("tier", tier_key)
        price = tier_data.get("price_per_kwh", 0.0)
        for hour in tier_data.get("hours", []):
            cur.execute(
                "INSERT INTO energy_price (hour, tier, price_per_kwh) VALUES (?, ?, ?)",
                (hour, tier_name, price),
            )
            count += 1
    return count


def _migrate_limits(cur: sqlite3.Cursor) -> int:
    """迁移 config/limits.yaml → limits 表。"""
    cur.execute("DELETE FROM limits")
    data = _load_yaml(CONFIG_DIR / "limits.yaml")
    count = 0
    for key, value in data.items():
        cur.execute(
            "INSERT INTO limits (key, value) VALUES (?, ?)",
            (key, str(value)),
        )
        count += 1
    return count


# ============================================================
# 主入口
# ============================================================

def init_database(db_path: Path | None = None) -> Path:
    """创建数据库并迁移所有数据。返回数据库文件路径。"""
    if db_path is None:
        db_path = DB_PATH

    print(f"数据库路径: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 建表
    cur.executescript(SCHEMA_SQL)
    print("表结构已创建")

    # 迁移数据
    n = _migrate_user_habits(cur)
    print(f"  user_habits: {n} 条记录")

    n = _migrate_user_schedule(cur)
    print(f"  user_schedule: {n} 条记录")

    n = _migrate_rooms(cur)
    print(f"  rooms: {n} 条记录")

    n = _migrate_energy_price(cur)
    print(f"  energy_price: {n} 条记录")

    n = _migrate_limits(cur)
    print(f"  limits: {n} 条记录")

    conn.commit()
    conn.close()

    print(f"\n✅ 数据库初始化完成: {db_path}")
    return db_path


if __name__ == "__main__":
    init_database()
