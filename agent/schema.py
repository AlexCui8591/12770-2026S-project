"""Schema definitions and validation for agent input and output."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

TEMP_MIN = 20.0
TEMP_MAX = 28.0

VALID_HVAC_MODES = {"cool", "heat", "off", "auto"}
VALID_PRESET_MODES = {"eco", "sleep", "comfort", "none"}
VALID_FAN_MODES = {"low", "medium", "high", "auto"}

DEADBAND_MIN = 0.0
DEADBAND_MAX = 2.0
DEADBAND_DEFAULT = 0.5


class SchemaError(Exception):
    """Raised when required fields are missing or cannot be repaired."""


class ParseError(Exception):
    """Raised when the LLM output is not valid JSON."""


@dataclass
class CurrentContext:
    room: str
    current_temperature: float
    current_hvac_mode: str
    current_fan_mode: str
    occupied: bool
    window_open: bool
    time: str


@dataclass
class AgentInput:
    user_command: str
    current_context: CurrentContext

    @classmethod
    def from_dict(cls, data: dict) -> AgentInput:
        if "user_command" not in data:
            raise SchemaError("Missing required field: user_command")
        ctx_data = data.get("current_context")
        if ctx_data is None:
            raise SchemaError("Missing required field: current_context")

        required_ctx_fields = [
            "room",
            "current_temperature",
            "current_hvac_mode",
            "current_fan_mode",
            "occupied",
            "window_open",
            "time",
        ]
        for field_name in required_ctx_fields:
            if field_name not in ctx_data:
                raise SchemaError(f"Missing required context field: {field_name}")

        return cls(
            user_command=data["user_command"],
            current_context=CurrentContext(**ctx_data),
        )


@dataclass
class AgentOutput:
    room: str
    target_temperature: float
    hvac_mode: str
    preset_mode: str = "none"
    fan_mode: str = "auto"
    deadband: float = DEADBAND_DEFAULT
    valid_until: Optional[str] = None
    solar_radiation: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "room": self.room,
            "target_temperature": self.target_temperature,
            "hvac_mode": self.hvac_mode,
            "preset_mode": self.preset_mode,
            "fan_mode": self.fan_mode,
            "deadband": self.deadband,
            "valid_until": self.valid_until,
            "solar_radiation": self.solar_radiation,
            "reason": self.reason,
        }


def validate_and_fix(raw: dict, fallback_context: CurrentContext) -> AgentOutput:
    reasons = []

    room = raw.get("room") or fallback_context.room
    if "room" not in raw:
        reasons.append("room was missing, so the context room was used")

    temp = raw.get("target_temperature")
    if temp is None:
        raise SchemaError("Missing required field: target_temperature")
    temp = float(temp)
    if temp < TEMP_MIN:
        reasons.append(
            f"target_temperature {temp}C was below the lower bound and was clamped to {TEMP_MIN}C"
        )
        temp = TEMP_MIN
    elif temp > TEMP_MAX:
        reasons.append(
            f"target_temperature {temp}C exceeded the upper bound and was clamped to {TEMP_MAX}C"
        )
        temp = TEMP_MAX

    hvac_mode = raw.get("hvac_mode", "")
    if hvac_mode not in VALID_HVAC_MODES:
        fallback = fallback_context.current_hvac_mode
        reasons.append(f"hvac_mode '{hvac_mode}' was invalid and fell back to '{fallback}'")
        hvac_mode = fallback

    preset_mode = raw.get("preset_mode", "none")
    if preset_mode not in VALID_PRESET_MODES:
        reasons.append(f"preset_mode '{preset_mode}' was invalid and fell back to 'none'")
        preset_mode = "none"

    fan_mode = raw.get("fan_mode", "auto")
    if fan_mode not in VALID_FAN_MODES:
        reasons.append(f"fan_mode '{fan_mode}' was invalid and fell back to 'auto'")
        fan_mode = "auto"

    deadband = raw.get("deadband", DEADBAND_DEFAULT)
    try:
        deadband = float(deadband)
    except (TypeError, ValueError):
        deadband = DEADBAND_DEFAULT
    deadband = max(DEADBAND_MIN, min(DEADBAND_MAX, deadband))

    valid_until = raw.get("valid_until")

    solar_radiation = raw.get("solar_radiation")
    if not isinstance(solar_radiation, dict):
        solar_radiation = {
            "data_status": "unavailable",
            "note": "solar_radiation was missing, so an unavailable placeholder was used",
        }
        reasons.append("solar_radiation was missing, so an unavailable placeholder was used")

    original_reason = raw.get("reason", "")
    if reasons:
        fix_note = "; ".join(reasons)
        logger.warning("Agent output required validation fixes: %s", fix_note)
        if original_reason:
            original_reason = f"{original_reason} (Validation adjusted: {fix_note})"
        else:
            original_reason = f"Validation adjusted: {fix_note}"

    return AgentOutput(
        room=room,
        target_temperature=temp,
        hvac_mode=hvac_mode,
        preset_mode=preset_mode,
        fan_mode=fan_mode,
        deadband=deadband,
        valid_until=valid_until,
        solar_radiation=solar_radiation,
        reason=original_reason,
    )
