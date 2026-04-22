"""Schema definitions and validation for agent input and output."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Operating temperature bounds. The validator clamps any LLM output to this
# range. Widened from the previous 20-28 to match the Milestone's safe
# operating range so that eco/away setpoints (e.g. 18°C) are reachable.
TEMP_MIN = 16.0
TEMP_MAX = 30.0

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
class CostWeights:
    """Multi-objective cost function weights for the downstream PID optimizer.

    J = w_comfort * integral(e^2) + w_energy * integral(P) + w_response * integral((du/dt)^2)

    Stored after normalization so that the three weights sum to 1. The raw
    LLM output does NOT need to be pre-normalized; validate_and_fix handles it.
    """

    energy: float
    comfort: float
    response: float

    def to_dict(self) -> dict:
        return {"energy": self.energy, "comfort": self.comfort, "response": self.response}

    @classmethod
    def equal(cls) -> CostWeights:
        return cls(energy=1 / 3, comfort=1 / 3, response=1 / 3)


@dataclass
class AgentOutput:
    room: str
    target_temperature: float
    hvac_mode: str
    cost_weights: CostWeights
    preset_mode: str = "none"
    fan_mode: str = "auto"
    deadband: float = DEADBAND_DEFAULT
    valid_until: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "room": self.room,
            "target_temperature": self.target_temperature,
            "hvac_mode": self.hvac_mode,
            "cost_weights": self.cost_weights.to_dict(),
            "preset_mode": self.preset_mode,
            "fan_mode": self.fan_mode,
            "deadband": self.deadband,
            "valid_until": self.valid_until,
            "reason": self.reason,
        }


def _normalize_cost_weights(raw: Any) -> tuple[CostWeights, Optional[str]]:
    """Parse, sanitize, and normalize cost_weights.

    Returns (weights, reason_note_or_None). The reason_note is appended to
    the AgentOutput.reason so that downstream analysis can detect when the
    LLM failed to produce valid weights (important for H2 evaluation).
    """
    if not isinstance(raw, dict):
        return CostWeights.equal(), "cost_weights was missing or malformed; equal weights applied"

    try:
        energy = max(0.0, float(raw.get("energy", 0)))
        comfort = max(0.0, float(raw.get("comfort", 0)))
        response = max(0.0, float(raw.get("response", 0)))
    except (TypeError, ValueError):
        return CostWeights.equal(), "cost_weights contained non-numeric values; equal weights applied"

    total = energy + comfort + response
    if total == 0:
        return CostWeights.equal(), "all cost_weights were zero; equal weights applied"

    # Normalize so the three weights sum to 1 (decision: normalization happens
    # in the validator, not in the LLM prompt).
    return CostWeights(energy / total, comfort / total, response / total), None


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

    # preset_mode: treat null / missing as "none" silently (it's a valid
    # signal meaning "no special preset applies"). Only log a correction
    # when the LLM actively outputs a string that isn't in the allowed set.
    preset_mode_raw = raw.get("preset_mode")
    if preset_mode_raw is None:
        preset_mode = "none"
    elif preset_mode_raw not in VALID_PRESET_MODES:
        reasons.append(f"preset_mode '{preset_mode_raw}' was invalid and fell back to 'none'")
        preset_mode = "none"
    else:
        preset_mode = preset_mode_raw

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

    cost_weights, cw_note = _normalize_cost_weights(raw.get("cost_weights"))
    if cw_note is not None:
        reasons.append(cw_note)

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
        cost_weights=cost_weights,
        preset_mode=preset_mode,
        fan_mode=fan_mode,
        deadband=deadband,
        valid_until=valid_until,
        reason=original_reason,
    )
