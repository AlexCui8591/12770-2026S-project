"""
Phase 1: Structured schema for LLM-parsed HVAC directives.

Design notes:
- cost_weights NOT required to sum to 1 at the schema level (LLM can't reliably
  enforce this). Normalization happens in the validation module (Phase 4).
- Temperature bounds (10-35°C) are "plausible physical" not "safe operating";
  Phase 4 validation will clamp to the stricter [16, 30] operating range.
- All enums are str-valued so Pydantic serializes cleanly to JSON.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class HVACMode(str, Enum):
    HEAT = "heat"
    COOL = "cool"
    AUTO = "auto"
    OFF = "off"


class FanSpeed(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class PresetMode(str, Enum):
    HOME = "home"
    AWAY = "away"
    SLEEP = "sleep"
    ECO = "eco"
    COMFORT = "comfort"
    BOOST = "boost"


class CostWeights(BaseModel):
    """Multi-objective cost function weights: J = w_c*comfort + w_e*energy + w_r*response."""

    energy: float = Field(..., ge=0.0, description="Weight penalizing electricity consumption")
    comfort: float = Field(..., ge=0.0, description="Weight penalizing temperature deviation")
    response: float = Field(..., ge=0.0, description="Weight penalizing aggressive actuation")

    @model_validator(mode="after")
    def _at_least_one_positive(self):
        if self.energy + self.comfort + self.response == 0:
            raise ValueError("At least one cost weight must be strictly positive")
        return self

    def normalized(self) -> "CostWeights":
        """Return a copy with weights summing to 1.0. Called by Phase 4 validator."""
        total = self.energy + self.comfort + self.response
        return CostWeights(
            energy=self.energy / total,
            comfort=self.comfort / total,
            response=self.response / total,
        )


class TemperatureRange(BaseModel):
    lower: float = Field(..., ge=10.0, le=35.0)
    upper: float = Field(..., ge=10.0, le=35.0)

    @model_validator(mode="after")
    def _lower_leq_upper(self):
        if self.lower > self.upper:
            raise ValueError(f"lower ({self.lower}) must be <= upper ({self.upper})")
        return self


class StructuredDirective(BaseModel):
    """
    Structured representation of a user's natural-language HVAC directive.
    This is the output of Phase 1 (intent parsing) and the input to the
    control layer's setpoint / weight configuration.
    """

    target_temperature: float = Field(
        ..., ge=10.0, le=35.0, description="Primary target temperature in Celsius"
    )
    temperature_range: Optional[TemperatureRange] = Field(
        None, description="Optional acceptable range; if None, use target ± deadband"
    )
    hvac_mode: HVACMode = Field(..., description="HVAC operating mode")
    fan_speed: FanSpeed = Field(FanSpeed.AUTO, description="Fan speed preference")
    deadband: float = Field(
        0.5, ge=0.1, le=3.0, description="Tolerance around target temperature (°C)"
    )
    preset_mode: PresetMode = Field(PresetMode.HOME, description="High-level preset")
    cost_weights: CostWeights = Field(..., description="Multi-objective priorities")
    rationale: str = Field(
        ..., min_length=5, description="LLM's explanation of its decision"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "target_temperature": 22.0,
                "temperature_range": {"lower": 21.0, "upper": 23.0},
                "hvac_mode": "cool",
                "fan_speed": "low",
                "deadband": 1.0,
                "preset_mode": "eco",
                "cost_weights": {"energy": 0.6, "comfort": 0.3, "response": 0.1},
                "rationale": "User prioritized energy savings over precise tracking.",
            }
        }
    }
