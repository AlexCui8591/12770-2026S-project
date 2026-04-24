"""Mock PID controller for Phase 2 Round 2 / Phase 5 testing.

This exists because the real PID + 2R2C thermal model is still being written
by the control teammate. It lets the LLM supervisor loop run end-to-end
without the real PID code, by providing a plausible, observable telemetry
stream that responds to gain changes.

Design summary
--------------
The simulation is DELIBERATELY SIMPLE. It is NOT a 2R2C model, and it does
NOT claim to be physically accurate. Its only job is to produce telemetry
that looks reactive enough for Phase 1-5 LLM testing. When the real PID is
ready we swap the underlying dynamics, keeping the `get_telemetry()` /
`set_gains()` / `set_setpoint()` interface stable.

Dynamics per step(dt=60s):
    error(t)         = setpoint - indoor_temp(t)
    control_signal   = saturate(kp·error + ki·integral + kd·derivative, 0..3000 W)
    indoor_temp(t+1) = indoor_temp(t) + alpha · control_signal / SCALE + noise
                      - beta · (indoor_temp - outdoor_temp)

Where alpha reflects HVAC effectiveness and beta reflects wall heat loss.
Both are small constants tuned so that at full control the room reaches
setpoint in ~10-20 steps (i.e. 10-20 minutes of simulated time), matching
the time-scale the Phase 5 supervisor expects.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


# -- Tuning constants (not user-facing) --
_DT_SECONDS = 60.0
_CONTROL_MAX_WATTS = 3000.0
_CONTROL_MIN_WATTS = 0.0
_ALPHA_HEATING_GAIN = 1.2e-4   # °C per (W · step) — how fast HVAC moves the room
_BETA_WALL_LEAK = 0.008        # fraction of (indoor - outdoor) lost per step
_NOISE_STD = 0.05              # °C of sensor noise per step
_INTEGRAL_CLAMP = 500.0        # anti-windup
_OSCILLATION_THRESHOLD_C = 0.1 # min delta to count as a direction change
_COST_COMFORT_WEIGHT = 0.34
_COST_ENERGY_WEIGHT = 0.33
_COST_RESPONSE_WEIGHT = 0.33


@dataclass
class MockPIDController:
    """One room's worth of simulated PID + thermal state.

    Fields named in snake_case match the get_telemetry() output keys.
    Fields prefixed with underscore are internal and not exposed.
    """

    # Observable state (returned by get_telemetry)
    kp: float = 2.0
    ki: float = 0.05
    kd: float = 0.8
    setpoint: float = 22.0
    indoor_temp: float = 22.0
    tracking_error: float = 0.0
    control_signal: float = 0.0           # watts
    cumulative_energy_kwh: float = 0.0
    oscillation_count: int = 0
    cost_J: float = 0.0
    comfort_weight: float = _COST_COMFORT_WEIGHT
    energy_weight: float = _COST_ENERGY_WEIGHT
    response_weight: float = _COST_RESPONSE_WEIGHT
    # timestamp is produced on read, not stored

    # Environment (settable, not returned by pid_telemetry — lives in env tools)
    outdoor_temp: float = 10.0

    # Private controller state
    _integral: float = 0.0
    _prev_error: float = 0.0
    _last_error_sign: int = 0
    _step_count: int = 0
    _start_time: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_telemetry(self) -> dict:
        """Return the 11-field snapshot the LLM supervisor reads.

        Exactly the fields discussed in Phase 2 Round 2 planning: the 8
        fields listed in the Milestone Report plus 3 semantic anchors
        (timestamp, setpoint, indoor_temp) needed to interpret the rest.
        """
        sim_time = self._start_time + timedelta(seconds=self._step_count * _DT_SECONDS)
        return {
            "timestamp": sim_time.isoformat(timespec="seconds"),
            # Gains
            "kp": round(self.kp, 4),
            "ki": round(self.ki, 4),
            "kd": round(self.kd, 4),
            # Core state
            "setpoint": round(self.setpoint, 2),
            "indoor_temp": round(self.indoor_temp, 2),
            "tracking_error": round(self.tracking_error, 3),
            # Milestone-required metrics
            "control_signal": round(self.control_signal, 1),
            "cumulative_energy_kwh": round(self.cumulative_energy_kwh, 4),
            "oscillation_count": self.oscillation_count,
            "cost_J": round(self.cost_J, 3),
            "cost_weights": {
                "comfort": round(self.comfort_weight, 4),
                "energy": round(self.energy_weight, 4),
                "response": round(self.response_weight, 4),
            },
        }

    def set_gains(self, kp: float | None = None, ki: float | None = None,
                  kd: float | None = None) -> None:
        """Supervisor's write path for online gain tuning."""
        if kp is not None:
            self.kp = max(0.0, min(10.0, float(kp)))
        if ki is not None:
            self.ki = max(0.0, min(1.0, float(ki)))
        if kd is not None:
            self.kd = max(0.0, min(5.0, float(kd)))

    def set_setpoint(self, setpoint: float) -> None:
        self.setpoint = max(10.0, min(35.0, float(setpoint)))

    def set_outdoor_temp(self, outdoor_temp: float) -> None:
        """Allow callers (e.g. test harnesses) to reflect outdoor weather."""
        self.outdoor_temp = float(outdoor_temp)

    def step(self, noise_std: float = _NOISE_STD) -> None:
        """Advance one simulated timestep (_DT_SECONDS)."""
        # Compute PID output
        error = self.setpoint - self.indoor_temp
        self._integral += error * _DT_SECONDS
        # Anti-windup
        self._integral = max(-_INTEGRAL_CLAMP, min(_INTEGRAL_CLAMP, self._integral))
        derivative = (error - self._prev_error) / _DT_SECONDS
        raw_u = self.kp * error + self.ki * self._integral + self.kd * derivative
        # Scale to watts and saturate. 100 is an arbitrary scaling that maps
        # typical PID outputs (order of magnitude 0-30) into 0-3000 W.
        u_watts = max(_CONTROL_MIN_WATTS, min(_CONTROL_MAX_WATTS, raw_u * 100.0))
        self.control_signal = u_watts

        # Advance thermal state: heating/cooling effect + wall loss + noise
        dT_hvac = _ALPHA_HEATING_GAIN * u_watts * (1.0 if error > 0 else -1.0 if error < 0 else 0.0)
        dT_leak = _BETA_WALL_LEAK * (self.outdoor_temp - self.indoor_temp)
        noise = random.gauss(0.0, noise_std)
        self.indoor_temp = self.indoor_temp + dT_hvac + dT_leak + noise

        # Track error and oscillation
        self.tracking_error = self.setpoint - self.indoor_temp
        sign = 0
        if self.tracking_error > _OSCILLATION_THRESHOLD_C:
            sign = 1
        elif self.tracking_error < -_OSCILLATION_THRESHOLD_C:
            sign = -1
        if sign != 0 and self._last_error_sign != 0 and sign != self._last_error_sign:
            self.oscillation_count += 1
        if sign != 0:
            self._last_error_sign = sign

        # Cumulative energy (kWh)
        self.cumulative_energy_kwh += u_watts * (_DT_SECONDS / 3600.0) / 1000.0

        # Cost J per Milestone eq. (2): weighted quadratic comfort + energy + du/dt
        du = self.control_signal - (self.kp * self._prev_error * 100.0)  # rough du/dt
        self.cost_J = (
            self.comfort_weight * (self.tracking_error ** 2)
            + self.energy_weight * (u_watts / 1000.0)  # normalize to kW
            + self.response_weight * (du / 1000.0) ** 2
        )

        self._prev_error = error
        self._step_count += 1


# -- Module-level singleton used by the tool layer ---------------------------
#
# The tool function get_pid_telemetry() does NOT own a controller; it
# READS from this singleton. A test harness or the Phase 5 supervisor owns
# the writes (setpoint, gains, outdoor_temp, step()).
#
# This is intentional: it keeps the LLM-facing tool layer pure (read-only)
# while letting tests and the supervisor drive the dynamics.
# ---------------------------------------------------------------------------
DEFAULT_CONTROLLER = MockPIDController()


def update_default_controller(
    *,
    kp: float | None = None,
    ki: float | None = None,
    kd: float | None = None,
    setpoint: float | None = None,
    indoor_temp: float | None = None,
    tracking_error: float | None = None,
    control_signal: float | None = None,
    cumulative_energy_kwh: float | None = None,
    oscillation_count: int | None = None,
    cost_J: float | None = None,
    cost_weights: dict[str, float] | None = None,
    outdoor_temp: float | None = None,
    timestamp: datetime | None = None,
) -> MockPIDController:
    """Mutate the module singleton so tool reads reflect an external simulator."""
    if kp is not None:
        DEFAULT_CONTROLLER.kp = float(kp)
    if ki is not None:
        DEFAULT_CONTROLLER.ki = float(ki)
    if kd is not None:
        DEFAULT_CONTROLLER.kd = float(kd)
    if setpoint is not None:
        DEFAULT_CONTROLLER.setpoint = float(setpoint)
    if indoor_temp is not None:
        DEFAULT_CONTROLLER.indoor_temp = float(indoor_temp)
    if tracking_error is not None:
        DEFAULT_CONTROLLER.tracking_error = float(tracking_error)
    if control_signal is not None:
        DEFAULT_CONTROLLER.control_signal = float(control_signal)
    if cumulative_energy_kwh is not None:
        DEFAULT_CONTROLLER.cumulative_energy_kwh = float(cumulative_energy_kwh)
    if oscillation_count is not None:
        DEFAULT_CONTROLLER.oscillation_count = int(oscillation_count)
    if cost_J is not None:
        DEFAULT_CONTROLLER.cost_J = float(cost_J)
    if cost_weights is not None:
        if "comfort" in cost_weights:
            DEFAULT_CONTROLLER.comfort_weight = float(cost_weights["comfort"])
        if "energy" in cost_weights:
            DEFAULT_CONTROLLER.energy_weight = float(cost_weights["energy"])
        if "response" in cost_weights:
            DEFAULT_CONTROLLER.response_weight = float(cost_weights["response"])
    if outdoor_temp is not None:
        DEFAULT_CONTROLLER.outdoor_temp = float(outdoor_temp)
    if timestamp is not None:
        DEFAULT_CONTROLLER._start_time = timestamp
        DEFAULT_CONTROLLER._step_count = 0
    return DEFAULT_CONTROLLER


# -- Scenario factories (for tests) ------------------------------------------
def scenario_steady_state() -> MockPIDController:
    """S1-like: temp near setpoint, small error, nothing to do."""
    c = MockPIDController(
        kp=2.0, ki=0.05, kd=0.8,
        setpoint=22.0, indoor_temp=21.85, tracking_error=0.15,
        outdoor_temp=15.0,
        oscillation_count=1, cost_J=0.8,
    )
    return c


def scenario_weather_shock() -> MockPIDController:
    """S2-like: outdoor dropped, large error growing."""
    c = MockPIDController(
        kp=2.0, ki=0.05, kd=0.8,
        setpoint=22.0, indoor_temp=19.2, tracking_error=2.8,
        outdoor_temp=2.0, control_signal=2600.0,
        cost_J=21.5,
    )
    return c


def scenario_oscillating() -> MockPIDController:
    """Gain tuned too aggressively — many direction reversals."""
    c = MockPIDController(
        kp=6.0, ki=0.2, kd=0.3,
        setpoint=22.0, indoor_temp=22.4, tracking_error=-0.4,
        outdoor_temp=14.0, control_signal=1800.0,
        oscillation_count=17, cost_J=7.2,
    )
    return c


def scenario_saturated() -> MockPIDController:
    """Control signal rails, but we're still far from setpoint."""
    c = MockPIDController(
        kp=2.0, ki=0.05, kd=0.8,
        setpoint=24.0, indoor_temp=17.5, tracking_error=6.5,
        outdoor_temp=-5.0, control_signal=3000.0,  # saturated
        cumulative_energy_kwh=0.8, cost_J=35.0,
    )
    return c


SCENARIOS = {
    "steady_state": scenario_steady_state,
    "weather_shock": scenario_weather_shock,
    "oscillating": scenario_oscillating,
    "saturated": scenario_saturated,
}


def install_scenario(name: str) -> MockPIDController:
    """Replace DEFAULT_CONTROLLER's state with a scenario preset.

    This mutates the module-level singleton so the tool layer sees the new
    state on the next get_pid_telemetry() call. Returns the controller so
    tests can step it / inspect it directly.
    """
    global DEFAULT_CONTROLLER
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Valid: {list(SCENARIOS)}")
    DEFAULT_CONTROLLER = SCENARIOS[name]()
    return DEFAULT_CONTROLLER


if __name__ == "__main__":
    # Quick visual check: run a steady_state scenario 10 steps, print telemetry
    import json
    ctrl = install_scenario("steady_state")
    print("Initial telemetry:")
    print(json.dumps(ctrl.get_telemetry(), indent=2))
    print("\nAfter 10 steps:")
    for _ in range(10):
        ctrl.step()
    print(json.dumps(ctrl.get_telemetry(), indent=2))
