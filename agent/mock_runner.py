"""
Mock Simulation Runner
======================

Runs the full LLM supervision loop with a simplified thermal model,
without requiring the PID team's code. This lets you:

  1. Validate that the LLM makes reasonable decisions over time
  2. Test C2 vs C3 mode differences
  3. Test failure injection (S6)
  4. Generate decision logs for analysis
  5. Iterate on prompts before the real PID code is ready

The thermal model here is intentionally simple (first-order + noise).
It does NOT replace the real 2R2C model — it only exists so the LLM
has something to react to.

Usage:
    python -m agent.mock_runner --scenario weather_disturbance --steps 60 --mode proactive
    python -m agent.mock_runner --scenario all --steps 30
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agent.interfaces import (
    PIDTelemetry, EnvironmentContext, UserIntent, CostWeights, SupervisorDecision,
)
from agent.intent_parser import parse_user_intent_offline
from agent.supervisor import supervisor_step, SupervisorConfig, DecisionLogger
from agent.mock_telemetry import MockScenarios

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Simplified Thermal Simulation (NOT the real 2R2C model)
# ──────────────────────────────────────────────────────────────

@dataclass
class SimpleThermalState:
    """Minimal state for a fake thermal simulation."""
    indoor_temp: float = 20.0
    setpoint: float = 22.0
    outdoor_temp: float = 10.0
    kp: float = 2.0
    ki: float = 0.05
    kd: float = 0.8
    w_energy: float = 0.33
    w_comfort: float = 0.34
    w_response: float = 0.33
    cumulative_energy_kwh: float = 0.0
    oscillation_count: int = 0
    control_signal: float = 0.0
    max_overshoot: float = 0.0
    cost_J: float = 0.0
    _prev_error: float = 0.0
    _integral: float = 0.0
    _last_temp: float = 20.0
    _dt: float = 60.0  # seconds

    def step(self, noise_std: float = 0.2) -> None:
        """Advance one timestep (60s) with a simplified PID + first-order thermal."""
        error = self.setpoint - self.indoor_temp

        # Simple PID
        self._integral += error * self._dt
        self._integral = max(-500, min(500, self._integral))  # anti-windup
        derivative = (error - self._prev_error) / self._dt
        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        self.control_signal = max(0, min(3000, u * 100))  # scale to watts

        # Simplified thermal dynamics
        # dT/dt ≈ (Q_hvac - heat_loss) / C_air
        heat_loss = (self.indoor_temp - self.outdoor_temp) / 0.004  # W, via R_wall
        net_power = self.control_signal - heat_loss * 0.001  # simplified scaling
        dT = net_power * self._dt / 250000  # C_air = 250 kJ/°C
        noise = random.gauss(0, noise_std)
        self.indoor_temp += dT + noise

        # Track metrics
        self.cumulative_energy_kwh += self.control_signal * self._dt / 3_600_000
        overshoot = abs(self.indoor_temp - self.setpoint)
        self.max_overshoot = max(self.max_overshoot, overshoot)

        # Count oscillations (setpoint crossings)
        if (self.indoor_temp - self.setpoint) * (self._last_temp - self.setpoint) < 0:
            self.oscillation_count += 1

        # Update cost (simplified)
        self.cost_J = (
            self.w_comfort * error ** 2
            + self.w_energy * self.control_signal / 3000
            + self.w_response * derivative ** 2 * 1000
        )

        self._prev_error = error
        self._last_temp = self.indoor_temp

    def to_telemetry(self, timestamp: str) -> PIDTelemetry:
        return PIDTelemetry(
            timestamp=timestamp,
            kp=self.kp, ki=self.ki, kd=self.kd,
            indoor_temp=round(self.indoor_temp, 2),
            setpoint=self.setpoint,
            tracking_error=round(self.setpoint - self.indoor_temp, 2),
            max_overshoot=round(self.max_overshoot, 2),
            control_signal=round(self.control_signal, 1),
            cumulative_energy_kwh=round(self.cumulative_energy_kwh, 3),
            oscillation_count=self.oscillation_count,
            cost_J=round(self.cost_J, 2),
            w_energy=self.w_energy,
            w_comfort=self.w_comfort,
            w_response=self.w_response,
        )

    def apply_decision(self, decision: SupervisorDecision) -> None:
        """Apply a supervisor decision to the thermal state."""
        if decision.action == "adjust_weights" and decision.cost_weights:
            self.w_energy = decision.cost_weights.energy
            self.w_comfort = decision.cost_weights.comfort
            self.w_response = decision.cost_weights.response

        if decision.action == "shift_setpoint":
            self.setpoint += decision.setpoint_correction

        if decision.action == "override_gains":
            if decision.pid_override_kp is not None:
                self.kp = decision.pid_override_kp
            if decision.pid_override_ki is not None:
                self.ki = decision.pid_override_ki
            if decision.pid_override_kd is not None:
                self.kd = decision.pid_override_kd

        # Feedforward always applied
        self.control_signal += decision.feedforward_offset


# ──────────────────────────────────────────────────────────────
#  Scenario Definitions
# ──────────────────────────────────────────────────────────────

def make_scenario_env(name: str, step: int, total_steps: int) -> EnvironmentContext:
    """Generate time-varying environment context for each scenario."""
    t_frac = step / max(total_steps, 1)

    if name == "steady_state":
        return EnvironmentContext(outdoor_temp=15.0, electricity_price=0.08, tariff_tier="off_peak")

    elif name == "weather_disturbance":
        # Outdoor temp drops 10°C at step 30
        outdoor = 12.0 if step < 30 else 2.0
        forecast = "Temperature dropping to 2°C within the hour" if 20 <= step < 30 else None
        return EnvironmentContext(
            outdoor_temp=outdoor, weather_condition="snow" if step >= 30 else "clear",
            weather_forecast_3h=forecast,
            electricity_price=0.10, tariff_tier="mid_peak",
        )

    elif name == "peak_tariff":
        # Price spikes 3x during middle third
        if 0.33 < t_frac < 0.66:
            price, tier = 0.30, "peak"
        else:
            price, tier = 0.08, "off_peak"
        return EnvironmentContext(outdoor_temp=10.0, electricity_price=price, tariff_tier=tier)

    elif name == "guests_arriving":
        events = "Guests arriving at 18:00" if t_frac < 0.5 else None
        return EnvironmentContext(
            outdoor_temp=8.0, electricity_price=0.12, tariff_tier="mid_peak",
            is_occupied=t_frac > 0.5, scheduled_events=events,
        )

    elif name == "multi_disturbance":
        outdoor = 10.0 - 15.0 * t_frac  # gradually drops to -5
        price = 0.08 + 0.22 * (0.5 + 0.5 * __import__("math").sin(t_frac * 6.28))
        tier = "peak" if price > 0.20 else "mid_peak" if price > 0.12 else "off_peak"
        return EnvironmentContext(
            outdoor_temp=outdoor, weather_condition="snow",
            weather_forecast_3h=f"Continued cold, outdoor temp ~{outdoor - 2:.0f}°C",
            electricity_price=round(price, 3), tariff_tier=tier,
        )

    else:
        return EnvironmentContext(outdoor_temp=12.0, electricity_price=0.10, tariff_tier="mid_peak")


SCENARIO_COMMANDS = {
    "steady_state": "Set the bedroom to 22°C.",
    "weather_disturbance": "Keep the room comfortable.",
    "peak_tariff": "Keep it warm but minimize electricity costs.",
    "guests_arriving": "Guests coming at 6 PM, make it comfortable.",
    "multi_disturbance": "Balance comfort and cost for 24 hours.",
}


# ──────────────────────────────────────────────────────────────
#  Main Runner
# ──────────────────────────────────────────────────────────────

def run_simulation(
    scenario: str,
    total_steps: int = 60,
    config: Optional[SupervisorConfig] = None,
    verbose: bool = True,
) -> dict:
    """Run a mock simulation with LLM supervision.

    Returns dict with full history for analysis.
    """
    if config is None:
        config = SupervisorConfig()

    # Parse intent (offline, no Ollama needed)
    command = SCENARIO_COMMANDS.get(scenario, "Keep the room at 22°C.")
    intent = parse_user_intent_offline(command)

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  Scenario: {scenario}  |  Mode: {config.mode}  |  Steps: {total_steps}")
        print(f"  Command: \"{command}\"")
        print(f"  Initial weights: {intent.initial_cost_weights.to_dict()}")
        print(f"{'═'*65}")

    # Init state
    state = SimpleThermalState(
        indoor_temp=20.0,
        setpoint=intent.target_temperature,
    )
    if intent.initial_cost_weights:
        state.w_energy = intent.initial_cost_weights.energy
        state.w_comfort = intent.initial_cost_weights.comfort
        state.w_response = intent.initial_cost_weights.response

    # Set up logging
    log_db = tempfile.mktemp(suffix=".db")
    decision_logger = DecisionLogger(log_db)

    # Run
    history = {
        "temps": [], "setpoints": [], "errors": [], "energy": [],
        "decisions": [], "cost_J": [], "control_signal": [],
    }

    random.seed(config.seed or 42)

    for step in range(total_steps):
        timestamp = f"2026-04-05T{14 + step // 60:02d}:{step % 60:02d}:00"

        # Get environment for this step
        env = make_scenario_env(scenario, step, total_steps)

        # Apply outdoor temp to state
        state.outdoor_temp = env.outdoor_temp

        # Advance thermal simulation
        state.step(noise_std=0.15)

        # Record history
        history["temps"].append(round(state.indoor_temp, 2))
        history["setpoints"].append(state.setpoint)
        history["errors"].append(round(state.setpoint - state.indoor_temp, 2))
        history["energy"].append(round(state.cumulative_energy_kwh, 3))
        history["cost_J"].append(round(state.cost_J, 2))
        history["control_signal"].append(round(state.control_signal, 1))

        # Supervision cycle
        if step > 0 and step % config.supervision_interval_steps == 0:
            telemetry = state.to_telemetry(timestamp)

            decision = supervisor_step(
                user_intent=intent,
                telemetry=telemetry,
                env_context=env,
                config=config,
                sim_step=step,
                decision_logger=decision_logger,
            )

            # Apply decision
            state.apply_decision(decision)

            history["decisions"].append({
                "step": step,
                "action": decision.action,
                "weights": decision.cost_weights.to_dict() if decision.cost_weights else {},
                "setpoint_correction": decision.setpoint_correction,
                "feedforward_offset": decision.feedforward_offset,
                "reason": decision.reason,
            })

            if verbose:
                print(f"  [Step {step:3d}] T={state.indoor_temp:5.1f}°C  "
                      f"err={state.setpoint - state.indoor_temp:+5.2f}  "
                      f"→ {decision.action:16s}  "
                      f"reason: {decision.reason[:50]}")

    # Summary
    avg_error = sum(abs(e) for e in history["errors"]) / len(history["errors"])
    final_energy = history["energy"][-1]
    n_decisions = len(history["decisions"])
    n_adjustments = sum(1 for d in history["decisions"] if d["action"] != "hold")

    summary = {
        "scenario": scenario,
        "mode": config.mode,
        "total_steps": total_steps,
        "avg_absolute_error": round(avg_error, 3),
        "final_energy_kwh": final_energy,
        "total_supervision_cycles": n_decisions,
        "active_adjustments": n_adjustments,
        "final_temp": history["temps"][-1],
        "final_setpoint": history["setpoints"][-1],
    }

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Summary:")
        print(f"    Avg |error|:        {avg_error:.3f}°C")
        print(f"    Final energy:       {final_energy:.3f} kWh")
        print(f"    Supervision cycles: {n_decisions}")
        print(f"    Active adjustments: {n_adjustments}")
        print(f"    Final temp:         {history['temps'][-1]:.1f}°C → setpoint {history['setpoints'][-1]:.1f}°C")

    decision_logger.close()

    return {"summary": summary, "history": history, "log_db": log_db}


def main():
    parser = argparse.ArgumentParser(description="Run mock LLM-supervised HVAC simulation")
    parser.add_argument("--scenario", default="weather_disturbance",
                        choices=list(SCENARIO_COMMANDS.keys()) + ["all"])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--mode", choices=["reactive", "proactive"], default="proactive")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interval", type=int, default=5, help="Supervision interval in steps")
    parser.add_argument("--fail-inject", action="store_true", help="Enable S6 failure injection")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = SupervisorConfig(
        mode=args.mode,
        seed=args.seed,
        supervision_interval_steps=args.interval,
        enable_failure_injection=args.fail_inject,
    )

    if args.scenario == "all":
        results = {}
        for scenario in SCENARIO_COMMANDS:
            result = run_simulation(scenario, args.steps, config)
            results[scenario] = result["summary"]

        print(f"\n\n{'═'*65}")
        print("  OVERALL RESULTS")
        print(f"{'═'*65}")
        for name, s in results.items():
            print(f"  {name:25s}  err={s['avg_absolute_error']:.3f}°C  "
                  f"energy={s['final_energy_kwh']:.3f}kWh  "
                  f"adj={s['active_adjustments']}/{s['total_supervision_cycles']}")
    else:
        run_simulation(args.scenario, args.steps, config)


if __name__ == "__main__":
    main()
