"""Demo Version A — scenario definitions and simulation runner.

A "scenario" is a dict that describes how the environment evolves over time:
    - outdoor_temp(t_min): outdoor temperature at minute t
    - initial_indoor_temp: where the room starts
    - setpoint: the target temperature (for version A, constant)
    - duration_min: how long to simulate

A "run" drives a MockPIDController through the scenario, stepping once per
minute, collecting per-minute samples that metrics.compute_metrics() consumes.

This file is intentionally dumb. It doesn't know about LLM, conditions,
supervisors, or anything at Layer 2. It just:
    - configures the controller from a scenario
    - calls on_step hook each minute (for condition-specific logic)
    - returns the sample list
"""

from __future__ import annotations

from typing import Callable, Optional

from agent.mock_pid import MockPIDController


# -- Scenario definitions --

S1 = {
    "name": "S1_steady_state",
    "description": "Cold-start steady tracking. Outdoor 15°C, target 22°C.",
    "outdoor_temp_fn": lambda t_min: 15.0,   # constant outdoor temperature
    "initial_indoor_temp": 18.0,              # cold start, 4°C below setpoint
    "setpoint": 22.0,
    "duration_min": 120,
}


def make_controller_for(scenario: dict,
                        kp: float = 2.0,
                        ki: float = 0.01,
                        kd: float = 0.8) -> MockPIDController:
    """Build a fresh MockPIDController configured for this scenario.

    Default gains are tuned for Version A:
      - Kp=2.0, Ki=0.01, Kd=0.8
    This gives C0 a sensible baseline (MAD~0.9°C, RT~60min, OC<10) that
    leaves visible room for LLM conditions (C1-C3) to improve upon.
    Earlier default Ki=0.05 caused integral windup and OC>80, which made
    C0-C3 indistinguishable.
    """
    c = MockPIDController(
        kp=kp, ki=ki, kd=kd,
        setpoint=scenario["setpoint"],
        indoor_temp=scenario["initial_indoor_temp"],
        outdoor_temp=scenario["outdoor_temp_fn"](0),
    )
    return c


def run_simulation(
    controller: MockPIDController,
    scenario: dict,
    on_step: Optional[Callable[[int, MockPIDController], None]] = None,
) -> list[dict]:
    """Drive `controller` through `scenario`, return per-minute samples.

    Parameters
    ----------
    controller : MockPIDController
        Pre-configured controller (see make_controller_for).
    scenario : dict
        Must contain 'outdoor_temp_fn', 'setpoint', 'duration_min'.
    on_step : callable, optional
        Hook called at the start of each minute, BEFORE stepping. Signature:
            on_step(t_min: int, controller: MockPIDController) -> None
        Used by C2/C3 to let the LLM supervisor intervene periodically.
        For C0/C1, pass None.

    Returns
    -------
    list of dicts
        Per-minute samples, each with {t_min, indoor_temp, setpoint,
        control_signal}. Feed this directly to metrics.compute_metrics().
    """
    samples = []
    duration = scenario["duration_min"]
    outdoor_fn = scenario["outdoor_temp_fn"]

    for t in range(duration):
        # Update outdoor temperature for this minute
        controller.set_outdoor_temp(outdoor_fn(t))

        # Let any supervisor intervene BEFORE stepping
        if on_step is not None:
            on_step(t, controller)

        # Advance one minute
        controller.step()

        # Record the post-step state
        samples.append({
            "t_min": t,
            "indoor_temp": controller.indoor_temp,
            "setpoint": controller.setpoint,
            "control_signal": controller.control_signal,
        })

    return samples


if __name__ == "__main__":
    # Self-test: run S1 with a fixed-PID controller, no LLM.
    # This is essentially what C0 will do.
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from demo.metrics import compute_metrics, format_metrics

    print(f"Scenario: {S1['name']}")
    print(f"  {S1['description']}")
    print(f"  Duration: {S1['duration_min']} min")
    print(f"  Initial indoor_temp: {S1['initial_indoor_temp']}°C")
    print(f"  Setpoint: {S1['setpoint']}°C")
    print("")

    ctrl = make_controller_for(S1)
    samples = run_simulation(ctrl, S1)

    result = compute_metrics(samples)
    print(format_metrics(result, "FIXED_PID_S1"))
    print("")

    # Sanity: print first/mid/last sample
    print("Sample trajectory:")
    for idx in (0, 30, 60, 90, 119):
        s = samples[idx]
        print(f"  t={s['t_min']:>3}min  "
              f"indoor={s['indoor_temp']:>5.2f}°C  "
              f"setpoint={s['setpoint']:>4.1f}°C  "
              f"u={s['control_signal']:>6.1f}W")
