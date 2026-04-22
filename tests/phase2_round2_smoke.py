"""Phase 2 Round 2 smoke test.

Verifies:
  1. MockPIDController produces the 11 required telemetry fields
  2. step() actually changes state (not a constant mock)
  3. set_gains() / set_setpoint() take effect
  4. Each scenario preset produces distinct, plausible telemetry
  5. get_pid_telemetry() tool returns the expected shape
  6. get_pid_state alias points to the same function

Usage:  uv run python -m tests.phase2_round2_smoke
"""

from __future__ import annotations

import json

from agent.mock_pid import (
    MockPIDController,
    DEFAULT_CONTROLLER,
    install_scenario,
    SCENARIOS,
)
from agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS, dispatch_tool_call


REQUIRED_FIELDS = {
    "timestamp", "kp", "ki", "kd",
    "setpoint", "indoor_temp", "tracking_error",
    "control_signal", "cumulative_energy_kwh",
    "oscillation_count", "cost_J",
}


def check_telemetry_shape(tel: dict, context: str) -> list[str]:
    """Return a list of problems. Empty list == pass."""
    problems = []
    missing = REQUIRED_FIELDS - set(tel.keys())
    if missing:
        problems.append(f"[{context}] missing fields: {sorted(missing)}")
    extra_ok = {"room", "source"}  # tool layer additions are fine
    unknown = set(tel.keys()) - REQUIRED_FIELDS - extra_ok
    if unknown:
        problems.append(f"[{context}] unexpected fields: {sorted(unknown)}")
    # Type sanity
    for f in ("kp", "ki", "kd", "setpoint", "indoor_temp", "tracking_error",
              "control_signal", "cumulative_energy_kwh", "cost_J"):
        if f in tel and not isinstance(tel[f], (int, float)):
            problems.append(f"[{context}] field {f} not numeric: {tel[f]!r}")
    if "oscillation_count" in tel and not isinstance(tel["oscillation_count"], int):
        problems.append(f"[{context}] oscillation_count not int: {tel['oscillation_count']!r}")
    if "timestamp" in tel and not isinstance(tel["timestamp"], str):
        problems.append(f"[{context}] timestamp not a string: {tel['timestamp']!r}")
    return problems


def test_basic_telemetry_shape() -> None:
    print("--- Test 1: fresh MockPIDController returns 11 fields ---")
    ctrl = MockPIDController()
    tel = ctrl.get_telemetry()
    problems = check_telemetry_shape(tel, "fresh_controller")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        raise AssertionError("shape check failed")
    print(f"  OK  — fields: {sorted(tel.keys())}")
    print()


def test_step_changes_state() -> None:
    print("--- Test 2: step() actually advances state ---")
    ctrl = MockPIDController(setpoint=22.0, indoor_temp=18.0)
    before = ctrl.get_telemetry()
    for _ in range(5):
        ctrl.step()
    after = ctrl.get_telemetry()

    changes = []
    if before["timestamp"] == after["timestamp"]:
        changes.append("timestamp did not advance")
    if abs(before["indoor_temp"] - after["indoor_temp"]) < 0.01:
        changes.append(f"indoor_temp barely moved: {before['indoor_temp']} -> {after['indoor_temp']}")
    if abs(before["cumulative_energy_kwh"] - after["cumulative_energy_kwh"]) < 1e-6:
        changes.append("cumulative_energy_kwh did not increase")

    if changes:
        for c in changes:
            print(f"  FAIL: {c}")
        raise AssertionError("step() did not produce state changes")
    print(f"  OK  — temp {before['indoor_temp']:.2f} -> {after['indoor_temp']:.2f}, "
          f"energy {before['cumulative_energy_kwh']:.4f} -> {after['cumulative_energy_kwh']:.4f}")
    print()


def test_set_gains_and_setpoint() -> None:
    print("--- Test 3: set_gains() and set_setpoint() take effect ---")
    ctrl = MockPIDController()
    ctrl.set_gains(kp=4.5, ki=0.1, kd=1.2)
    ctrl.set_setpoint(24.5)
    tel = ctrl.get_telemetry()
    assert tel["kp"] == 4.5, f"kp not applied: {tel['kp']}"
    assert tel["ki"] == 0.1, f"ki not applied: {tel['ki']}"
    assert tel["kd"] == 1.2, f"kd not applied: {tel['kd']}"
    assert tel["setpoint"] == 24.5, f"setpoint not applied: {tel['setpoint']}"
    print(f"  OK  — gains=({tel['kp']}, {tel['ki']}, {tel['kd']}) setpoint={tel['setpoint']}")
    print()


def test_scenarios_distinct() -> None:
    print("--- Test 4: each scenario produces a distinct telemetry fingerprint ---")
    signatures = {}
    for name in SCENARIOS:
        ctrl = install_scenario(name)
        tel = ctrl.get_telemetry()
        # Fingerprint on the dynamic fields (not kp/ki/kd, which aren't
        # always changed across scenarios)
        sig = (
            round(tel["tracking_error"], 2),
            round(tel["indoor_temp"], 2),
            tel["oscillation_count"],
            round(tel["cost_J"], 2),
        )
        signatures[name] = sig
        print(f"  {name:16s} err={tel['tracking_error']:>5.2f}  "
              f"temp={tel['indoor_temp']:>5.2f}  "
              f"osc={tel['oscillation_count']:>2d}  "
              f"J={tel['cost_J']:>6.2f}")
    # All four should be unique
    unique = set(signatures.values())
    if len(unique) != len(SCENARIOS):
        raise AssertionError(f"scenarios not distinct: {signatures}")
    print(f"  OK  — {len(unique)} distinct signatures")
    print()


def test_tool_registration() -> None:
    print("--- Test 5: get_pid_telemetry + get_pid_state registered ---")
    problems = []
    if "get_pid_telemetry" not in TOOL_REGISTRY:
        problems.append("get_pid_telemetry not in TOOL_REGISTRY")
    if "get_pid_state" not in TOOL_REGISTRY:
        problems.append("get_pid_state not in TOOL_REGISTRY")
    schema_names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    if "get_pid_telemetry" not in schema_names:
        problems.append("get_pid_telemetry not in TOOL_SCHEMAS")
    # Alias should NOT appear in schemas (convention from Round 1)
    if "get_pid_state" in schema_names:
        problems.append("get_pid_state should be alias-only, not in TOOL_SCHEMAS")

    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        raise AssertionError("registration check failed")
    print(f"  OK  — registry now has {len(TOOL_REGISTRY)} entries, "
          f"schemas has {len(TOOL_SCHEMAS)}")
    print()


def test_tool_dispatch() -> None:
    print("--- Test 6: dispatch_tool_call('get_pid_telemetry') returns telemetry ---")
    install_scenario("oscillating")
    result = dispatch_tool_call("get_pid_telemetry", {"room": "bedroom"})
    if result is None:
        raise AssertionError("dispatch returned None")
    problems = check_telemetry_shape(result, "tool_output")
    if problems:
        for p in problems:
            print(f"  FAIL: {p}")
        raise AssertionError("tool output shape invalid")
    # Tool-specific fields
    assert result.get("room") == "bedroom", f"room not preserved: {result.get('room')}"
    assert result.get("source") == "mock", f"source not set: {result.get('source')}"
    # Scenario content
    assert result["oscillation_count"] >= 10, \
        f"oscillating scenario should have high osc_count, got {result['oscillation_count']}"
    print(f"  OK  — tool returned {len(result)} fields, osc_count={result['oscillation_count']}")
    print()


def test_alias_equivalence() -> None:
    print("--- Test 7: get_pid_state returns same shape as get_pid_telemetry ---")
    install_scenario("steady_state")
    main = dispatch_tool_call("get_pid_telemetry", {"room": "test"})
    alias = dispatch_tool_call("get_pid_state", {"room": "test"})
    if main is None or alias is None:
        raise AssertionError("one of the calls returned None")
    # They're independent calls, timestamps might differ by microseconds, but
    # the shape and controller state should match
    if set(main.keys()) != set(alias.keys()):
        raise AssertionError(f"key mismatch: {set(main.keys()) ^ set(alias.keys())}")
    print(f"  OK  — both calls returned same field set")
    print()


def main() -> None:
    tests = [
        test_basic_telemetry_shape,
        test_step_changes_state,
        test_set_gains_and_setpoint,
        test_scenarios_distinct,
        test_tool_registration,
        test_tool_dispatch,
        test_alias_equivalence,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  *** ASSERTION FAILED: {e} ***\n")
        except Exception as e:
            print(f"  *** UNEXPECTED ERROR ({type(e).__name__}): {e} ***\n")

    print("=" * 60)
    print(f"  PHASE 2 ROUND 2 SMOKE: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    if passed == len(tests):
        print("  Green. Ready for Phase 5 (supervisor loop).")
    else:
        print("  *** Some tests failed — fix before moving on. ***")


if __name__ == "__main__":
    main()
