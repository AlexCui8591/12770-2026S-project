"""Demo Version A — main entry point.

Usage:
    python demo.py --condition C0
    python demo.py --condition C1 --user "Set the bedroom to 22°C"
    python demo.py --condition C2 --user "Set the bedroom to 22°C"
    python demo.py --condition C3 --user "..."   # handed off to Alex

C0: Fixed PID baseline, no LLM involvement.
C1: LLM translates user text into a setpoint; PID is static.
C2: LLM supervises PID every 5 minutes, reactive.
C3: C2 + forecast tools, proactive. (Alex)
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime

from demo.scenarios import S1, make_controller_for, run_simulation
from demo.metrics import compute_metrics, format_metrics


# ---------------------------------------------------------------------------
# C0 — Fixed PID baseline
# ---------------------------------------------------------------------------

def run_c0(seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)

    controller = make_controller_for(S1)
    samples = run_simulation(controller, S1, on_step=None)
    return compute_metrics(samples)


# ---------------------------------------------------------------------------
# C1 — LLM one-shot setpoint translation, fixed PID
# ---------------------------------------------------------------------------

def _parse_user_intent(user_text: str):
    """Shared helper: run the intent parser on user_text, return AgentOutput."""
    from agent.parser import parse
    from agent.schema import AgentInput, CurrentContext

    context = CurrentContext(
        room="bedroom",
        current_temperature=S1["initial_indoor_temp"],
        current_hvac_mode="off",
        current_fan_mode="auto",
        occupied=True,
        window_open=False,
        time=datetime.now().replace(microsecond=0).isoformat(),
    )
    agent_input = AgentInput(
        user_command=user_text,
        current_context=context,
    )
    return parse(agent_input)


def run_c1(user_text: str, seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)

    print(f"    Calling LLM to parse intent...")
    result = _parse_user_intent(user_text)
    target = result.target_temperature
    print(f"    LLM produced: target_temperature = {target}°C "
          f"(mode={result.hvac_mode}, preset={result.preset_mode})")

    scenario = dict(S1)
    scenario["setpoint"] = target

    controller = make_controller_for(scenario)
    samples = run_simulation(controller, scenario, on_step=None)
    return compute_metrics(samples)


# ---------------------------------------------------------------------------
# C2 — LLM supervisor every 5 minutes (reactive)
# ---------------------------------------------------------------------------

SUPERVISOR_INTERVAL_MIN = 5


def run_c2(user_text: str, seed: int | None = None) -> dict:
    """Condition C2: LLM supervisor reads telemetry every 5 min, decides
    hold/set_pid/set_setpoint. MockPID starts with default gains and
    LLM-translated setpoint (just like C1's starting point)."""
    if seed is not None:
        random.seed(seed)

    from agent.supervisor import ask_supervisor, apply_decision

    # Step 1: same as C1 — parse user intent, get initial setpoint
    print(f"    Calling LLM to parse intent...")
    intent = _parse_user_intent(user_text)
    target = intent.target_temperature
    print(f"    Initial setpoint from intent parser: {target}°C "
          f"(mode={intent.hvac_mode}, preset={intent.preset_mode})")

    scenario = dict(S1)
    scenario["setpoint"] = target
    controller = make_controller_for(scenario)

    # Step 2: build supervisor callback
    decision_log = []

    def supervisor_callback(t_min: int, ctrl) -> None:
        # Only wake up every 5 min, and skip t=0 (let PID do one step first)
        if t_min == 0 or t_min % SUPERVISOR_INTERVAL_MIN != 0:
            return

        telemetry = ctrl.get_telemetry()
        decision = ask_supervisor(telemetry, user_text)
        apply_decision(ctrl, decision)
        decision_log.append({"t_min": t_min, "decision": decision})

        action = decision["action"]
        if action == "hold":
            marker = "  "
        else:
            marker = " ↑" if action == "set_pid" else " →"
        print(f"    t={t_min:>3}min  supervisor: {action:<13}{marker}  "
              f"[{decision['rationale'][:70]}]")

    # Step 3: run simulation
    samples = run_simulation(controller, scenario, on_step=supervisor_callback)

    # Brief decision summary
    action_counts = {}
    for entry in decision_log:
        a = entry["decision"]["action"]
        action_counts[a] = action_counts.get(a, 0) + 1
    print(f"    Supervisor made {len(decision_log)} decisions: {action_counts}")

    return compute_metrics(samples)


# ---------------------------------------------------------------------------
# C3 — handed off to Alex
# ---------------------------------------------------------------------------

def run_c3(user_text: str, seed: int | None = None) -> dict:
    raise NotImplementedError(
        "C3 (proactive supervisor with forecast tools) is handed off to Alex. "
        "See HANDOFF_C2_C3.md for interface contract."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo Version A: 12-770 HVAC LLM Agent"
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["C0", "C1", "C2", "C3"],
        help="Experimental condition to run",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="User natural-language request (required for C1/C2/C3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)",
    )
    args = parser.parse_args()

    print(f"=== Running condition {args.condition} on scenario {S1['name']} ===")
    print(f"    {S1['description']}")
    if args.user:
        print(f'    User: "{args.user}"')
    if args.seed is not None:
        print(f"    Seed: {args.seed}")
    print("")

    if args.condition == "C0":
        if args.user:
            print("Note: C0 ignores --user (no LLM involvement)")
        result = run_c0(seed=args.seed)
    elif args.condition == "C1":
        if not args.user:
            print("Error: C1 requires --user", file=sys.stderr)
            return 1
        result = run_c1(args.user, seed=args.seed)
    elif args.condition == "C2":
        if not args.user:
            print("Error: C2 requires --user", file=sys.stderr)
            return 1
        result = run_c2(args.user, seed=args.seed)
    elif args.condition == "C3":
        if not args.user:
            print("Error: C3 requires --user", file=sys.stderr)
            return 1
        result = run_c3(args.user, seed=args.seed)
    else:
        raise AssertionError("unreachable")

    print("")
    print("=== Result ===")
    print(format_metrics(result, args.condition))
    return 0


if __name__ == "__main__":
    sys.exit(main())
