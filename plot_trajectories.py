"""Plot indoor_temp trajectories for C0/C1/C2 on S1.

Runs each condition independently, collects per-minute temperature samples,
and plots all three on one figure for visual comparison.

Usage:
    uv run python plot_trajectories.py
    uv run python plot_trajectories.py --user "It's freezing in here!"
    uv run python plot_trajectories.py --save fig.png
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime


def _run_and_collect(condition: str, user_text: str, seed: int = 42) -> tuple[list, dict]:
    """Run a condition, return (samples, metrics)."""
    random.seed(seed)
    from demo.scenarios import S1, make_controller_for, run_simulation
    from demo.metrics import compute_metrics

    if condition == "C0":
        controller = make_controller_for(S1)
        samples = run_simulation(controller, S1, on_step=None)
        return samples, compute_metrics(samples)

    # C1 and C2 need intent parsing
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
    intent = parse(AgentInput(user_command=user_text, current_context=context))
    target = intent.target_temperature

    scenario = dict(S1)
    scenario["setpoint"] = target
    controller = make_controller_for(scenario)

    if condition == "C1":
        samples = run_simulation(controller, scenario, on_step=None)
        return samples, compute_metrics(samples)

    if condition == "C2":
        from agent.supervisor import ask_supervisor, apply_decision

        def supervisor_cb(t_min, ctrl):
            if t_min == 0 or t_min % 5 != 0:
                return
            tel = ctrl.get_telemetry()
            d = ask_supervisor(tel, user_text)
            apply_decision(ctrl, d)

        samples = run_simulation(controller, scenario, on_step=supervisor_cb)
        return samples, compute_metrics(samples)

    raise ValueError(f"Unknown condition: {condition}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="Set the bedroom to 22°C")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, help="Save figure to file (otherwise show)")
    args = ap.parse_args()

    try:
        import matplotlib
        if args.save:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: uv add matplotlib")
        return 1

    print(f"User: {args.user!r}   Seed: {args.seed}")
    print("")

    traces = {}
    for cond in ("C0", "C1", "C2"):
        print(f"Running {cond}...")
        samples, metrics = _run_and_collect(cond, args.user, seed=args.seed)
        traces[cond] = (samples, metrics)
        print(f"  MAD={metrics['MAD']:.2f}°C  CE={metrics['CE']:.2f}kWh  "
              f"RT={metrics['RT']}min  OC={metrics['OC']}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"C0": "#6b7280", "C1": "#3b82f6", "C2": "#10b981"}
    styles = {"C0": "-", "C1": "-", "C2": "-"}

    for cond, (samples, metrics) in traces.items():
        t = [s["t_min"] for s in samples]
        y = [s["indoor_temp"] for s in samples]
        label = (f"{cond}: MAD={metrics['MAD']:.2f}°C, "
                 f"RT={metrics['RT']}min")
        ax.plot(t, y, label=label, color=colors[cond],
                linestyle=styles[cond], linewidth=1.8)

    # Setpoint reference (use C0's, which is the scenario default)
    setpoint = traces["C0"][0][0]["setpoint"]
    ax.axhline(y=setpoint, color="red", linestyle=":", alpha=0.5,
               label=f"setpoint ({setpoint}°C)")
    ax.axhspan(setpoint - 0.5, setpoint + 0.5, color="red", alpha=0.08,
               label="± 0.5°C comfort band")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Indoor temperature (°C)")
    ax.set_title(f"S1 Steady-State Tracking  |  User: {args.user!r}")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 120)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"\nSaved to {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
