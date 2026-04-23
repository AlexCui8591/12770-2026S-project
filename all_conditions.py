"""One-command runner for Demo Version A.

Runs C0, C1, C2 on the same scenario with the same user text and prints
a formatted comparison table.

Usage:
    python all_conditions.py
    python all_conditions.py --user "It's freezing in here!"
    python all_conditions.py --seed 42
"""

from __future__ import annotations

import argparse
import sys
import time

import importlib.util
import sys
from pathlib import Path

_demo_spec = importlib.util.spec_from_file_location(
    "demo_entry", Path(__file__).parent / "demo.py"
)
_demo_module = importlib.util.module_from_spec(_demo_spec)
sys.modules["demo_entry"] = _demo_module
_demo_spec.loader.exec_module(_demo_module)
run_c0 = _demo_module.run_c0
run_c1 = _demo_module.run_c1
run_c2 = _demo_module.run_c2

DEFAULT_USER = "Set the bedroom to 22°C"


def _fmt_row(cond: str, result: dict, elapsed: float) -> str:
    return (
        f"  {cond:<4} | MAD={result['MAD']:>5.2f}°C | "
        f"CE={result['CE']:>5.2f}kWh | "
        f"RT={result['RT']:>4}min | "
        f"OC={result['OC']:>3} | "
        f"{elapsed:>5.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all conditions and compare")
    parser.add_argument("--user", default=DEFAULT_USER,
                        help=f"User query text (default: {DEFAULT_USER!r})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducibility")
    parser.add_argument("--skip-c2", action="store_true",
                        help="Skip C2 (faster, for dev)")
    args = parser.parse_args()

    print(f"Scenario: S1_steady_state (cold start 18°C → setpoint 22°C, 120 min)")
    print(f"User:     {args.user!r}")
    if args.seed is not None:
        print(f"Seed:     {args.seed}")
    print("")

    results = {}

    # -- C0 --
    print("[C0] Fixed PID baseline...")
    t0 = time.time()
    results["C0"] = run_c0(seed=args.seed)
    elapsed_c0 = time.time() - t0
    print(_fmt_row("C0", results["C0"], elapsed_c0))
    print("")

    # -- C1 --
    print("[C1] LLM one-shot setpoint translation + fixed PID...")
    t0 = time.time()
    results["C1"] = run_c1(args.user, seed=args.seed)
    elapsed_c1 = time.time() - t0
    print(_fmt_row("C1", results["C1"], elapsed_c1))
    print("")

    # -- C2 --
    if not args.skip_c2:
        print("[C2] LLM supervisor every 5 min (reactive)...")
        t0 = time.time()
        results["C2"] = run_c2(args.user, seed=args.seed)
        elapsed_c2 = time.time() - t0
        print(_fmt_row("C2", results["C2"], elapsed_c2))
        print("")

    # -- Comparison table --
    print("=" * 60)
    print("  Summary (lower is better for all metrics)")
    print("=" * 60)
    print(f"  {'Cond':<4} | {'MAD (°C)':<9} | {'CE (kWh)':<9} | "
          f"{'RT (min)':<9} | {'OC':<4} | Time")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*4}-+-{'-'*6}")
    for cond in ("C0", "C1", "C2"):
        if cond not in results:
            continue
        r = results[cond]
        elapsed = {"C0": elapsed_c0, "C1": elapsed_c1,
                   "C2": elapsed_c2 if not args.skip_c2 else 0}[cond]
        print(_fmt_row(cond, r, elapsed))
    print("")

    # -- Highlight C2 vs C0 improvement if both present --
    if "C0" in results and "C2" in results:
        c0, c2 = results["C0"], results["C2"]
        mad_change = (c2["MAD"] - c0["MAD"]) / c0["MAD"] * 100
        ce_change = (c2["CE"] - c0["CE"]) / c0["CE"] * 100
        rt_change = (c2["RT"] - c0["RT"]) / c0["RT"] * 100
        print("  C2 vs C0:")
        print(f"    MAD: {mad_change:+.1f}% {'(better)' if mad_change < 0 else '(worse)'}")
        print(f"    CE:  {ce_change:+.1f}% {'(better)' if ce_change < 0 else '(worse)'}")
        print(f"    RT:  {rt_change:+.1f}% {'(better)' if rt_change < 0 else '(worse)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
