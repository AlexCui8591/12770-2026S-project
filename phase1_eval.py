"""
Phase 1 evaluation harness.

Runs the intent parser on a set of (query, expected) pairs and reports:
  - Overall parse success rate (raw LLM compliance)
  - Breakdown of failure types
  - Per-field accuracy on successful parses
  - Latency distribution

This script is the measurement instrument for H2 ("LLM can map NL to cost
weights"). The numbers it produces go directly into the final report.

Usage:
    uv run python -m tests.phase1_eval
"""

import json
import statistics
from collections import Counter
from pathlib import Path

from agent.intent_parser import parse_intent, ParseResult, ParseErrorType

TEST_CASES_PATH = Path(__file__).parent / "phase1_test_cases.json"
RESULTS_PATH = Path(__file__).parent / "phase1_results.json"


def evaluate_one(result: ParseResult, expected: dict) -> dict:
    """Compare parsed output against expected. Returns per-case report dict."""
    report = {
        "success": result.success,
        "error_type": result.error_type.value,
        "latency_ms": result.latency_ms,
        "json_repaired": result.json_repaired,
    }
    if not result.success:
        return report

    d = result.directive

    # Temperature: within ±1°C tolerance (LLM isn't a thermostat, don't demand exact)
    report["target_temp_close"] = abs(d.target_temperature - expected["target_temperature"]) <= 1.0

    # HVAC mode: exact match
    report["hvac_mode_match"] = d.hvac_mode.value == expected["hvac_mode"]

    # Preset: exact match (if expected provides one)
    if "preset_mode" in expected:
        report["preset_match"] = d.preset_mode.value == expected["preset_mode"]

    # Cost weights: we don't demand exact values, only that the TOP-PRIORITY
    # weight matches. This is what H2 really tests — does the LLM identify
    # which objective the user cares about most?
    exp_w = expected["cost_weights"]
    act_w = {
        "energy": d.cost_weights.energy,
        "comfort": d.cost_weights.comfort,
        "response": d.cost_weights.response,
    }
    report["top_weight_match"] = max(act_w, key=act_w.get) == max(exp_w, key=exp_w.get)

    # Stronger check: full ranking of the three weights matches
    exp_rank = sorted(exp_w, key=exp_w.get, reverse=True)
    act_rank = sorted(act_w, key=act_w.get, reverse=True)
    report["full_ranking_match"] = exp_rank == act_rank

    return report


def main():
    if not TEST_CASES_PATH.exists():
        print(f"ERROR: {TEST_CASES_PATH} not found.")
        print("Create this file with your 15 annotated test cases first.")
        print("See phase1_test_cases.template.json for the expected schema.")
        return

    with open(TEST_CASES_PATH) as f:
        cases = json.load(f)

    print(f"Running {len(cases)} test cases against qwen2.5:7b...\n")
    all_reports = []

    for i, case in enumerate(cases, 1):
        print(f"[{i:2d}/{len(cases)}] tag={case.get('tag', '?'):20s} | {case['query']}")
        result = parse_intent(case["query"])
        report = evaluate_one(result, case["expected"])
        report["query"] = case["query"]
        report["tag"] = case.get("tag", "")
        report["raw_output"] = result.raw_output
        if result.directive:
            report["directive"] = result.directive.model_dump()
        all_reports.append(report)

        marker = "OK " if result.success else "FAIL"
        print(f"         [{marker}] err={result.error_type.value} lat={result.latency_ms:.0f}ms", end="")
        if result.success:
            d = result.directive
            print(
                f" | T={d.target_temperature:.1f} mode={d.hvac_mode.value} "
                f"w=(e={d.cost_weights.energy:.2f},c={d.cost_weights.comfort:.2f},r={d.cost_weights.response:.2f})"
            )
        else:
            print()

    # ---- Summary ----
    n = len(all_reports)
    success = [r for r in all_reports if r["success"]]
    fail = [r for r in all_reports if not r["success"]]

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLIANCE REPORT")
    print("=" * 60)
    print(f"Total cases:        {n}")
    print(f"Parse success:      {len(success):2d}/{n} ({len(success) / n:.1%})")
    print(f"Parse failure:      {len(fail):2d}/{n}")

    if fail:
        print("\nFailure breakdown:")
        for err_type, count in Counter(r["error_type"] for r in fail).most_common():
            print(f"  {err_type:25s} {count}")

    if success:
        print("\nAccuracy on successful parses:")
        metrics = [
            ("target_temp_close (±1°C)", "target_temp_close"),
            ("hvac_mode exact match", "hvac_mode_match"),
            ("preset_mode exact match", "preset_match"),
            ("top-priority weight match", "top_weight_match"),
            ("full weight ranking match", "full_ranking_match"),
        ]
        for label, key in metrics:
            vals = [r.get(key) for r in success if key in r]
            if vals:
                rate = sum(bool(v) for v in vals) / len(vals)
                print(f"  {label:30s} {sum(bool(v) for v in vals):2d}/{len(vals)} ({rate:.1%})")

    # Latency stats
    latencies = [r["latency_ms"] for r in all_reports]
    if latencies:
        print(f"\nLatency (ms):  mean={statistics.mean(latencies):.0f}  "
              f"median={statistics.median(latencies):.0f}  "
              f"max={max(latencies):.0f}")

    # JSON repair stats
    repaired = sum(1 for r in all_reports if r.get("json_repaired"))
    if repaired:
        print(f"JSON locally repaired: {repaired}/{n}")

    # Save detailed results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
