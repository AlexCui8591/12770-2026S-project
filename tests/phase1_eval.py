"""Phase 1 evaluation harness.

Runs the agent on 15 (query, expected) pairs and reports:
  - Overall parse success rate (did parse() return at all?)
  - Validator-fix rate (how often did validate_and_fix patch something?)
  - Per-field accuracy on successful parses
  - cost_weights top-priority + full-ranking match (the H2 test)
  - Latency distribution

Usage:  uv run python -m tests.phase1_eval
"""

from __future__ import annotations

import json
import statistics
import traceback
from collections import Counter
from pathlib import Path

from agent.parser import parse
from agent.schema import AgentInput, ParseError, SchemaError

TEST_CASES_PATH = Path(__file__).parent / "phase1_test_cases.json"
RESULTS_PATH = Path(__file__).parent / "phase1_results.json"

DEFAULT_CONTEXT = {
    "room": "living_room",
    "current_temperature": 23.0,
    "current_hvac_mode": "auto",
    "current_fan_mode": "auto",
    "occupied": True,
    "window_open": False,
    "time": "2026-04-22T14:00:00",
}


def _build_agent_input(case: dict) -> AgentInput:
    ctx = {**DEFAULT_CONTEXT, **case.get("context_overrides", {})}
    return AgentInput.from_dict({
        "user_command": case["query"],
        "current_context": ctx,
    })


def _detect_validator_fixes(reason: str) -> list[str]:
    """Scan the output's reason string for validator-fix markers."""
    fixes = []
    if "Validation adjusted:" in reason:
        body = reason.split("Validation adjusted:", 1)[1]
        for marker in [
            "target_temperature",
            "hvac_mode",
            "preset_mode",
            "fan_mode",
            "cost_weights",
            "room was missing",
        ]:
            if marker in body:
                fixes.append(marker)
    return fixes


def evaluate_one(case: dict) -> dict:
    """Run one test case and produce a per-case report dict."""
    import time
    start = time.time()
    agent_input = _build_agent_input(case)
    report: dict = {
        "tag": case.get("tag", ""),
        "query": case["query"],
    }

    try:
        output = parse(agent_input)
        report["success"] = True
        report["error"] = None
        report["error_type"] = "none"
        report["output"] = output.to_dict()
        report["validator_fixes"] = _detect_validator_fixes(output.reason)
    except ParseError as e:
        report.update({"success": False, "error": str(e), "error_type": "json_invalid"})
        report["output"] = None
        report["validator_fixes"] = []
    except SchemaError as e:
        report.update({"success": False, "error": str(e), "error_type": "missing_or_schema"})
        report["output"] = None
        report["validator_fixes"] = []
    except Exception as e:
        report.update({"success": False, "error": f"{type(e).__name__}: {e}", "error_type": "other"})
        report["output"] = None
        report["validator_fixes"] = []
        report["traceback"] = traceback.format_exc()

    report["latency_ms"] = (time.time() - start) * 1000

    # Accuracy checks (only if successful)
    if report["success"]:
        exp = case["expected"]
        out = report["output"]
        report["target_temp_close"] = abs(out["target_temperature"] - exp["target_temperature"]) <= 1.0
        report["hvac_mode_match"] = out["hvac_mode"] == exp["hvac_mode"]
        if "preset_mode" in exp:
            report["preset_match"] = out["preset_mode"] == exp["preset_mode"]

        # cost_weights analysis (core H2 test)
        exp_w = exp["cost_weights"]
        act_w = out["cost_weights"]
        report["top_weight_match"] = (
            max(act_w, key=act_w.get) == max(exp_w, key=exp_w.get)
        )
        report["full_ranking_match"] = (
            sorted(act_w, key=act_w.get, reverse=True)
            == sorted(exp_w, key=exp_w.get, reverse=True)
        )
        # Was cost_weights defaulted by the validator? (= LLM didn't output it)
        report["cost_weights_defaulted"] = "cost_weights" in report["validator_fixes"]

    return report


def print_summary(reports: list[dict]) -> None:
    n = len(reports)
    success = [r for r in reports if r["success"]]
    fail = [r for r in reports if not r["success"]]

    print("\n" + "=" * 66)
    print("  PHASE 1 COMPLIANCE REPORT")
    print("=" * 66)
    print(f"  Total cases:        {n}")
    print(f"  Parse success:      {len(success):2d}/{n} ({len(success) / n:.1%})")
    print(f"  Parse failure:      {len(fail):2d}/{n}")

    if fail:
        print("\n  Failure breakdown:")
        for err, c in Counter(r["error_type"] for r in fail).most_common():
            print(f"    {err:25s} {c}")

    if success:
        print("\n  Accuracy on successful parses:")
        metrics = [
            ("target_temp within ±1°C", "target_temp_close"),
            ("hvac_mode exact match", "hvac_mode_match"),
            ("preset_mode exact match", "preset_match"),
            ("top cost_weight match (H2)", "top_weight_match"),
            ("full weight ranking match", "full_ranking_match"),
        ]
        for label, key in metrics:
            vals = [r.get(key) for r in success if key in r]
            if vals:
                hits = sum(bool(v) for v in vals)
                print(f"    {label:32s} {hits:2d}/{len(vals)} ({hits / len(vals):.1%})")

        # Validator intervention stats (important for Phase 4 motivation)
        fix_counter: Counter = Counter()
        for r in success:
            for f in r.get("validator_fixes", []):
                fix_counter[f] += 1
        if fix_counter:
            print("\n  Validator interventions (lower is better — raw LLM quality):")
            for field, c in fix_counter.most_common():
                print(f"    {field:32s} {c}/{len(success)} ({c / len(success):.1%})")

        defaulted = sum(1 for r in success if r.get("cost_weights_defaulted"))
        if defaulted:
            print(
                f"\n  [!] cost_weights was DEFAULTED by validator in {defaulted}/{len(success)} "
                "cases — LLM did not produce valid weights for those queries."
            )

    latencies = [r["latency_ms"] for r in reports if r["latency_ms"]]
    if latencies:
        print(
            f"\n  Latency (ms): mean={statistics.mean(latencies):.0f}  "
            f"median={statistics.median(latencies):.0f}  "
            f"max={max(latencies):.0f}"
        )


def main() -> None:
    if not TEST_CASES_PATH.exists():
        print(f"ERROR: {TEST_CASES_PATH} not found.")
        return

    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Running {len(cases)} cases against qwen2.5:7b\n")
    reports: list[dict] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i:2d}/{len(cases)}] {case.get('tag', ''):40s} | {case['query']}")
        r = evaluate_one(case)
        reports.append(r)
        marker = "OK" if r["success"] else "FAIL"
        extra = ""
        if r["success"]:
            out = r["output"]
            w = out["cost_weights"]
            extra = (
                f" | T={out['target_temperature']:.1f} mode={out['hvac_mode']:4s} "
                f"preset={out['preset_mode']:7s} "
                f"w=(e{w['energy']:.2f},c{w['comfort']:.2f},r{w['response']:.2f})"
            )
            if r.get("validator_fixes"):
                extra += f"  [FIX: {','.join(r['validator_fixes'])}]"
        print(f"     [{marker}] err={r['error_type']:18s} lat={r['latency_ms']:5.0f}ms{extra}")

    print_summary(reports)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"\n  Detailed results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
