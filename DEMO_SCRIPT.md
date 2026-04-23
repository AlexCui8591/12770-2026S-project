# Demo Script — Prepared Queries

These are the user queries to test and showcase during demo. Each picked to exercise a specific LLM behavior. Pre-test each one so there are no surprises on demo day.

## Primary demo query (most neutral, best contrast)

**Query**: `Set the bedroom to 22°C`

**Expected**:
- C0: MAD ≈ 0.9, RT ≈ 55 (baseline)
- C1: MAD ≈ 0.9, RT ≈ 55 (LLM sets same setpoint, no real difference)
- C2: MAD ≈ 0.7, RT ≈ 46 (supervisor tunes Kp early, MAD improves ~25%)

**Story**: "Same user request, three different levels of LLM involvement. C2 demonstrates that online supervision measurably improves tracking."

## Alternative queries (use if time permits, or if C2 fails with primary)

### Query A: Strong discomfort signal
`It's freezing in here, do something!`

Intent parser should produce target ≈ 22°C (comfort zone) with high w_comfort. Same PID dynamics as primary, but shows LLM understands emotional intent.

### Query B: Energy-conscious intent
`Keep it warm but minimize electricity costs`

Intent parser should produce target ≈ 20°C with high w_energy. C2 might see slightly different behavior because initial setpoint differs.

**Caveat**: C2 does NOT use cost_weights (Version A scope); only target_temperature feeds into PID. So this query mostly showcases the intent parser, not C2's difference from C1.

### Query C: Relative adjustment
`Make it a bit cooler`

Tests whether intent parser can infer target from context. Expected target ≈ current - 1°C (so around 17°C given initial 18°C). This produces WEIRD results because setpoint below initial creates backward dynamics.

**Caveat**: Use only if time permits and you're prepared to explain the weirdness.

## Demo flow recommendation

```
1. Show README (30 sec)   — orient audience
2. Run C0 with --seed 42   — show baseline
3. Run C1 with primary query   — show intent parsing
4. Run C2 with primary query   — the main event; show supervisor decisions
5. Run all_conditions.py with primary query   — final summary table
```

Total time: ~8 minutes live (C2 takes 2-4 min due to 23 LLM calls).

## Pre-demo checklist (run these tonight before sleeping)

- [ ] `ollama serve` is running
- [ ] `uv run python demo.py --condition C0 --seed 42` — baseline check
- [ ] `uv run python demo.py --condition C1 --user "Set the bedroom to 22°C"` — C1 works
- [ ] `uv run python demo.py --condition C2 --user "Set the bedroom to 22°C"` — C2 works, reasonable numbers
- [ ] `uv run python all_conditions.py` — summary table prints
- [ ] `git pull` — confirm Alex's C3 if he pushed anything
- [ ] One of the alternative queries also works (as backup)

## Fallback plan

**If on demo day C2 suddenly fails**: fall back to C0 + C1 narrative. You can say: "C2 involves 23 consecutive LLM calls over 4 minutes; occasionally the local model times out. Our dual-strategy fallback catches this and holds current parameters — in our tested runs, C2 achieves X% improvement on MAD."

**If Ollama dies**: restart with `ollama serve &` in a new terminal. Qwen model is cached locally so first call takes a few seconds to warm up.
