"""
test_runner.py — Test suite + accuracy report for the Apex Strategy Agent.

Deterministic tests (no API key needed):
  1. Reproducibility    — same inputs => byte-identical projection.
  2. Hand-verified math — model output equals numbers calculated by hand
                          (proves the numbers come from the model, not an LLM).
  3. Scenario set       — best/base/worst generated and correctly ordered.
  4. Insufficient input — the graph asks for missing data instead of simulating
                          (runs the real graph; ingest short-circuits before any
                          LLM call, so no key is required).
  5. Schema enforcement — a Recommendation cannot omit assumptions/sensitivities/
                          risks.

Live test (only when ANTHROPIC_API_KEY is set; otherwise SKIPPED, not failed):
  6. Full end-to-end on sample inputs — status ok, verification passes, the
     recommendation carries assumptions + confidence + sensitivities, and its
     figures match a fresh deterministic model run.

Exit code 0 iff all RUN tests pass (skipped live test does not fail the suite).
"""
from __future__ import annotations

import json
import logging

import config
import financial_model as fm
from graph import build_graph, run
from sources import insufficient_inputs, sample_inputs
from state import BusinessMetrics, Recommendation, SimulationParams

logging.getLogger("apex").setLevel(logging.WARNING)

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Hand-verifiable fixture
# ---------------------------------------------------------------------------
HAND_METRICS = BusinessMetrics(
    company_name="HandCalc Inc.", customers=1000, arpu_monthly=100.0,
    monthly_churn_rate=0.05, monthly_gross_adds=80,
    variable_cost_per_customer=10.0, fixed_costs_monthly=5000.0,
    support_cost_monthly=2000.0, marketing_spend_monthly=1000.0,
    headcount=10, cost_per_head_monthly=3000.0)


def test_reproducibility() -> None:
    p = SimulationParams(price_change_pct=8, support_cost_change_pct=-15, horizon_months=18)
    a = fm.simulate(HAND_METRICS, p)
    b = fm.simulate(HAND_METRICS, p)
    same = a.model_dump_json() == b.model_dump_json()
    check("1. Reproducibility (same inputs => identical output)", same)


def test_hand_math_no_change() -> None:
    # price 0% => sensitivities vanish; clean hand calculation.
    p = SimulationParams(price_change_pct=0, horizon_months=1)
    m = fm.simulate(HAND_METRICS, p).months[0]
    # By hand:
    #   customers = 1000*(1-0.05)+80 = 1030
    #   revenue   = 1030*100 = 103000
    #   variable  = 1030*10  = 10300
    #   payroll   = 10*3000  = 30000
    #   cost      = 5000+2000+10300+30000+1000 = 48300
    #   margin    = 103000-48300 = 54700
    ok = (m.customers == 1030.0 and m.revenue == 103000.0
          and m.total_cost == 48300.0 and m.margin == 54700.0)
    check("2a. Hand math, no price change", ok,
          f"customers={m.customers} revenue={m.revenue} cost={m.total_cost} margin={m.margin}")


def test_hand_math_price_change() -> None:
    # price +10% exercises churn & acquisition elasticity exactly.
    p = SimulationParams(price_change_pct=10, horizon_months=1)
    m = fm.simulate(HAND_METRICS, p).months[0]
    # By hand:
    #   arpu      = 100*1.10 = 110
    #   churn     = 0.05*(1+0.60*0.10) = 0.053
    #   adds      = 80*(1+(-0.80)*0.10) = 73.6
    #   customers = 1000*(1-0.053)+73.6 = 1020.6
    #   revenue   = 1020.6*110 = 112266.0
    #   variable  = 1020.6*10  = 10206.0
    #   cost      = 5000+2000+10206+30000+1000 = 48206.0
    #   margin    = 112266 - 48206 = 64060.0
    ok = (m.arpu == 110.0 and m.churn_rate == 0.053 and m.customers == 1020.6
          and m.revenue == 112266.0 and m.total_cost == 48206.0 and m.margin == 64060.0)
    check("2b. Hand math, +10% price (elasticity applied)", ok,
          f"customers={m.customers} churn={m.churn_rate} revenue={m.revenue} margin={m.margin}")


def test_scenarios() -> None:
    p = SimulationParams(price_change_pct=8, support_cost_change_pct=-15, horizon_months=18)
    ss = fm.generate_scenarios(HAND_METRICS, p)
    labels = {c.label for c in ss.cases}
    by = {c.label: c.summary["final_monthly_margin"] for c in ss.cases}
    ordered = by["worst"] <= by["base"] <= by["best"]
    check("3. Scenario set generated and ordered (worst<=base<=best)",
          labels == {"best", "base", "worst"} and ordered,
          f"{ {k: round(v) for k, v in by.items()} }")


def test_insufficient_input() -> None:
    # Runs the REAL graph; ingest detects missing fields before any LLM call.
    state = run(insufficient_inputs(), thread_id="test-insufficient", app=build_graph())
    summ = state.get("summary", {})
    asked = summ.get("status") == "insufficient_input"
    has_missing = bool(summ.get("missing_fields"))
    check("4. Insufficient input asks for data (no fabricated projection)",
          asked and has_missing, f"missing={summ.get('missing_fields')}")


def test_schema_enforcement() -> None:
    try:
        Recommendation(headline="x", rationale="y", confidence=0.5,
                       confidence_label="moderate", assumptions=[],
                       sensitivities=["s"], key_risks=["r"], model_used="m")
        check("5. Schema rejects missing assumptions", False, "no error raised")
    except Exception:
        check("5. Schema rejects missing assumptions", True)


def test_live_end_to_end() -> bool:
    if not config.api_key_present():
        print("[SKIP] 6. Live end-to-end - ANTHROPIC_API_KEY not set "
              "(deterministic tests still validate the math).")
        return True  # skip does not fail the suite
    state = run(sample_inputs(), thread_id="test-live", app=build_graph())
    summ = state.get("summary", {})
    rec = state.get("recommendation")
    ver = state.get("verification")
    ok = (summ.get("status") == "ok" and ver is not None and ver.passed
          and rec is not None and len(rec.assumptions) >= 1
          and len(rec.sensitivities) >= 1 and 0.0 <= rec.confidence <= 1.0)
    # numbers integrity: recommendation base case == fresh deterministic run
    if rec:
        fresh = fm.generate_scenarios(state["metrics"], state["projection"].params)
        base = next(c.summary for c in fresh.cases if c.label == "base")
        ok = ok and (rec.projected_outcomes.get("base_case") == base)
    check("6. Live end-to-end (status ok, verified, grounded numbers)", ok,
          f"status={summ.get('status')} verified={getattr(ver,'passed',None)} "
          f"confidence={getattr(rec,'confidence',None)}")
    return ok


def main() -> int:
    print("=" * 72)
    print("Apex Strategy Agent - test suite")
    print("=" * 72)
    test_reproducibility()
    test_hand_math_no_change()
    test_hand_math_price_change()
    test_scenarios()
    test_insufficient_input()
    test_schema_enforcement()
    test_live_end_to_end()

    n = len(results)
    passed = sum(1 for _, p, _ in results if p)
    repro = next((p for nm, p, _ in results if nm.startswith("1.")), False)
    math_ok = all(p for nm, p, _ in results if nm.startswith("2"))
    insuff = next((p for nm, p, _ in results if nm.startswith("4.")), False)
    live = next(((nm, p) for nm, p, _ in results if nm.startswith("6.")), None)

    print("\n" + "=" * 72)
    print("ACCURACY REPORT")
    print("=" * 72)
    print(f"Tests passed                         : {passed}/{n}")
    print(f"Simulation reproducibility           : {'deterministic - PASS' if repro else 'FAIL'}")
    print(f"Math correctness vs hand calculation : {'PASS' if math_ok else 'FAIL'}")
    print(f"Insufficient input -> asks for data  : {'PASS' if insuff else 'FAIL'}")
    print(f"Live end-to-end (real API)           : "
          f"{'PASS' if live and live[1] else ('SKIPPED' if not config.api_key_present() else 'FAIL')}")
    print("=" * 72)

    all_passed = all(p for _, p, _ in results)
    print(f"\nRESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
