"""
main.py — Run the Apex Strategy Agent end-to-end and print a readable report.

Usage:
    python main.py                 # uses built-in sample inputs
    python main.py inputs.json     # uses your own JSON inputs
    python main.py --write-sample sample_inputs.json   # write a sample to edit

Requires ANTHROPIC_API_KEY in the environment (the LLM nodes call the real API).
"""
from __future__ import annotations

import json
import sys

import config
from graph import build_graph, run
from sources import load_inputs, sample_inputs


def _print_report(state) -> None:
    summ = state.get("summary", {})
    if summ.get("status") == "insufficient_input":
        print("\n=== INSUFFICIENT INPUTS ===")
        print(summ.get("message"))
        return
    if summ.get("status") == "error":
        print("\n=== ERROR ===")
        print(json.dumps(summ.get("error"), indent=2))
        return

    ia = state.get("internal_analysis")
    ma = state.get("market_analysis")
    proj = state.get("projection")
    scenarios = state.get("scenarios")
    rec = state.get("recommendation")
    ver = state.get("verification")

    if ia:
        print("\n=== INTERNAL ANALYSIS ===")
        print(ia.trends_summary)
        print("Strengths :", "; ".join(ia.strengths))
        print("Weaknesses:", "; ".join(ia.weaknesses))
    if ma:
        print("\n=== MARKET ANALYSIS ===")
        print(f"Competitive pressure: {ma.competitive_pressure.value}")
        print(ma.summary)
    if proj:
        p = proj.params
        print("\n=== SIMULATION LEVERS (LLM-mapped from scenario) ===")
        print(f"price {p.price_change_pct:+}%, support {p.support_cost_change_pct:+}%, "
              f"marketing {p.marketing_change_pct:+}%, headcount {p.headcount_change:+}, "
              f"horizon {p.horizon_months}mo")
        print(f"rationale: {p.rationale}")
        print("\n=== DETERMINISTIC PROJECTION (base case) ===")
        b, s = proj.baseline, proj.summary
        print(f"baseline monthly: revenue={b['revenue']:.0f}, margin={b['margin']:.0f}")
        print(f"month {p.horizon_months}: revenue={s['final_monthly_revenue']:.0f}, "
              f"margin={s['final_monthly_margin']:.0f} ({s['final_margin_pct']*100:.1f}%), "
              f"customers={s['final_customers']:.0f}")
        print(f"cumulative margin over horizon: {s['cumulative_margin']:.0f}")
    if scenarios:
        print("\n=== SCENARIOS (best / base / worst — final monthly margin) ===")
        for c in scenarios.cases:
            print(f"  {c.label:<5}: margin={c.summary['final_monthly_margin']:.0f}, "
                  f"revenue={c.summary['final_monthly_revenue']:.0f}  ({c.description})")
    if rec:
        print("\n=== RECOMMENDATION (decision support) ===")
        print(f">> {rec.headline}")
        print(rec.rationale)
        print(f"\nConfidence: {rec.confidence:.2f} ({rec.confidence_label})")
        print("Assumptions:");   [print(f"  - {a}") for a in rec.assumptions]
        print("Sensitivities:"); [print(f"  - {x}") for x in rec.sensitivities]
        print("Key risks:");     [print(f"  - {r}") for r in rec.key_risks]
        print(f"\n{rec.decision_support_notice}")
    if ver:
        print("\n=== VERIFICATION ===")
        print(f"passed={ver.passed}  checks={json.dumps(ver.checks)}")
        for n in ver.notes:
            print(f"  note: {n}")

    print("\n=== AUDIT TRAIL ===")
    for i, e in enumerate(state.get("audit_log", []), 1):
        m = f" | model={e.model_used}" if e.model_used else ""
        print(f"{i:>2}. {e.step:<18} {e.status:<18}{m}")
        if e.outputs:
            print(f"     out: {e.outputs}")

    print("\n=== SUMMARY ===")
    print(json.dumps(summ, indent=2, default=str))


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--write-sample":
        path = args[1] if len(args) > 1 else "sample_inputs.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(sample_inputs(), fh, indent=2)
        print(f"wrote sample inputs to {path}")
        return 0

    config.setup_logging()
    if not config.api_key_present():
        print(config.require_api_key.__doc__)
        # require_api_key raises the precise message:
        try:
            config.require_api_key()
        except RuntimeError as exc:
            print(f"\nERROR: {exc}")
        return 2

    raw = load_inputs(args[0]) if args else sample_inputs()
    app = build_graph()
    state = run(raw, thread_id="apex-cli", app=app)
    _print_report(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
