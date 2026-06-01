"""
financial_model.py — The deterministic strategic-simulation engine.

ALL projection math lives here. The LLM never computes any financial number; it
only translates a plain-English scenario into the levers in SimulationParams and
explains the results afterwards. Given the same inputs, this module always
produces the same outputs (pure Python, no randomness).

Every economic assumption is an explicit, tunable constant defined below and
echoed into the projection output (`assumptions_used`) so an executive can see
and challenge them.

------------------------------------------------------------------------------
THE MODEL (monthly discrete simulation)
------------------------------------------------------------------------------
For each month t = 1..H, starting from the business's current state:

  price_frac   = price_change_pct / 100
  arpu_t       = arpu_0 * (1 + price_frac)                      # price lever

  # Demand response to price (a simplified elasticity):
  churn_t      = churn_0 * (1 + CHURN_PRICE_SENSITIVITY * price_frac)
  gross_adds_t = adds_0 * (1 + ACQ_PRICE_SENSITIVITY * price_frac)
                        * (1 + ACQ_MARKETING_SENSITIVITY * mkt_frac)

  customers_t  = customers_{t-1} * (1 - churn_t) + gross_adds_t

  revenue_t    = customers_t * arpu_t
  support_t    = support_0 * (1 + support_change_frac)
  variable_t   = customers_t * variable_cost_per_customer
  payroll_t    = (headcount_0 + headcount_change) * cost_per_head
  marketing_t  = marketing_0 * (1 + mkt_frac)
  cost_t       = fixed_0 + support_t + variable_t + payroll_t + marketing_t
  margin_t     = revenue_t - cost_t

These relationships are intentionally simple and linear. See LIMITATIONS in the
README — real markets are nonlinear, lagged, and competitive.
"""
from __future__ import annotations

from state import (BusinessMetrics, MonthlyProjection, ProjectionResult,
                   ScenarioCase, ScenarioSet, SimulationParams)

# ===========================================================================
# TUNABLE ECONOMIC ASSUMPTIONS  (challenge these — they drive every number)
# ===========================================================================
# How strongly a price change moves monthly churn.
#   churn_t = churn_0 * (1 + CHURN_PRICE_SENSITIVITY * price_frac)
#   e.g. +10% price with 0.60 -> churn rises 6% (relative).
CHURN_PRICE_SENSITIVITY = 0.60

# How strongly a price change moves new-customer acquisition (negative: higher
# price -> fewer gross adds).
#   gross_adds_t *= (1 + ACQ_PRICE_SENSITIVITY * price_frac)
ACQ_PRICE_SENSITIVITY = -0.80

# How strongly a marketing-spend change moves acquisition.
ACQ_MARKETING_SENSITIVITY = 0.50

# Hard clamps so a runaway lever cannot produce nonsense.
CHURN_FLOOR, CHURN_CEILING = 0.0, 0.95
ADDS_FLOOR = 0.0

# Effective price elasticity of demand implied by the levers above is roughly
# ACQ_PRICE_SENSITIVITY combined with churn drag (~ -0.8 to -1.4 region). This
# is documented, not learned, and not produced by the LLM.

# Best/worst cases scale the demand-sensitivity assumptions. "best" = customers
# are less price-sensitive than assumed; "worst" = more sensitive.
SCENARIO_BANDS = {
    "best":  {"churn_price_sensitivity_scale": 0.5, "acq_price_sensitivity_scale": 0.5},
    "base":  {"churn_price_sensitivity_scale": 1.0, "acq_price_sensitivity_scale": 1.0},
    "worst": {"churn_price_sensitivity_scale": 1.5, "acq_price_sensitivity_scale": 1.5},
}

# Minimum inputs required to simulate responsibly.
REQUIRED_METRIC_FIELDS = [
    "customers", "arpu_monthly", "monthly_churn_rate", "monthly_gross_adds",
    "variable_cost_per_customer", "fixed_costs_monthly", "support_cost_monthly",
    "marketing_spend_monthly", "headcount", "cost_per_head_monthly",
]


def assumptions_snapshot() -> dict[str, float]:
    return {
        "CHURN_PRICE_SENSITIVITY": CHURN_PRICE_SENSITIVITY,
        "ACQ_PRICE_SENSITIVITY": ACQ_PRICE_SENSITIVITY,
        "ACQ_MARKETING_SENSITIVITY": ACQ_MARKETING_SENSITIVITY,
        "CHURN_FLOOR": CHURN_FLOOR,
        "CHURN_CEILING": CHURN_CEILING,
    }


def _baseline_costs(m: BusinessMetrics) -> dict[str, float]:
    revenue = m.customers * m.arpu_monthly
    variable = m.customers * m.variable_cost_per_customer
    payroll = m.headcount * m.cost_per_head_monthly
    cost = m.fixed_costs_monthly + m.support_cost_monthly + variable + payroll \
        + m.marketing_spend_monthly
    return {
        "customers": float(m.customers),
        "arpu": m.arpu_monthly,
        "revenue": revenue,
        "total_cost": cost,
        "margin": revenue - cost,
        "margin_pct": (revenue - cost) / revenue if revenue else 0.0,
        "headcount": float(m.headcount),
        "churn_rate": m.monthly_churn_rate,
    }


def simulate(metrics: BusinessMetrics, params: SimulationParams,
             churn_price_sensitivity_scale: float = 1.0,
             acq_price_sensitivity_scale: float = 1.0) -> ProjectionResult:
    """Run the deterministic monthly projection. Pure function: same inputs ->
    same outputs. The scale arguments let scenario generation flex the demand
    sensitivities without touching the global constants."""
    price_frac = params.price_change_pct / 100.0
    support_frac = params.support_cost_change_pct / 100.0
    mkt_frac = params.marketing_change_pct / 100.0

    churn_sens = CHURN_PRICE_SENSITIVITY * churn_price_sensitivity_scale
    acq_sens = ACQ_PRICE_SENSITIVITY * acq_price_sensitivity_scale

    arpu = metrics.arpu_monthly * (1 + price_frac)
    churn = metrics.monthly_churn_rate * (1 + churn_sens * price_frac)
    churn = max(CHURN_FLOOR, min(CHURN_CEILING, churn))
    gross_adds = metrics.monthly_gross_adds \
        * (1 + acq_sens * price_frac) \
        * (1 + ACQ_MARKETING_SENSITIVITY * mkt_frac)
    gross_adds = max(ADDS_FLOOR, gross_adds)

    headcount = max(0, metrics.headcount + params.headcount_change)
    payroll = headcount * metrics.cost_per_head_monthly
    support = metrics.support_cost_monthly * (1 + support_frac)
    marketing = metrics.marketing_spend_monthly * (1 + mkt_frac)

    customers = float(metrics.customers)
    months: list[MonthlyProjection] = []
    cumulative_margin = 0.0
    for t in range(1, params.horizon_months + 1):
        customers = customers * (1 - churn) + gross_adds
        revenue = customers * arpu
        variable = customers * metrics.variable_cost_per_customer
        total_cost = metrics.fixed_costs_monthly + support + variable + payroll + marketing
        margin = revenue - total_cost
        cumulative_margin += margin
        months.append(MonthlyProjection(
            month=t, customers=round(customers, 4), arpu=round(arpu, 6),
            revenue=round(revenue, 4), total_cost=round(total_cost, 4),
            margin=round(margin, 4),
            margin_pct=round(margin / revenue, 6) if revenue else 0.0,
            headcount=headcount, churn_rate=round(churn, 6)))

    baseline = _baseline_costs(metrics)
    final = months[-1]
    summary = {
        "final_customers": final.customers,
        "final_monthly_revenue": final.revenue,
        "final_monthly_margin": final.margin,
        "final_margin_pct": final.margin_pct,
        "cumulative_margin": round(cumulative_margin, 4),
        "revenue_change_pct_vs_baseline":
            round((final.revenue - baseline["revenue"]) / baseline["revenue"] * 100, 4)
            if baseline["revenue"] else 0.0,
        "margin_change_vs_baseline": round(final.margin - baseline["margin"], 4),
        "effective_monthly_churn": round(churn, 6),
        "effective_monthly_gross_adds": round(gross_adds, 4),
    }
    return ProjectionResult(
        params=params, baseline=baseline, months=months, summary=summary,
        assumptions_used={
            **assumptions_snapshot(),
            "churn_price_sensitivity_scale": churn_price_sensitivity_scale,
            "acq_price_sensitivity_scale": acq_price_sensitivity_scale,
        })


def generate_scenarios(metrics: BusinessMetrics, params: SimulationParams) -> ScenarioSet:
    """Best / base / worst by flexing the demand-sensitivity assumptions within
    documented bands. All three are deterministic runs of `simulate`."""
    base_proj = simulate(metrics, params)
    cases: list[ScenarioCase] = []
    for label in ("best", "base", "worst"):
        band = SCENARIO_BANDS[label]
        proj = simulate(
            metrics, params,
            churn_price_sensitivity_scale=band["churn_price_sensitivity_scale"],
            acq_price_sensitivity_scale=band["acq_price_sensitivity_scale"])
        cases.append(ScenarioCase(
            label=label,
            description={
                "best": "Customers are less price-sensitive than the base assumption.",
                "base": "Base-case demand sensitivity (nominal assumptions).",
                "worst": "Customers are more price-sensitive than the base assumption.",
            }[label],
            assumption_overrides=band,
            summary=proj.summary))
    return ScenarioSet(base_projection=base_proj, cases=cases)


def compute_trends(metrics: BusinessMetrics) -> dict[str, float]:
    """Deterministic trend stats from revenue history (LLM only narrates these)."""
    hist = metrics.revenue_history
    out: dict[str, float] = {
        "current_monthly_revenue": round(metrics.customers * metrics.arpu_monthly, 4),
        "current_monthly_margin": round(_baseline_costs(metrics)["margin"], 4),
        "current_margin_pct": round(_baseline_costs(metrics)["margin_pct"], 6),
        "monthly_churn_rate": metrics.monthly_churn_rate,
    }
    if len(hist) >= 2:
        first, last = hist[0], hist[-1]
        out["revenue_history_change_pct"] = round((last - first) / first * 100, 4) if first else 0.0
        # average month-over-month growth
        deltas = [(hist[i] - hist[i - 1]) / hist[i - 1] for i in range(1, len(hist)) if hist[i - 1]]
        out["avg_mom_growth_pct"] = round(sum(deltas) / len(deltas) * 100, 4) if deltas else 0.0
        out["history_points"] = float(len(hist))
    return out


def missing_required_fields(raw_metrics: dict) -> list[str]:
    """Return required metric fields that are absent or None (used to decide
    whether we can simulate responsibly)."""
    missing = []
    for f in REQUIRED_METRIC_FIELDS:
        if raw_metrics.get(f) is None:
            missing.append(f)
    return missing
