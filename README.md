# Apex Strategy Agent

A corporate **strategy & simulation** agent. It analyzes a business and its
market, runs **deterministic** quantitative strategic simulations ("what if we
raise prices 8% and cut support 15%?"), generates best/base/worst scenarios, and
produces a recommendation with its **explicit assumptions, confidence, and key
sensitivities** — so leaders see the reasoning, not a black box.

Built with LangGraph + Pydantic. It calls the **real Anthropic API** (no offline
or mock mode). This is **decision support**, not an autonomous actor.

> This is a simplified model, not a guarantee of real-world outcomes. Real
> markets are nonlinear, lagged, and competitive. Treat every number as
> conditional on the documented assumptions and apply human judgment.

---

## How it works (anti-hallucination design)

The core principle: **the LLM never produces an authoritative number.**

| Concern | Who does it |
|--------|-------------|
| Project revenue / churn / margin / customers / headcount over time | **Deterministic Python** ([`financial_model.py`](financial_model.py)) |
| Interpret a plain-English scenario into numeric levers | LLM (Sonnet) — levers only, no math |
| Narrate trends, strengths/weaknesses, market context | LLM, grounded in pre-computed stats |
| Write the recommendation text, assumptions, sensitivities, risks, confidence | LLM (text fields only) |
| The recommendation's actual figures | **Filled from the model**, not the LLM |
| Recompute and confirm the numbers weren't altered | `verify` node (deterministic) |

Every economic assumption (price elasticity, churn sensitivity, acquisition
sensitivity, cost behavior) is an explicit constant in `financial_model.py` and
is echoed into the output (`assumptions_used`) so an executive can challenge it.
If required inputs are missing, the agent **asks for them instead of
simulating**.

This reduces fabricated numbers; it does not eliminate model risk. The LLM's
*narrative* can still be wrong or over-confident — which is why the numbers are
deterministic, the assumptions are visible, and a human decides.

---

## Architecture

```
START
  └─ ingest ──(insufficient inputs)──────────────────────────► error ─► END
        │ ok
        ▼
   analyze_internal ─► analyze_market ─► simulate ─► generate_scenarios
        ─► recommend ─► verify ─► END

   any node raises an exception ───────────────────────────────► error ─► END
```

| Node | Does | Model |
|------|------|-------|
| `ingest` | Load + validate inputs; detect insufficient data | — |
| `analyze_internal` | Deterministic trend stats + narrative | Sonnet |
| `analyze_market` | Classify competitive pressure; threats/opportunities | Haiku |
| `simulate` | LLM maps scenario→levers; **deterministic** projection | Sonnet (levers only) |
| `generate_scenarios` | best/base/worst by flexing documented assumption bands | — |
| `recommend` | Recommendation text + confidence + assumptions/risks | Sonnet |
| `verify` | Re-runs the math; confirms numbers came from the model | — |
| `error` | Graceful handler / insufficient-input request | — |

- Conditional edges; `MemorySaver` checkpointer keyed by `thread_id`.
- Models are configurable constants at the top of [`config.py`](config.py)
  (`ANALYSIS_MODEL`, `CLASSIFIER_MODEL`) — never hardcoded deeper.

---

## Setup

Python 3.10+ (developed on 3.14).

```bash
python -m pip install -r requirements.txt
```

**Set your own API key** (this tool reads it from the environment only — there
is no fallback, and the key is never written to any file):

```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```
```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

If the key is missing, the tool fails with a clear message telling you to set it.

---

## How to run

```bash
python main.py                       # built-in sample company + scenario
python main.py my_inputs.json        # your own inputs
python main.py --write-sample s.json # write an editable sample inputs file
```

Run the test suite + accuracy report:

```bash
python test_runner.py
```

Compile-check the graph only (no key needed):

```bash
python -c "from graph import build_graph; build_graph(); print('graph OK')"
```

### Interactive web UI

A browser dashboard: edit the business inputs + scenario, run the pipeline, and
watch each node complete live, then see the deterministic projection (with a
revenue/margin chart), best/base/worst scenarios, the recommendation
(assumptions + confidence + sensitivities + risks), the verification badges, and
the audit trail.

```bash
# set YOUR key in the environment first (it is read from os.environ only and is
# never written to any file):
#   PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
#   bash:        export ANTHROPIC_API_KEY=sk-ant-...
python -m pip install -r requirements.txt   # includes fastapi + uvicorn
python webapp.py
# open http://127.0.0.1:8001
```

Files: [`webapp.py`](webapp.py) (FastAPI + SSE progress streaming) and
[`web/index.html`](web/index.html) (dashboard). The server reads the key from
the environment at request time; if it is missing, a run returns a clear error.
The dev server has no authentication — don't expose it to the public internet
as-is.

### Input format (JSON)

```json
{
  "business_metrics": {
    "company_name": "Northwind SaaS", "currency": "USD",
    "customers": 12000, "arpu_monthly": 90.0, "monthly_churn_rate": 0.025,
    "monthly_gross_adds": 450, "variable_cost_per_customer": 18.0,
    "fixed_costs_monthly": 220000.0, "support_cost_monthly": 160000.0,
    "marketing_spend_monthly": 140000.0, "headcount": 130,
    "cost_per_head_monthly": 9500.0,
    "revenue_history": [980000, 995000, 1010000, 1030000, 1055000, 1080000]
  },
  "market_context": { "competitors": ["..."], "market_growth_rate_annual": 0.11, "notes": "..." },
  "scenario": "Raise prices 8%, cut support costs 15%, hold marketing flat. 18 months.",
  "horizon_months": 18
}
```

---

## The financial model — assumptions and how to tune them

All in [`financial_model.py`](financial_model.py), as named constants:

| Constant | Meaning | Default |
|----------|---------|---------|
| `CHURN_PRICE_SENSITIVITY` | How much a price change moves churn: `churn = churn₀·(1 + k·price_frac)` | `0.60` |
| `ACQ_PRICE_SENSITIVITY` | How much a price change moves new-customer acquisition (negative) | `-0.80` |
| `ACQ_MARKETING_SENSITIVITY` | How much a marketing-spend change moves acquisition | `0.50` |
| `CHURN_FLOOR/CEILING` | Clamps so a runaway lever can't produce nonsense | `0.0 / 0.95` |
| `SCENARIO_BANDS` | How best/worst flex the demand-sensitivity assumptions | ±50% |

The monthly dynamics are documented in full at the top of `financial_model.py`.
To recalibrate to your business, change these constants (ideally fit them to
historical price-change experiments). Tighter bands = narrower scenario spread.

**Effective price elasticity** implied by these levers sits roughly in the −0.8
to −1.4 region. It is documented and chosen, not learned and not produced by the
LLM.

---

## Where real data connectors plug in

[`sources.py`](sources.py) defines a `DataConnector` interface. Today:
`LocalJSONConnector` and `LocalCSVConnector` read local files. For production,
implement a connector that returns the same dict shape:

- **Internal metrics** → billing/ERP/data-warehouse connector populating
  `business_metrics`.
- **Market/competitor context** → a news/market-data API. A documented stub
  `MarketNewsAPIConnector` marks the exact integration point; it raises
  `NotImplementedError` on purpose so it can never silently return fake data.

Nothing downstream changes — the nodes consume the dict regardless of source.

---

## Governance & audit

- **Append-only audit log**: every step records step, status, model used, model
  parameters used, inputs, outputs, timestamp — in memory (graph state) and as
  JSONL at `audit_log.jsonl` (configurable via `APEX_AUDIT_LOG`).
- **Decision support, not autonomy**: every recommendation carries the
  `decision_support_notice`, its assumptions, confidence, and sensitivities, and
  is presented for human judgment. The agent never "acts."

## Reliability

- Retry with exponential backoff on every LLM call (max 3, `tenacity`).
- Robust JSON parsing: strip fences → regex-extract the outermost `{...}` → raise
  with the real error on total failure (never fabricate).
- Pydantic validation on every node output. Any exception routes to the `error`
  node; the pipeline degrades gracefully instead of crashing.

## Testing

`python test_runner.py` runs (no key needed for 1–5):

1. **Reproducibility** — identical inputs produce byte-identical projections.
2. **Hand-verified math** — model output equals numbers computed by hand
   (e.g. +10% price → churn 0.053, customers 1020.6, revenue 112,266, margin
   64,060), proving figures come from the model, not an LLM.
3. **Scenario set** — best/base/worst generated and ordered worst ≤ base ≤ best.
4. **Insufficient input** — the real graph asks for the missing fields instead
   of simulating.
5. **Schema enforcement** — a recommendation cannot omit assumptions /
   sensitivities / risks.
6. **Live end-to-end** (only when `ANTHROPIC_API_KEY` is set; otherwise
   **SKIPPED**, never failed) — full pipeline returns `status: ok`, verification
   passes, and the recommendation's figures match a fresh deterministic run.

The report prints reproducibility, math-correctness, and insufficient-input
handling.

---

## Limitations (read this)

- The model is **linear and simplified**: no lagged effects, no competitor
  reaction, no seasonality, no segment mix, constant elasticity. Real outcomes
  will differ.
- Sensitivity constants are **assumptions**, not measured truths for your
  business — tune them and treat scenario spread as illustrative.
- The LLM narrative and confidence can be wrong; numbers are deterministic but
  the *interpretation* is not guaranteed.
- No persistence beyond the process (in-memory checkpointer; local audit JSONL).
- This is not financial advice and makes no claim of being "perfect" or
  hallucination-free.

---

## Files

| File | Purpose |
|------|---------|
| [`config.py`](config.py) | Models, retry, paths, API-key loading (env only) |
| [`state.py`](state.py) | Pydantic models + LangGraph state schema |
| [`financial_model.py`](financial_model.py) | The deterministic simulation engine + assumptions |
| [`sources.py`](sources.py) | Input loaders + `DataConnector` interface (real-source swap point) |
| [`nodes.py`](nodes.py) | LLM wrapper, robust JSON, retry, audit, the 8 nodes |
| [`graph.py`](graph.py) | Graph assembly, conditional edges, checkpointer |
| [`main.py`](main.py) | CLI runner + readable report |
| [`test_runner.py`](test_runner.py) | Test suite + accuracy report |
