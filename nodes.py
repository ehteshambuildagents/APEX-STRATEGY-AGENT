"""
nodes.py — LangGraph nodes, the Anthropic LLM wrapper, robust JSON parsing,
retry/backoff, and the append-only audit helper.

Division of labor (anti-hallucination core):
  - The LLM analyzes qualitatively, translates the scenario text into numeric
    LEVERS (SimulationParams), and explains results in plain English.
  - The deterministic financial_model computes every projected number.
  - Authoritative numeric fields on the Recommendation are populated from the
    model, never from LLM text. `verify` recomputes and confirms.
"""
from __future__ import annotations

import functools
import json
import re
from typing import Any, Callable

import anthropic
import tenacity

import config
import financial_model as fm
from state import (AuditEntry, BusinessMetrics, CompetitivePressure,
                   InternalAnalysis, MarketAnalysis, MarketContext,
                   Recommendation, SimulationParams, StrategyState, VerifyResult)

log = config.get_logger("nodes")

_CLIENT: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """Lazy Anthropic client. Reads the key from the environment only; raises a
    clear error if it is missing. Importing this module / compiling the graph
    does NOT require a key — only running an LLM node does."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=config.require_api_key(),
                                      timeout=config.LLM_REQUEST_TIMEOUT_SECONDS)
    return _CLIENT


# ===========================================================================
# Robust JSON parsing
# ===========================================================================
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_outermost_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def parse_json(text: str) -> dict:
    """Strip markdown fences, then fall back to regex-extracting the outermost
    {...} block. Surface the real error on total failure — never fabricate."""
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first:
        block = _extract_outermost_object(cleaned)
        if block is not None:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                pass
        snippet = text[:300].replace("\n", "\\n")
        raise ValueError(f"Could not parse JSON from model output ({first}). "
                         f"Output started with: {snippet!r}") from first


# ===========================================================================
# LLM call with retry/backoff
# ===========================================================================
# Only retry genuinely transient API failures (network, rate limit, 5xx,
# timeout). A truncated response or a bad request is not retried.
_RETRYABLE = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def call_llm_json(system: str, user: str, model: str, max_tokens: int) -> tuple[dict, dict]:
    client = get_client()

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(config.LLM_MAX_RETRIES),
        wait=tenacity.wait_exponential(multiplier=config.LLM_BACKOFF_BASE_SECONDS,
                                       max=config.LLM_BACKOFF_MAX_SECONDS),
        retry=tenacity.retry_if_exception_type(_RETRYABLE),
        reraise=True,
    )
    def _invoke() -> tuple[str, str | None]:
        msg = client.messages.create(model=model, max_tokens=max_tokens,
                                     system=system, messages=[{"role": "user", "content": user}])
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts), getattr(msg, "stop_reason", None)

    raw, stop_reason = _invoke()
    if stop_reason == "max_tokens":
        # Surface the real cause instead of a confusing JSON parse error.
        raise ValueError(
            f"Model response was truncated at max_tokens={max_tokens}; the JSON is "
            f"incomplete. Increase config.MAX_TOKENS for this node.")
    data = parse_json(raw)
    meta = {"model_used": model, "max_tokens": max_tokens}
    return data, meta


# ===========================================================================
# Audit
# ===========================================================================
def record_audit(step: str, status: str, inputs: str = "", outputs: str = "",
                 model_used: str | None = None, model_params: dict | None = None) -> AuditEntry:
    entry = AuditEntry(step=step, status=status, inputs=inputs[:2000],
                       outputs=outputs[:2000], model_used=model_used,
                       model_params=model_params or {})
    try:
        with open(config.AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
    except OSError as exc:
        log.warning("could not append to audit file: %s", exc)
    return entry


# ===========================================================================
# Node decorator: logging + graceful error capture
# ===========================================================================
def node(name: str) -> Callable:
    def deco(fn: Callable[[StrategyState], dict]) -> Callable[[StrategyState], dict]:
        @functools.wraps(fn)
        def wrapper(state: StrategyState) -> dict:
            log.info("[%s] start", name)
            try:
                update = fn(state) or {}
                log.info("[%s] ok", name)
                return update
            except Exception as exc:
                log.exception("[%s] failed: %s", name, exc)
                entry = record_audit(name, "error", outputs=f"{type(exc).__name__}: {exc}")
                return {"error": {"node": name, "type": type(exc).__name__,
                                  "message": str(exc)}, "audit_log": [entry]}
        return wrapper
    return deco


def _confidence_label(c: float) -> str:
    if c >= 0.75:
        return "high"
    if c >= 0.50:
        return "moderate"
    if c >= 0.25:
        return "low"
    return "very low"


# ===========================================================================
# Nodes
# ===========================================================================
@node("ingest")
def ingest(state: StrategyState) -> dict:
    raw = state.get("raw_inputs", {})
    bm = raw.get("business_metrics", {}) or {}
    scenario = (raw.get("scenario") or state.get("scenario_text") or "").strip()
    missing = fm.missing_required_fields(bm)

    if missing or not scenario:
        info = {
            "missing_fields": missing,
            "missing_scenario": not bool(scenario),
            "message": _missing_message(missing, not bool(scenario)),
        }
        entry = record_audit("ingest", "insufficient_input",
                             inputs=f"provided fields={sorted(bm.keys())}",
                             outputs=info["message"])
        return {"insufficient": info, "audit_log": [entry]}

    metrics = BusinessMetrics(**bm)            # Pydantic validation
    market = MarketContext(**(raw.get("market_context", {}) or {}))
    horizon = int(raw.get("horizon_months", config.DEFAULT_HORIZON_MONTHS))
    entry = record_audit("ingest", "ok",
                         inputs=f"{metrics.company_name}, {metrics.customers} customers",
                         outputs=f"scenario chars={len(scenario)}, horizon={horizon}")
    return {"metrics": metrics, "market": market, "scenario_text": scenario,
            "horizon_months": horizon, "audit_log": [entry]}


@node("analyze_internal")
def analyze_internal(state: StrategyState) -> dict:
    metrics: BusinessMetrics = state["metrics"]
    trends = fm.compute_trends(metrics)
    system = ("You are a corporate FP&A analyst. You are given DETERMINISTIC, "
              "pre-computed metrics and trend statistics. Do not invent or "
              "recompute numbers; interpret them. Respond with one JSON object.")
    user = (f"Company: {metrics.company_name}\n"
            f"Computed trend statistics (authoritative): {json.dumps(trends)}\n"
            f"Cost structure (monthly): fixed={metrics.fixed_costs_monthly}, "
            f"support={metrics.support_cost_monthly}, marketing={metrics.marketing_spend_monthly}, "
            f"variable/customer={metrics.variable_cost_per_customer}, "
            f"payroll={metrics.headcount}x{metrics.cost_per_head_monthly}\n\n"
            'Return JSON: {"trends_summary": "...", "strengths": ["..."], '
            '"weaknesses": ["..."]}')
    data, meta = call_llm_json(system, user, config.ANALYSIS_MODEL,
                               config.MAX_TOKENS["analyze_internal"])
    analysis = InternalAnalysis(
        trends_summary=data["trends_summary"], strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []), computed_trends=trends,
        model_used=meta["model_used"])
    entry = record_audit("analyze_internal", "ok", inputs=json.dumps(trends),
                         outputs=f"{len(analysis.strengths)} strengths / "
                                 f"{len(analysis.weaknesses)} weaknesses",
                         model_used=meta["model_used"])
    return {"internal_analysis": analysis, "audit_log": [entry]}


@node("analyze_market")
def analyze_market(state: StrategyState) -> dict:
    market: MarketContext = state["market"]
    system = ("You classify competitive pressure and summarize market context "
              "from the provided notes ONLY. Do not invent facts. One JSON object.")
    user = (f"Competitors: {market.competitors}\n"
            f"Market growth (annual): {market.market_growth_rate_annual}\n"
            f"Notes: {market.notes}\n\n"
            'Return JSON: {"competitive_pressure": "low|medium|high", '
            '"threats": ["..."], "opportunities": ["..."], "summary": "..."}')
    data, meta = call_llm_json(system, user, config.CLASSIFIER_MODEL,
                               config.MAX_TOKENS["analyze_market"])
    analysis = MarketAnalysis(
        competitive_pressure=CompetitivePressure(str(data["competitive_pressure"]).lower()),
        threats=data.get("threats", []), opportunities=data.get("opportunities", []),
        summary=data.get("summary", ""), model_used=meta["model_used"])
    entry = record_audit("analyze_market", "ok", inputs=f"competitors={len(market.competitors)}",
                         outputs=f"pressure={analysis.competitive_pressure.value}",
                         model_used=meta["model_used"])
    return {"market_analysis": analysis, "audit_log": [entry]}


@node("simulate")
def simulate(state: StrategyState) -> dict:
    metrics: BusinessMetrics = state["metrics"]
    scenario = state["scenario_text"]
    horizon = state.get("horizon_months", config.DEFAULT_HORIZON_MONTHS)

    system = ("You translate an executive's plain-English strategic scenario into "
              "numeric MODEL LEVERS. You do NOT compute any financial outcome — a "
              "separate deterministic model does that. Extract only the levers. "
              "Percentages are signed: a price cut is negative. One JSON object.")
    user = (f"Scenario: {scenario}\n"
            f"Default horizon if unspecified: {horizon} months.\n\n"
            'Return JSON with ONLY these keys: {"price_change_pct": 0.0, '
            '"support_cost_change_pct": 0.0, "marketing_change_pct": 0.0, '
            '"headcount_change": 0, "horizon_months": 18, "rationale": '
            '"how you mapped the text to these levers"}')
    data, meta = call_llm_json(system, user, config.ANALYSIS_MODEL,
                               config.MAX_TOKENS["simulate_params"])
    data.setdefault("horizon_months", horizon)
    if not data.get("horizon_months"):
        data["horizon_months"] = horizon
    params = SimulationParams(**{k: data[k] for k in data
                                 if k in SimulationParams.model_fields})
    projection = fm.simulate(metrics, params)   # DETERMINISTIC numbers
    entry = record_audit("simulate", "ok",
                         inputs=f"scenario chars={len(scenario)}",
                         outputs=(f"levers price={params.price_change_pct}%, "
                                  f"support={params.support_cost_change_pct}%, "
                                  f"final_margin={projection.summary['final_monthly_margin']}"),
                         model_used=meta["model_used"],
                         model_params=params.model_dump())
    return {"projection": projection, "audit_log": [entry]}


@node("generate_scenarios")
def generate_scenarios(state: StrategyState) -> dict:
    metrics: BusinessMetrics = state["metrics"]
    params = state["projection"].params
    scenarios = fm.generate_scenarios(metrics, params)   # DETERMINISTIC
    by = {c.label: c.summary["final_monthly_margin"] for c in scenarios.cases}
    entry = record_audit("generate_scenarios", "ok",
                         outputs=f"best={by.get('best')}, base={by.get('base')}, "
                                 f"worst={by.get('worst')}",
                         model_params=params.model_dump())
    return {"scenarios": scenarios, "audit_log": [entry]}


@node("recommend")
def recommend(state: StrategyState) -> dict:
    metrics: BusinessMetrics = state["metrics"]
    proj = state["projection"]
    scenarios = state["scenarios"]
    ia: InternalAnalysis = state["internal_analysis"]
    ma: MarketAnalysis = state["market_analysis"]

    case = {c.label: c for c in scenarios.cases}
    # Authoritative numbers, assembled from the deterministic model only.
    projected_outcomes = {
        "currency": metrics.currency,
        "horizon_months": proj.params.horizon_months,
        "levers": proj.params.model_dump(),
        "baseline_current": proj.baseline,
        "base_case": case["base"].summary,
        "best_case": case["best"].summary,
        "worst_case": case["worst"].summary,
        "assumptions_used": proj.assumptions_used,
    }

    system = ("You are a strategy advisor producing DECISION SUPPORT for an "
              "executive. The numbers below were produced by a deterministic "
              "financial model. CRITICAL: use ONLY numbers that appear in the "
              "provided projection. Do NOT compute, estimate, extrapolate, or "
              "introduce any new figure of your own (no invented customer counts, "
              "dollar amounts, or percentages). Discuss sensitivities QUALITATIVELY "
              "or by referring to the best/base/worst cases already given — never "
              "by inventing a new number. Make assumptions and sensitivities "
              "explicit. State confidence honestly; do not claim certainty. "
              "Be concise: keep 'rationale' under ~150 words and give at most 4 "
              "short items in each list. Respond with one JSON object only.")
    user = (f"Internal analysis: {ia.trends_summary}\n"
            f"Strengths: {ia.strengths}\nWeaknesses: {ia.weaknesses}\n"
            f"Market: pressure={ma.competitive_pressure.value}; {ma.summary}\n"
            f"Threats: {ma.threats}\nOpportunities: {ma.opportunities}\n"
            f"Deterministic projection (authoritative): {json.dumps(projected_outcomes, default=str)}\n\n"
            'Return JSON: {"headline": "...", "rationale": "...", '
            '"confidence": 0.0, "assumptions": ["..."], "sensitivities": ["..."], '
            '"key_risks": ["..."]}')
    data, meta = call_llm_json(system, user, config.ANALYSIS_MODEL,
                               config.MAX_TOKENS["recommend"])
    conf = float(data.get("confidence", 0.5))
    conf = max(0.0, min(1.0, conf))
    rec = Recommendation(
        headline=data["headline"], rationale=data["rationale"],
        confidence=conf, confidence_label=_confidence_label(conf),
        assumptions=data.get("assumptions") or ["(model assumptions, see projected_outcomes.assumptions_used)"],
        sensitivities=data.get("sensitivities") or ["Outcome depends on the price-elasticity assumptions listed."],
        key_risks=data.get("key_risks") or ["Real-market dynamics may differ from this simplified model."],
        projected_outcomes=projected_outcomes, model_used=meta["model_used"])
    entry = record_audit("recommend", "ok",
                         outputs=f"confidence={conf:.2f} ({rec.confidence_label})",
                         model_used=meta["model_used"])
    return {"recommendation": rec, "audit_log": [entry]}


@node("verify")
def verify(state: StrategyState) -> dict:
    metrics: BusinessMetrics = state["metrics"]
    proj = state["projection"]
    scenarios = state["scenarios"]
    rec: Recommendation = state["recommendation"]
    checks: dict[str, bool] = {}
    notes: list[str] = []

    # 1) Reproducibility: re-run the deterministic model and compare summaries.
    rerun = fm.simulate(metrics, proj.params)
    checks["reproducible"] = (rerun.summary == proj.summary)
    if not checks["reproducible"]:
        notes.append("Re-running the model did not reproduce identical numbers.")

    # 2) Numbers come from the model: the recommendation's base-case figures must
    #    equal a fresh deterministic base run (proves the LLM didn't alter them).
    fresh_base = fm.generate_scenarios(metrics, proj.params)
    base_summary = next(c.summary for c in fresh_base.cases if c.label == "base")
    checks["numbers_from_model"] = (rec.projected_outcomes.get("base_case") == base_summary)
    if not checks["numbers_from_model"]:
        notes.append("Recommendation figures do not match the deterministic model.")

    # 3) Scenario ordering sanity: worst <= base <= best on final monthly margin.
    by = {c.label: c.summary["final_monthly_margin"] for c in scenarios.cases}
    checks["scenario_ordering"] = (by["worst"] <= by["base"] <= by["best"])
    if not checks["scenario_ordering"]:
        notes.append(f"Scenario margins not ordered worst<=base<=best: {by}")

    # 4) Recommendation completeness.
    checks["has_assumptions"] = len(rec.assumptions) > 0
    checks["has_sensitivities"] = len(rec.sensitivities) > 0
    checks["has_risks"] = len(rec.key_risks) > 0
    checks["confidence_in_range"] = 0.0 <= rec.confidence <= 1.0

    passed = all(checks.values())
    result = VerifyResult(passed=passed, checks=checks, notes=notes)
    entry = record_audit("verify", "ok" if passed else "failed_checks",
                         outputs=json.dumps(checks))
    summary = build_summary(state, verification=result)
    return {"verification": result, "summary": summary, "audit_log": [entry]}


@node("error")
def error(state: StrategyState) -> dict:
    if state.get("insufficient"):
        info = state["insufficient"]
        entry = record_audit("error", "insufficient_input", outputs=info["message"])
        summary = {"status": "insufficient_input", **info,
                   "audit_entries": len(state.get("audit_log", [])) + 1}
        return {"summary": summary, "audit_log": [entry]}
    err = state.get("error") or {"message": "unknown error"}
    entry = record_audit("error", "handled",
                         outputs=f"{err.get('node')}: {err.get('message')}")
    return {"summary": {"status": "error", "error": err,
                        "audit_entries": len(state.get("audit_log", [])) + 1},
            "audit_log": [entry]}


# ===========================================================================
# Helpers
# ===========================================================================
def _missing_message(missing: list[str], missing_scenario: bool) -> str:
    parts = ["Insufficient inputs to run a responsible simulation."]
    if missing:
        parts.append("Missing required business metrics: " + ", ".join(missing) + ".")
    if missing_scenario:
        parts.append("No scenario was provided (e.g. 'raise prices 8%, cut support 15%').")
    parts.append("Please provide the missing inputs; the agent will not fabricate "
                 "a projection from incomplete data.")
    return " ".join(parts)


def build_summary(state: StrategyState, verification: VerifyResult | None = None) -> dict:
    if state.get("insufficient"):
        return {"status": "insufficient_input", **state["insufficient"]}
    if state.get("error"):
        return {"status": "error", "error": state["error"]}
    proj = state.get("projection")
    rec = state.get("recommendation")
    out: dict[str, Any] = {"status": "ok"}
    if proj:
        out["levers"] = proj.params.model_dump()
        out["base_case_summary"] = proj.summary
    if state.get("scenarios"):
        out["scenarios"] = {c.label: c.summary for c in state["scenarios"].cases}
    if rec:
        out["recommendation_headline"] = rec.headline
        out["confidence"] = rec.confidence
        out["confidence_label"] = rec.confidence_label
    if verification:
        out["verification_passed"] = verification.passed
        out["verification_checks"] = verification.checks
    out["audit_entries"] = len(state.get("audit_log", []))
    return out
