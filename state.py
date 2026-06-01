"""
state.py — Pydantic models and the LangGraph state schema.

Every value that flows between nodes is a validated Pydantic model. Numeric
projection fields are always populated from the deterministic financial model —
never from LLM free-text — which is what lets `verify` prove the numbers were
not invented by the model.
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
class BusinessMetrics(BaseModel):
    """Internal performance snapshot (monthly basis)."""
    company_name: str = "Unnamed Co."
    currency: str = "USD"
    customers: int = Field(gt=0)
    arpu_monthly: float = Field(gt=0, description="Avg revenue per user / month")
    monthly_churn_rate: float = Field(ge=0, le=1)
    monthly_gross_adds: float = Field(ge=0, description="New customers per month")
    variable_cost_per_customer: float = Field(ge=0)
    fixed_costs_monthly: float = Field(ge=0)
    support_cost_monthly: float = Field(ge=0)
    marketing_spend_monthly: float = Field(ge=0)
    headcount: int = Field(ge=0)
    cost_per_head_monthly: float = Field(ge=0)
    # Optional trailing monthly revenue history for trend analysis.
    revenue_history: list[float] = Field(default_factory=list)


class MarketContext(BaseModel):
    competitors: list[str] = Field(default_factory=list)
    market_growth_rate_annual: Optional[float] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
class SimulationParams(BaseModel):
    """Levers the LLM extracts from the scenario text. The LLM chooses these
    qualitative levers; it does NOT compute any financial outcome."""
    price_change_pct: float = 0.0          # +8.0 == raise prices 8%
    support_cost_change_pct: float = 0.0   # -15.0 == cut support cost 15%
    marketing_change_pct: float = 0.0
    headcount_change: int = 0              # absolute change in heads
    horizon_months: int = Field(default=18, ge=1, le=120)
    rationale: str = ""

    @field_validator("price_change_pct", "support_cost_change_pct",
                     "marketing_change_pct")
    @classmethod
    def _sane_pct(cls, v: float) -> float:
        # Guardrail: clamp absurd values rather than projecting nonsense.
        return max(-90.0, min(300.0, v))


class MonthlyProjection(BaseModel):
    month: int
    customers: float
    arpu: float
    revenue: float
    total_cost: float
    margin: float
    margin_pct: float
    headcount: int
    churn_rate: float


class ProjectionResult(BaseModel):
    params: SimulationParams
    baseline: dict[str, float]
    months: list[MonthlyProjection]
    summary: dict[str, float]
    assumptions_used: dict[str, float]


class ScenarioCase(BaseModel):
    label: str                              # "best" | "base" | "worst"
    description: str
    assumption_overrides: dict[str, float]
    summary: dict[str, float]


class ScenarioSet(BaseModel):
    base_projection: ProjectionResult
    cases: list[ScenarioCase]


# ---------------------------------------------------------------------------
# Qualitative analysis (LLM-authored text, grounded in deterministic stats)
# ---------------------------------------------------------------------------
class InternalAnalysis(BaseModel):
    trends_summary: str
    strengths: list[str]
    weaknesses: list[str]
    computed_trends: dict[str, float]       # deterministic, not LLM
    model_used: str


class CompetitivePressure(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketAnalysis(BaseModel):
    competitive_pressure: CompetitivePressure
    threats: list[str]
    opportunities: list[str]
    summary: str
    model_used: str


# ---------------------------------------------------------------------------
# Recommendation (decision support — not an autonomous action)
# ---------------------------------------------------------------------------
DECISION_SUPPORT_NOTICE = (
    "Decision support only. This is a simplified model-based projection, not a "
    "guarantee of real-world outcomes or an autonomous action. Review the "
    "assumptions and apply human judgment before deciding."
)


class Recommendation(BaseModel):
    headline: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: str
    assumptions: list[str] = Field(min_length=1)
    sensitivities: list[str] = Field(min_length=1)
    key_risks: list[str] = Field(min_length=1)
    # Authoritative numbers — set from the deterministic model, not the LLM.
    projected_outcomes: dict[str, Any] = Field(default_factory=dict)
    decision_support_notice: str = DECISION_SUPPORT_NOTICE
    model_used: str


# ---------------------------------------------------------------------------
# Verification + audit
# ---------------------------------------------------------------------------
class VerifyResult(BaseModel):
    passed: bool
    checks: dict[str, bool]
    notes: list[str] = Field(default_factory=list)


class AuditEntry(BaseModel):
    step: str
    status: str
    model_used: Optional[str] = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    inputs: str = ""
    outputs: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class StrategyState(TypedDict, total=False):
    # control / inputs
    thread_id: str
    raw_inputs: dict[str, Any]
    scenario_text: str
    horizon_months: int
    # parsed inputs
    metrics: BusinessMetrics
    market: MarketContext
    # pipeline products
    internal_analysis: InternalAnalysis
    market_analysis: MarketAnalysis
    projection: ProjectionResult
    scenarios: ScenarioSet
    recommendation: Recommendation
    verification: VerifyResult
    # observability
    audit_log: Annotated[list[AuditEntry], operator.add]
    # terminal
    insufficient: Optional[dict]
    error: Optional[dict]
    summary: dict
