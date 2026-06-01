"""
sources.py — Input loaders and the connector interface for real data sources.

Today we load business metrics + market context from local JSON or CSV. In
production you would implement `DataConnector` subclasses that pull the same
fields from real systems (billing/ERP for metrics, a market/news API for
context). Nothing downstream changes — the nodes consume the dicts returned
here regardless of where they came from.
"""
from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Connector interface (production swap point)
# ---------------------------------------------------------------------------
class DataConnector(ABC):
    """Implement one of these per real source. `fetch()` must return a dict with
    the same shape as the local JSON inputs (see sample_inputs())."""

    name: str = "abstract"

    @abstractmethod
    def fetch(self) -> dict[str, Any]:
        raise NotImplementedError


class LocalJSONConnector(DataConnector):
    """Reads a single JSON file containing business_metrics, market_context, and
    a scenario string."""
    name = "local_json"

    def __init__(self, path: str):
        self.path = path

    def fetch(self) -> dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as fh:
            return json.load(fh)


class LocalCSVConnector(DataConnector):
    """Reads business metrics from a one-row CSV (header = field names). Market
    context and scenario are passed in separately. Demonstrates that the metric
    fields can come from a flat export."""
    name = "local_csv"

    def __init__(self, metrics_csv_path: str, scenario: str = "",
                 market_context: dict | None = None):
        self.metrics_csv_path = metrics_csv_path
        self.scenario = scenario
        self.market_context = market_context or {}

    def fetch(self) -> dict[str, Any]:
        with open(self.metrics_csv_path, newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        metrics = {k: _coerce(v) for k, v in row.items()}
        return {"business_metrics": metrics, "market_context": self.market_context,
                "scenario": self.scenario}


# ---------------------------------------------------------------------------
# Example production stub (documented, not wired)
# ---------------------------------------------------------------------------
class MarketNewsAPIConnector(DataConnector):
    """STUB. In production, fetch competitor/market signals from a real news or
    market-data API here and map them into the `market_context` shape. Left
    unimplemented on purpose so it cannot silently return fake data."""
    name = "market_news_api"

    def __init__(self, api_base: str, api_key_env: str = "MARKET_API_KEY"):
        self.api_base = api_base
        self.api_key_env = api_key_env

    def fetch(self) -> dict[str, Any]:
        raise NotImplementedError(
            "MarketNewsAPIConnector is a documented integration point, not yet "
            "implemented. Provide market_context via local inputs for now.")


def _coerce(v: str) -> Any:
    if v is None or v == "":
        return None
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def load_inputs(path: str) -> dict[str, Any]:
    """Convenience loader: JSON by extension, else error."""
    if path.lower().endswith(".json"):
        return LocalJSONConnector(path).fetch()
    if path.lower().endswith(".csv"):
        return LocalCSVConnector(path).fetch()
    raise ValueError(f"Unsupported input file type: {path} (use .json or .csv)")


# ---------------------------------------------------------------------------
# Sample inputs (for the demo + tests)
# ---------------------------------------------------------------------------
def sample_inputs() -> dict[str, Any]:
    """A realistic mid-market SaaS company plus a sample scenario."""
    return {
        "business_metrics": {
            "company_name": "Northwind SaaS",
            "currency": "USD",
            "customers": 12000,
            "arpu_monthly": 90.0,
            "monthly_churn_rate": 0.025,
            "monthly_gross_adds": 450,
            "variable_cost_per_customer": 18.0,
            "fixed_costs_monthly": 220000.0,
            "support_cost_monthly": 160000.0,
            "marketing_spend_monthly": 140000.0,
            "headcount": 130,
            "cost_per_head_monthly": 9500.0,
            "revenue_history": [980000, 995000, 1010000, 1030000, 1055000, 1080000],
        },
        "market_context": {
            "competitors": ["Gale Systems", "Borealis Cloud", "Tradewind Apps"],
            "market_growth_rate_annual": 0.11,
            "notes": ("Category growing low-double-digits. Two competitors recently "
                      "raised prices ~6-9%; one launched a cheaper self-serve tier. "
                      "Customers are moderately price-sensitive; switching costs are "
                      "moderate due to integrations."),
        },
        "scenario": ("Raise prices by 8% across the base, cut support costs by 15% "
                     "through automation, and hold marketing flat. Project the impact "
                     "over 18 months."),
        "horizon_months": 18,
    }


def insufficient_inputs() -> dict[str, Any]:
    """Inputs missing several required metric fields -> should trigger a request
    for more data rather than a fabricated projection."""
    return {
        "business_metrics": {
            "company_name": "MysteryCo",
            "customers": 5000,
            "arpu_monthly": 50.0,
            # churn, adds, and the entire cost structure are missing
        },
        "market_context": {"notes": "Unknown."},
        "scenario": "Should we raise prices 10%?",
        "horizon_months": 12,
    }
