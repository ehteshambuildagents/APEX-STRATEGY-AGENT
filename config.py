"""
config.py — Central configuration for the Apex Strategy Agent.

Models, retry policy, paths, and run defaults live here. Financial-model
assumptions (elasticity, churn sensitivity, cost behavior) live in
financial_model.py, since that is the deterministic engine and the place an
analyst tunes the economics.

The Anthropic API key is read ONLY from the environment (os.environ). There is
no offline/mock backend and no hardcoded key. If the key is missing we fail with
a clear message.
"""
from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Models  (configurable constants — never hardcode these deeper in the code)
# ---------------------------------------------------------------------------
# Sonnet: analysis, scenario reasoning, recommendations.
# Haiku: lightweight classification (e.g. competitive-pressure tagging).
ANALYSIS_MODEL = "claude-sonnet-4-6"
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# API key — environment only, no fallback
# ---------------------------------------------------------------------------
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


def require_api_key() -> str:
    """Return the Anthropic API key from the environment, or raise a clear
    error. Never returns a hardcoded/default key, never silently falls back."""
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV_VAR} is not set. Export your Anthropic API key before "
            f"running, e.g.  (PowerShell)  $env:{API_KEY_ENV_VAR}='sk-ant-...'  "
            f"or  (bash)  export {API_KEY_ENV_VAR}=sk-ant-...  "
            f"This tool calls the real Anthropic API and has no offline mode."
        )
    return key


def api_key_present() -> bool:
    return bool(os.environ.get(API_KEY_ENV_VAR, "").strip())


# ---------------------------------------------------------------------------
# Reliability: retry/backoff for LLM calls
# ---------------------------------------------------------------------------
LLM_MAX_RETRIES = 3
LLM_BACKOFF_BASE_SECONDS = 0.6        # exponential: 0.6, 1.2, 2.4 ...
LLM_BACKOFF_MAX_SECONDS = 8.0
LLM_REQUEST_TIMEOUT_SECONDS = 90

# Per-call output ceilings (tokens).
MAX_TOKENS = {
    "analyze_internal": 1500,
    "analyze_market": 1000,
    "simulate_params": 800,
    "recommend": 3500,
}

# ---------------------------------------------------------------------------
# Run defaults
# ---------------------------------------------------------------------------
DEFAULT_HORIZON_MONTHS = 18
AUDIT_LOG_PATH = os.environ.get("APEX_AUDIT_LOG", "audit_log.jsonl")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("APEX_LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    root = logging.getLogger("apex")
    if root.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(h)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"apex.{name}")
