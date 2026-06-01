"""
graph.py — Assemble and compile the Apex Strategy LangGraph.

Flow:
    ingest -> analyze_internal -> analyze_market -> simulate ->
    generate_scenarios -> recommend -> verify -> END

    insufficient inputs at ingest ----------------------------> error -> END
    any node exception ---------------------------------------> error -> END

MemorySaver checkpointer; every run is keyed by a thread_id. The graph compiles
without an API key — only running the LLM nodes requires ANTHROPIC_API_KEY.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import config
import nodes
from state import StrategyState


def _after_ingest(s: StrategyState) -> str:
    if s.get("error") or s.get("insufficient"):
        return "error"
    return "analyze_internal"


def _gate(next_node: str) -> "function":
    """Router that goes to `error` on failure, else to next_node."""
    def _f(s: StrategyState) -> str:
        return "error" if s.get("error") else next_node
    return _f


def build_graph(checkpointer: MemorySaver | None = None):
    g = StateGraph(StrategyState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("analyze_internal", nodes.analyze_internal)
    g.add_node("analyze_market", nodes.analyze_market)
    g.add_node("simulate", nodes.simulate)
    g.add_node("generate_scenarios", nodes.generate_scenarios)
    g.add_node("recommend", nodes.recommend)
    g.add_node("verify", nodes.verify)
    g.add_node("error", nodes.error)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", _after_ingest,
                            {"analyze_internal": "analyze_internal", "error": "error"})
    g.add_conditional_edges("analyze_internal", _gate("analyze_market"),
                            {"analyze_market": "analyze_market", "error": "error"})
    g.add_conditional_edges("analyze_market", _gate("simulate"),
                            {"simulate": "simulate", "error": "error"})
    g.add_conditional_edges("simulate", _gate("generate_scenarios"),
                            {"generate_scenarios": "generate_scenarios", "error": "error"})
    g.add_conditional_edges("generate_scenarios", _gate("recommend"),
                            {"recommend": "recommend", "error": "error"})
    g.add_conditional_edges("recommend", _gate("verify"),
                            {"verify": "verify", "error": "error"})
    g.add_conditional_edges("verify", _gate(END), {END: END, "error": "error"})
    g.add_edge("error", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def initial_state(raw_inputs: dict, thread_id: str) -> StrategyState:
    return {"raw_inputs": raw_inputs, "thread_id": thread_id, "audit_log": []}


def run(raw_inputs: dict, thread_id: str | None = None, app=None) -> StrategyState:
    app = app or build_graph()
    thread_id = thread_id or "apex-run"
    cfg = {"configurable": {"thread_id": thread_id}}
    return app.invoke(initial_state(raw_inputs, thread_id), config=cfg)


if __name__ == "__main__":
    build_graph()
    print("graph compiled OK")
