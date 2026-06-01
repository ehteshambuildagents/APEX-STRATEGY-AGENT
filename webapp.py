"""
webapp.py — Interactive browser UI for the Apex Strategy Agent.

Serves a single-page dashboard: enter business metrics + a plain-English
scenario, run the pipeline, and watch each node complete in real time, then see
the deterministic projection (with a chart), best/base/worst scenarios, and the
recommendation with explicit assumptions, confidence, and sensitivities.

Run:
    # set YOUR key first (read from the environment only; never written to file)
    #   PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
    #   bash:        export ANTHROPIC_API_KEY=sk-ant-...
    python webapp.py
    # open http://127.0.0.1:8001

NOTE: dev server, no auth. Don't expose it to the public internet as-is.
"""
from __future__ import annotations

import json
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import config
from graph import build_graph, initial_state
from sources import sample_inputs

log = config.get_logger("webapp")
app = FastAPI(title="Apex Strategy Agent")
_GRAPH = build_graph()                 # compiles without a key
_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

NODE_LABELS = {
    "ingest": "Ingest & validate inputs",
    "analyze_internal": "Analyze internal performance",
    "analyze_market": "Analyze market & competitors",
    "simulate": "Run deterministic simulation",
    "generate_scenarios": "Generate best/base/worst",
    "recommend": "Draft recommendation",
    "verify": "Verify numbers",
    "error": "Error handler",
}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_WEB, "index.html"))


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "key_present": config.api_key_present(),
        "analysis_model": config.ANALYSIS_MODEL,
        "classifier_model": config.CLASSIFIER_MODEL,
    })


@app.get("/api/sample")
def sample() -> JSONResponse:
    return JSONResponse(sample_inputs())


@app.get("/api/nodes")
def node_list() -> JSONResponse:
    return JSONResponse({"nodes": [{"id": k, "label": v} for k, v in NODE_LABELS.items()
                                   if k != "error"]})


def _sse(event_type: str, payload: dict) -> str:
    return "data: " + json.dumps({"type": event_type, **payload}, default=str) + "\n\n"


def _dump(model):
    return model.model_dump(mode="json") if model is not None else None


def _serialize(state) -> dict:
    out: dict = {"summary": state.get("summary", {})}
    if state.get("error"):
        out["error"] = state["error"]
    if state.get("insufficient"):
        out["insufficient"] = state["insufficient"]
    out["internal_analysis"] = _dump(state.get("internal_analysis"))
    out["market_analysis"] = _dump(state.get("market_analysis"))
    proj = state.get("projection")
    out["projection"] = _dump(proj)
    sc = state.get("scenarios")
    out["scenarios"] = {"cases": [_dump(c) for c in sc.cases]} if sc else None
    out["recommendation"] = _dump(state.get("recommendation"))
    out["verification"] = _dump(state.get("verification"))
    out["audit"] = [{"step": e.step, "status": e.status, "model_used": e.model_used,
                     "outputs": e.outputs} for e in state.get("audit_log", [])]
    return out


@app.post("/api/run")
async def run(request: Request) -> StreamingResponse:
    inputs = await request.json()

    def gen():
        thread_id = "web-" + uuid.uuid4().hex[:10]
        cfg = {"configurable": {"thread_id": thread_id}}
        init = initial_state(inputs, thread_id)
        try:
            for update in _GRAPH.stream(init, config=cfg, stream_mode="updates"):
                for node_name in update:
                    yield _sse("progress", {"node": node_name,
                                            "label": NODE_LABELS.get(node_name, node_name)})
            final = _GRAPH.get_state(cfg).values
            yield _sse("result", {"state": _serialize(final)})
        except Exception as exc:  # backstop; nodes normally capture their own errors
            log.exception("run failed")
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    config.setup_logging()
    if not config.api_key_present():
        log.warning("ANTHROPIC_API_KEY is not set. The UI will load, but running a "
                    "simulation will return an error until you set it in the "
                    "environment and restart.")
    print("Apex Strategy Agent UI -> http://127.0.0.1:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
