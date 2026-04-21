# api/main.py
import contextlib
import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from agent.graph import graph
from agent.state import RunMetadata
from observability.tracer import get_tracer
from utils.cost_estimator import estimate_cost

app = FastAPI(title="DeepResearch Agent API", version="1.0.0")


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ─────────────────────────────── Request Models ───────────────────────────────


class ResearchRequest(BaseModel):
    query: str
    profile: str = "fast"


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool
    edited_subquestions: list[str] | None = None


# ─────────────────────────── Tracer Helper ────────────────────────────────────


async def _finalize_run(
    run_id: str,
    thread_config: RunnableConfig,
    start_time: float,
    status: str = "completed",
) -> None:
    """
    Read the final graph state and write the run's closing record to the tracer.
    Swallows all errors — tracing must never break the API response.
    """
    try:
        tracer = get_tracer()
        snapshot = graph.get_state(thread_config)
        if not snapshot:
            await tracer.end_run(run_id, status=status)
            return

        values = snapshot.values
        meta: RunMetadata | None = values.get("run_metadata")
        findings = values.get("findings", [])
        errors = values.get("error_log", [])
        actual_status = "failed" if errors and not values.get("final_report") else status

        await tracer.end_run(
            run_id,
            status=actual_status,
            total_cost_usd=meta.estimated_cost_usd if meta else 0.0,
            total_latency_ms=(time.perf_counter() - start_time) * 1000,
            iteration_count=meta.iteration_count if meta else 0,
            findings_count=len(findings),
        )
    except Exception:
        pass  # intentional — observability is best-effort


# ─────────────────────────── Endpoints ───────────────────────────────────────


@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    """
    Start a new research run and stream SSE events.

    The graph is compiled with interrupt_before=["planner"], so after the
    classifier runs, the stream pauses and emits a hitl_interrupt event.
    The client must call POST /research/approve to resume.
    """
    run_id = str(uuid.uuid4())
    thread_config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    # Register the run in the observability DB immediately
    with contextlib.suppress(Exception):
        await get_tracer().start_run(run_id, request.query, request.profile)

    async def event_generator():
        run_start = time.perf_counter()
        writer_completed = False

        try:
            async for event in graph.astream_events(
                {
                    "query": request.query,
                    "profile": request.profile,
                    "run_id": run_id,
                    "query_difficulty": "",
                    "subquestions": [],
                    "approved_plan": False,
                    "findings": [],
                    "critique": None,
                    "iteration_count": 0,
                    "final_report": None,
                    "run_metadata": RunMetadata(run_id=run_id, profile=request.profile),
                    "error_log": [],
                    "thought_log": [],
                },
                version="v2",
                config=thread_config,
            ):
                kind = event["event"]

                if kind == "on_chain_start":
                    yield sse({"type": "node_start", "node": event["name"]})

                elif kind == "on_tool_start":
                    yield sse(
                        {
                            "type": "tool_call",
                            "tool": event["name"],
                            "input": str(event.get("data", {}).get("input", ""))[:200],
                        }
                    )

                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", [])
                    yield sse(
                        {
                            "type": "tool_result",
                            "tool": event["name"],
                            "count": len(output) if isinstance(output, list) else 1,
                        }
                    )

                elif kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield sse({"type": "token", "content": chunk.content})

                elif kind == "on_chain_end" and event["name"] == "writer":
                    writer_completed = True
                    yield sse({"type": "complete", "run_id": run_id})

        finally:
            # HITL interrupt path: classifier finished, graph paused at planner
            state_snapshot = graph.get_state(thread_config)
            if state_snapshot and state_snapshot.next == ("planner",):
                values = state_snapshot.values
                difficulty = values.get("query_difficulty", "narrow")
                n_questions = {"narrow": 3, "broad": 6, "ambiguous": 4}.get(difficulty, 4)
                estimated_cost = estimate_cost("gpt-4o", n_questions * 800, n_questions * 400)
                yield sse(
                    {
                        "type": "hitl_interrupt",
                        "thread_id": run_id,
                        "query_difficulty": difficulty,
                        "estimated_subquestions": n_questions,
                        "estimated_cost_usd": round(estimated_cost, 4),
                        "message": "Plan ready for approval. POST /research/approve to continue.",
                    }
                )
            elif writer_completed:
                await _finalize_run(run_id, thread_config, run_start)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/research/approve")
async def approve_plan(request: ApproveRequest):
    """
    Resume a run that is paused at the HITL planner interrupt.

    If edited_subquestions is provided, those are injected directly into the
    graph state (bypassing the planner LLM call entirely) before resuming.
    """
    thread_config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}
    state_snapshot = graph.get_state(thread_config)

    if not state_snapshot:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{request.thread_id}' not found.",
        )

    if state_snapshot.next != ("planner",):
        raise HTTPException(
            status_code=409,
            detail=f"Thread is not paused at planner. Current next: {state_snapshot.next}",
        )

    if not request.approved:
        with contextlib.suppress(Exception):
            await get_tracer().end_run(request.thread_id, status="rejected")
        return {"status": "rejected", "thread_id": request.thread_id}

    # Inject human-edited subquestions into graph state if provided
    if request.edited_subquestions:
        graph.update_state(
            thread_config,
            {"subquestions": request.edited_subquestions, "approved_plan": True},
            as_node="planner",
        )

    async def resume_generator():
        run_start = time.perf_counter()
        writer_completed = False

        try:
            async for event in graph.astream_events(None, version="v2", config=thread_config):
                kind = event["event"]

                if kind == "on_chain_start":
                    yield sse({"type": "node_start", "node": event["name"]})

                elif kind == "on_tool_start":
                    yield sse(
                        {
                            "type": "tool_call",
                            "tool": event["name"],
                            "input": str(event.get("data", {}).get("input", ""))[:200],
                        }
                    )

                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", [])
                    yield sse(
                        {
                            "type": "tool_result",
                            "tool": event["name"],
                            "count": len(output) if isinstance(output, list) else 1,
                        }
                    )

                elif kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield sse({"type": "token", "content": chunk.content})

                elif kind == "on_chain_end" and event["name"] == "writer":
                    writer_completed = True
                    yield sse({"type": "complete", "run_id": request.thread_id})
        finally:
            if writer_completed:
                await _finalize_run(request.thread_id, thread_config, run_start)

    return StreamingResponse(resume_generator(), media_type="text/event-stream")


@app.get("/research/state/{thread_id}")
async def get_research_state(thread_id: str):
    """Return the current graph state for a thread (used by the Streamlit UI)."""
    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state_snapshot = graph.get_state(thread_config)
    if not state_snapshot:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return {
        "thread_id": thread_id,
        "next_node": state_snapshot.next,
        "query": state_snapshot.values.get("query"),
        "query_difficulty": state_snapshot.values.get("query_difficulty"),
        "subquestions": state_snapshot.values.get("subquestions", []),
        "findings_count": len(state_snapshot.values.get("findings", [])),
        "approved_plan": state_snapshot.values.get("approved_plan"),
    }


@app.get("/research/runs")
async def list_runs(limit: int = 20):
    """Return recent run summaries from the observability DB."""
    try:
        return {"runs": get_tracer().get_recent_runs(limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/research/runs/{run_id}")
async def get_run_detail(run_id: str):
    """Return full observability detail for a single run."""
    tracer = get_tracer()
    summary = tracer.get_run_summary(run_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return {
        "summary": summary,
        "tool_stats": tracer.get_tool_call_stats(run_id),
        "node_timings": tracer.get_node_timings(run_id),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "deep-research-agent-api"}
