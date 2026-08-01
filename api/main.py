# api/main.py
"""
FastAPI gateway for the DeepResearch Agent.

Endpoints:
  POST  /research/stream           — Start a new SSE-streamed research run
  POST  /research/approve          — Resume a HITL-paused run
  GET   /research/state/{tid}      — Current graph state for a thread
  GET   /research/report/{tid}     — Final report as JSON
  GET   /research/report/{tid}/pdf — Final report as downloadable PDF
  GET   /research/report/{tid}/markdown — Final report as downloadable Markdown
  GET   /research/runs             — Recent run summaries
  GET   /research/runs/{run_id}    — Full observability detail for a run
  GET   /health                    — Health check
  GET   /health/deep               — Deep health check (DB, MCP, keys)
"""

import contextlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent import __version__
from agent.graph import graph
from agent.state import RunMetadata
from config.settings import settings
from observability.tracer import get_tracer
from utils.context import bind_run_id
from utils.cost_estimator import estimate_cost
from utils.logger import setup_logging
from utils.report_formatter import export_to_pdf, to_html, to_markdown

setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Graceful shutdown: close tracer SQLite connection
    get_tracer().close()


app = FastAPI(
    title="DeepResearch Agent API",
    version="1.0.0",
    description="Autonomous multi-source research agent powered by LangGraph and MCP servers.",
    lifespan=lifespan,
)


# correctly typed wrapper, no type: ignore needed:
def _handle_rate_limit(request: Request, exc: Exception) -> Response:
    return _rate_limit_exceeded_handler(request, cast(RateLimitExceeded, exc))


app.add_exception_handler(RateLimitExceeded, _handle_rate_limit)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


# ─────────────────────────── CORS Middleware ──────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",  # Dev
        "http://streamlit:8501",  # Docker internal
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Version Header ───────────────────────────────────


@app.middleware("http")
async def add_version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Agent-Version"] = __version__
    return response


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ─────────────────────────── Request Models ───────────────────────────────────


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1500)
    profile: str = "fast"


class ApproveRequest(BaseModel):
    thread_id: str
    approved: bool
    edited_subquestions: list[str] | None = None


# ─────────────────────── Tracer Helper ────────────────────────────────────────


async def _finalize_run(
    run_id: str,
    thread_config: RunnableConfig,
    start_time: float,
    tracer,
    status: str = "completed",
) -> None:
    """
    Read the final graph state and write the run's closing record to the tracer.
    Swallows all errors — tracing must never break the API response.
    """
    with contextlib.suppress(Exception):
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


# ─────────────────────── Report Retrieval Helpers ─────────────────────────────


def _get_report_from_thread(thread_id: str):
    """Retrieve the final ReportOutput from a completed graph thread."""
    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(thread_config)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    report = snapshot.values.get("final_report")
    if not report:
        raise HTTPException(
            status_code=404,
            detail=f"Thread '{thread_id}' has no final report. Research may still be in progress.",
        )
    return report


# ─────────────────────── Shared SSE Event Processing ─────────────────────────


async def _stream_graph_events(
    run_id: str,
    thread_config: RunnableConfig,
    input_payload: dict | None,
):
    """
    Shared async generator that streams LangGraph events as SSE messages.

    Handles all event types (node_start, tool_call, tool_result, token,
    complete) and auto-resumes follow-up iterations when the graph pauses
    at the planner interrupt.

    Yields:
        SSE-formatted strings for each graph event.
        A final "writer_completed" sentinel is NOT yielded — callers check
        the graph state after this generator exhausts.
    """
    MAIN_NODES = {
        "classifier",
        "planner",
        "supervisor",
        "web_agent",
        "arxiv_agent",
        "github_agent",
        "critic",
        "synthesizer",
        "writer",
    }
    while True:
        async for event in graph.astream_events(
            input_payload,
            version="v2",
            config=thread_config,
        ):
            kind = event["event"]
            now_iso = time.strftime("%H:%M:%S")

            if kind == "on_chain_start" and event["name"] in MAIN_NODES:
                node_name = event["name"]
                raw_input = event.get("data", {}).get("input", {})
                if isinstance(raw_input, dict):
                    clean_in = {
                        k: v
                        for k, v in raw_input.items()
                        if k
                        in (
                            "query",
                            "subquestions",
                            "profile",
                            "run_id",
                            "query_difficulty",
                            "relevant_sources",
                            "iteration_count",
                        )
                        and v
                    }
                    input_str = json.dumps(clean_in or raw_input, indent=2, default=str)
                else:
                    input_str = str(raw_input)
                yield sse(
                    {
                        "type": "node_start",
                        "node": node_name,
                        "input": input_str,
                        "timestamp": now_iso,
                    }
                )

            elif kind == "on_chain_end" and event["name"] in MAIN_NODES:
                node_name = event["name"]
                output_obj = event.get("data", {}).get("output", {})
                output_str = ""
                if isinstance(output_obj, dict):
                    clean_out: dict[str, Any] = {}
                    for k, v in output_obj.items():
                        if k == "findings" and isinstance(v, list):
                            clean_findings: list[Any] = []
                            for f in v:
                                if hasattr(f, "model_dump"):
                                    clean_findings.append(f.model_dump())
                                elif isinstance(f, dict):
                                    clean_findings.append(f)
                                else:
                                    clean_findings.append(str(f))
                            clean_out["findings"] = clean_findings
                        elif hasattr(v, "model_dump"):
                            clean_out[k] = v.model_dump()
                        elif isinstance(v, (dict, list, str, int, float, bool)):
                            clean_out[k] = v
                        else:
                            clean_out[k] = str(v)
                    output_str = json.dumps(clean_out, indent=2, default=str)
                elif hasattr(output_obj, "model_dump"):
                    output_str = json.dumps(output_obj.model_dump(), indent=2, default=str)
                else:
                    output_str = str(output_obj)

                yield sse(
                    {
                        "type": "node_end",
                        "node": node_name,
                        "output": output_str,
                        "timestamp": now_iso,
                    }
                )

            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                raw_input = event.get("data", {}).get("input", {})
                if isinstance(raw_input, dict):
                    input_str = json.dumps(raw_input, indent=2)
                else:
                    input_str = str(raw_input)
                yield sse(
                    {
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": input_str,
                        "timestamp": now_iso,
                    }
                )

            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                output = event.get("data", {}).get("output", [])
                count = len(output) if isinstance(output, list) else (1 if output else 0)

                # Format full output preview text for scrollable UI cards
                items_detail = []
                if isinstance(output, list):
                    for idx, item in enumerate(output, 1):
                        if isinstance(item, dict):
                            t = (
                                item.get("title")
                                or item.get("name")
                                or item.get("url")
                                or f"Item {idx}"
                            )
                            s = (
                                item.get("snippet")
                                or item.get("abstract")
                                or item.get("description")
                                or ""
                            )
                            u = item.get("url", "")
                            items_detail.append(f"[{idx}] {t}\n    URL: {u}\n    Snippet: {s}")
                        elif hasattr(item, "model_dump"):
                            d = item.model_dump()
                            t = d.get("title") or d.get("name") or d.get("url") or f"Item {idx}"
                            s = d.get("snippet") or d.get("abstract") or d.get("description") or ""
                            u = d.get("url", "")
                            items_detail.append(f"[{idx}] {t}\n    URL: {u}\n    Snippet: {s}")
                        else:
                            items_detail.append(f"[{idx}] {str(item)}")
                elif isinstance(output, dict):
                    items_detail.append(json.dumps(output, indent=2))
                else:
                    items_detail.append(str(output))

                full_output_str = "\n\n".join(items_detail)

                yield sse(
                    {
                        "type": "tool_result",
                        "tool": tool_name,
                        "count": count,
                        "full_output": full_output_str,
                        "timestamp": now_iso,
                    }
                )

            elif kind == "on_chat_model_start":
                model_name = event.get("name", "LLM")
                data = event.get("data", {})
                input_msgs = data.get("input", {}).get("messages", []) or data.get("messages", [])

                msg_list = []
                if isinstance(input_msgs, list):
                    for m in input_msgs:
                        if isinstance(m, list):
                            for sub in m:
                                content = getattr(sub, "content", str(sub))
                                role = getattr(sub, "type", "user")
                                msg_list.append(f"[{role.upper()}]\n{content}")
                        elif hasattr(m, "content"):
                            role = getattr(m, "type", "user")
                            msg_list.append(f"[{role.upper()}]\n{m.content}")
                        elif isinstance(m, dict):
                            role = m.get("role", "user")
                            content = m.get("content", "")
                            msg_list.append(f"[{role.upper()}]\n{content}")
                        else:
                            msg_list.append(str(m))
                formatted_prompt = "\n\n".join(msg_list) if msg_list else str(input_msgs)
                yield sse(
                    {
                        "type": "llm_start",
                        "model": model_name,
                        "prompt": formatted_prompt,
                        "timestamp": now_iso,
                    }
                )

            elif kind == "on_chat_model_end":
                model_name = event.get("name", "LLM")
                data = event.get("data", {})
                output_obj = data.get("output", {})

                if hasattr(output_obj, "content") and output_obj.content:
                    resp_str = str(output_obj.content)
                elif hasattr(output_obj, "model_dump"):
                    resp_str = json.dumps(output_obj.model_dump(), indent=2)
                elif isinstance(output_obj, dict):
                    resp_str = json.dumps(output_obj, indent=2)
                else:
                    resp_str = str(output_obj)

                yield sse(
                    {
                        "type": "llm_end",
                        "model": model_name,
                        "response": resp_str,
                        "timestamp": now_iso,
                    }
                )

            elif kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield sse({"type": "token", "content": chunk.content})

            elif kind == "on_chain_end" and event["name"] == "writer":
                yield sse({"type": "complete", "run_id": run_id, "timestamp": now_iso})

        # Event stream ended. Check if paused at supervisor for auto-resume.
        state_snapshot = graph.get_state(thread_config)
        if state_snapshot and state_snapshot.next == ("supervisor",):
            values = state_snapshot.values
            meta = values.get("run_metadata")
            iteration = meta.iteration_count if meta else 0
            if iteration > 0:
                # Auto-resume follow-up iterations
                input_payload = None
                continue
        # Either not paused at supervisor, or iteration == 0 (needs HITL) — stop streaming
        break


# ─────────────────────── Endpoints ───────────────────────────────────────────


@app.post("/research/stream")
@limiter.limit("5/minute")
async def stream_research(payload: ResearchRequest, request: Request, tracer=Depends(get_tracer)):
    """
    Start a new research run and stream SSE events.

    The graph is compiled with interrupt_before=["supervisor"], so after the
    classifier and planner run, the stream pauses and emits a hitl_interrupt event.
    The client must call POST /research/approve to resume.
    """
    run_id = str(uuid.uuid4())
    bind_run_id(run_id)  # propagate to all downstream log records via contextvars
    thread_config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    # Register the run in the observability DB immediately
    with contextlib.suppress(Exception):
        await tracer.start_run(run_id, payload.query, payload.profile)

    async def event_generator():
        run_start = time.perf_counter()
        writer_completed = False

        try:
            input_payload = {
                "query": payload.query,
                "profile": payload.profile,
                "run_id": run_id,
                "query_difficulty": "",
                "subquestions": [],
                "approved_plan": False,
                "findings": [],
                "critique": None,
                "iteration_count": 0,
                "final_report": None,
                "run_metadata": RunMetadata(run_id=run_id, profile=payload.profile),
                "error_log": [],
                "thought_log": [],
            }

            async for chunk in _stream_graph_events(run_id, thread_config, input_payload):
                yield chunk
                if '"type": "complete"' in chunk:
                    writer_completed = True

            # If paused at supervisor on first iteration, emit HITL interrupt
            state_snapshot = graph.get_state(thread_config)
            if state_snapshot and state_snapshot.next == ("supervisor",):
                values = state_snapshot.values
                meta = values.get("run_metadata")
                iteration = meta.iteration_count if meta else 0

                if iteration == 0:
                    difficulty = values.get("query_difficulty", "narrow")
                    sources = values.get("relevant_sources", ["web"])
                    subquestions = values.get("subquestions", [])
                    n_questions = len(subquestions) or {
                        "narrow": 3,
                        "broad": 6,
                        "ambiguous": 4,
                    }.get(difficulty, 4)
                    estimated_cost = estimate_cost(
                        settings.default_model, n_questions * 800, n_questions * 400
                    )
                    yield sse(
                        {
                            "type": "hitl_interrupt",
                            "thread_id": run_id,
                            "query_difficulty": difficulty,
                            "relevant_sources": sources,
                            "subquestions": subquestions,
                            "estimated_subquestions": n_questions,
                            "estimated_cost_usd": round(estimated_cost, 4),
                            "message": "Plan ready for approval. POST /research/approve to continue.",
                        }
                    )

        finally:
            if writer_completed:
                await _finalize_run(run_id, thread_config, run_start, tracer)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/research/approve")
async def approve_plan(request: ApproveRequest, tracer=Depends(get_tracer)):
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

    if state_snapshot.next != ("supervisor",):
        raise HTTPException(
            status_code=409,
            detail=f"Thread is not paused at supervisor. Current next: {state_snapshot.next}",
        )

    if not request.approved:
        with contextlib.suppress(Exception):
            await tracer.end_run(request.thread_id, status="rejected")
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
            async for chunk in _stream_graph_events(request.thread_id, thread_config, None):
                yield chunk
                if '"type": "complete"' in chunk:
                    writer_completed = True
        finally:
            if writer_completed:
                await _finalize_run(request.thread_id, thread_config, run_start, tracer)

    return StreamingResponse(resume_generator(), media_type="text/event-stream")


@app.get("/research/state/{thread_id}")
async def get_research_state(thread_id: str):
    """Return the current graph state for a thread (used by the Streamlit UI)."""
    thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state_snapshot = graph.get_state(thread_config)
    if not state_snapshot:
        raise HTTPException(status_code=404, detail="Thread not found.")

    final_report_obj = state_snapshot.values.get("final_report")
    run_meta_obj = state_snapshot.values.get("run_metadata")

    return {
        "thread_id": thread_id,
        "next_node": state_snapshot.next,
        "query": state_snapshot.values.get("query"),
        "query_difficulty": state_snapshot.values.get("query_difficulty"),
        "subquestions": state_snapshot.values.get("subquestions", []),
        "findings_count": len(state_snapshot.values.get("findings", [])),
        "approved_plan": state_snapshot.values.get("approved_plan"),
        "thought_log": state_snapshot.values.get("thought_log", []),
        "has_final_report": bool(final_report_obj),
        "final_report": final_report_obj.model_dump()
        if hasattr(final_report_obj, "model_dump")
        else final_report_obj,
        "run_metadata": run_meta_obj.model_dump()
        if hasattr(run_meta_obj, "model_dump")
        else run_meta_obj,
    }


# ─────────────────────── Report Endpoints ─────────────────────────────────────


@app.get("/research/report/{thread_id}")
async def get_report(thread_id: str):
    """Return the final report as JSON."""
    report = _get_report_from_thread(thread_id)
    return report.model_dump()


@app.get("/research/report/{thread_id}/pdf")
async def get_report_pdf(thread_id: str):
    """Return the final report as a downloadable PDF."""
    report = _get_report_from_thread(thread_id)
    try:
        pdf_bytes = export_to_pdf(report)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="research_report_{thread_id[:8]}.pdf"'
            },
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires WeasyPrint. Install with: pip install weasyprint",
        ) from exc
    except Exception as exc:
        logger.exception("PDF generation failed for thread %s", thread_id)
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {type(exc).__name__}: {str(exc)[:200]}",
        ) from exc


@app.get("/research/report/{thread_id}/markdown")
async def get_report_markdown(thread_id: str):
    """Return the final report as downloadable Markdown."""
    report = _get_report_from_thread(thread_id)
    md_content = to_markdown(report)
    return Response(
        content=md_content.encode("utf-8"),
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="research_report_{thread_id[:8]}.md"'
        },
    )


@app.get("/research/report/{thread_id}/html")
async def get_report_html(thread_id: str):
    """Return the final report as styled HTML."""
    report = _get_report_from_thread(thread_id)
    html_content = to_html(report)
    return Response(
        content=html_content.encode("utf-8"),
        media_type="text/html",
    )


# ─────────────────────── Observability Endpoints ──────────────────────────────


@app.get("/research/runs")
async def list_runs(limit: int = 20, tracer=Depends(get_tracer)):
    """Return recent run summaries from the observability DB."""
    try:
        return {"runs": tracer.get_recent_runs(limit=limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/research/runs/{run_id}")
async def get_run_detail(run_id: str, tracer=Depends(get_tracer)):
    """Return full observability detail for a single run."""
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
    return {"status": "healthy", "service": "deep-research-agent-api", "version": __version__}


@app.get("/health/deep")
async def health_deep():
    """
    Deep health check that validates critical dependencies:
    - SQLite checkpoint DB is accessible
    - At least one MCP server is reachable
    - OpenAI API key is configured
    """
    import os

    import httpx as _httpx

    checks: dict[str, dict] = {}

    # 1. SQLite checkpoint DB
    try:
        tracer = get_tracer()
        tracer.get_recent_runs(limit=1)
        checks["sqlite_tracer"] = {"status": "ok"}
    except Exception as exc:
        checks["sqlite_tracer"] = {"status": "error", "detail": str(exc)[:200]}

    # 2. MCP server reachability (check all three, pass if >= 1 is up)
    mcp_endpoints = {
        "web_search": "http://web-search-mcp:8001/health",
        "arxiv": "http://arxiv-mcp:8002/health",
        "github": "http://github-mcp:8003/health",
    }
    mcp_results: dict[str, str] = {}
    mcp_up = 0
    async with _httpx.AsyncClient(timeout=5.0) as client:
        for name, url in mcp_endpoints.items():
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    mcp_results[name] = "ok"
                    mcp_up += 1
                else:
                    mcp_results[name] = f"http_{resp.status_code}"
            except Exception:
                mcp_results[name] = "unreachable"
    checks["mcp_servers"] = {
        "status": "ok" if mcp_up > 0 else "error",
        "reachable": mcp_up,
        "total": len(mcp_endpoints),
        "details": mcp_results,
    }

    # 3. API key configuration
    keys_configured = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "TAVILY_API_KEY": bool(os.environ.get("TAVILY_API_KEY")),
    }
    all_keys_set = all(keys_configured.values())
    checks["api_keys"] = {
        "status": "ok" if all_keys_set else "warning",
        "configured": keys_configured,
    }

    overall = (
        "healthy" if all(c["status"] in ("ok", "warning") for c in checks.values()) else "degraded"
    )

    status_code = 200 if overall == "healthy" else 503
    return Response(
        content=json.dumps(
            {"status": overall, "version": __version__, "checks": checks},
            indent=2,
        ),
        media_type="application/json",
        status_code=status_code,
    )
