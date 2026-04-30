# agent/nodes/web_agent.py
import contextlib
import time

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.circuit_breaker import circuit_breakers
from agent.middleware.pii_filter import filter_pii_simple
from agent.retry_policy import ToolDegradedError, retry_with_policy
from agent.state import ResearchFindings, ResearchState, WebResult
from config.profiles import load_profile
from config.settings import settings
from observability.tracer import ToolCallRecord, get_tracer
from utils.auth import get_jwt_token


async def run(state: ResearchState) -> dict:
    # Supervisor's Send() always passes exactly one subquestion per agent invocation.
    subquestions = state.get("subquestions", [])
    subquestion = subquestions[0] if subquestions else ""

    profile = load_profile(state.get("profile", "fast"))
    max_results = profile.get("max_web_results", 3)
    run_id = state.get("run_id", "")

    results: list[WebResult] = []
    errors: list[str] = []
    start = time.perf_counter()
    success = True

    try:
        async with MultiServerMCPClient(
            {
                "web_search": {
                    "url": settings.web_search_mcp_url,
                    "transport": "sse",
                    "headers": {"Authorization": f"Bearer {get_jwt_token()}"},
                }
            }
        ) as client:  # type: ignore[misc]
            tools = await client.get_tools()
            search_tool = next(t for t in tools if t.name == "search_web")

            # Retry policy wraps circuit breaker: retries happen inside the CB window
            async def _call_with_cb():
                async def _inner():
                    return await search_tool.ainvoke(
                        {"query": subquestion, "max_results": max_results}
                    )

                return await circuit_breakers["search_web"].call(_inner())

            raw = await retry_with_policy("search_web", _call_with_cb)

            if isinstance(raw, list):
                results = [WebResult(**r) for r in raw]
            elif isinstance(raw, dict) and "results" in raw:
                results = [WebResult(**r) for r in raw["results"]]

            # PII scrub string fields after model construction
            for wr in results:
                wr.title = filter_pii_simple(wr.title)
                wr.snippet = filter_pii_simple(wr.snippet)
    except ToolDegradedError as e:
        errors.append(f"web_search [degraded]: {e.failure_note}")
        success = False
    except Exception as e:
        error_msg = f"web_search [{type(e).__name__}]: {str(e)[:200]}"
        errors.append(error_msg)
        success = False

    latency_ms = (time.perf_counter() - start) * 1000

    # ── Observability ──────────────────────────────────────────────────────────
    # Fire-and-forget: never let tracing errors propagate to the agent.
    with contextlib.suppress(Exception):
        await get_tracer().log_tool_call(
            ToolCallRecord(
                run_id=run_id,
                node_name="web_agent",
                tool_name="search_web",
                input_summary=subquestion[:200],
                success=success,
                latency_ms=round(latency_ms, 2),
                error_message=errors[0] if errors else None,
            )
        )

    findings = ResearchFindings(
        subquestion=subquestion,
        web_results=results,
        tool_errors=errors,
    )

    status = f"DEGRADED: {errors[0]}" if errors else f"{len(results)} results"
    return {
        "findings": [findings],
        "thought_log": [f"[WebAgent] '{subquestion[:60]}...' → {status}"],
    }
