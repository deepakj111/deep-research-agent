# agent/nodes/web_agent.py
import contextlib
import time

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.circuit_breaker import circuit_breakers
from agent.state import ResearchFindings, ResearchState, WebResult
from config.settings import settings
from observability.tracer import ToolCallRecord, get_tracer
from utils.cost_estimator import get_jwt_token

_profile_cache: dict = {}


def load_profile(name: str) -> dict:
    import yaml

    if name not in _profile_cache:
        with open(f"config/profiles/{name}.yaml") as f:
            _profile_cache[name] = yaml.safe_load(f)
    return _profile_cache[name]


async def run(state: ResearchState) -> dict:
    # When called via Send, subquestions contains exactly 1 item.
    # Sequential fallback uses coverage index.
    subquestions = state.get("subquestions", [])
    subquestion = (
        subquestions[0] if len(subquestions) == 1 else subquestions[len(state.get("findings", []))]
    )

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

            async def _call():
                return await search_tool.ainvoke({"query": subquestion, "max_results": max_results})

            raw = await circuit_breakers["search_web"].call(_call())
            if isinstance(raw, list):
                results = [WebResult(**r) for r in raw]
            elif isinstance(raw, dict) and "results" in raw:
                results = [WebResult(**r) for r in raw["results"]]
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
