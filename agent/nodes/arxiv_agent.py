# agent/nodes/arxiv_agent.py
import contextlib
import time

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.circuit_breaker import circuit_breakers
from agent.middleware.pii_filter import filter_pii_simple
from agent.retry_policy import ToolDegradedError, retry_with_policy
from agent.state import ArxivPaper, ResearchFindings, ResearchState
from config.settings import settings
from observability.tracer import ToolCallRecord, get_tracer
from utils.cost_estimator import get_jwt_token


def load_profile(name: str) -> dict:
    import yaml

    with open(f"config/profiles/{name}.yaml") as f:
        return yaml.safe_load(f)


async def run(state: ResearchState) -> dict:
    subquestions = state.get("subquestions", [])
    subquestion = (
        subquestions[0] if len(subquestions) == 1 else subquestions[len(state.get("findings", []))]
    )

    profile = load_profile(state.get("profile", "fast"))
    max_papers = profile.get("max_arxiv_papers", 2)
    run_id = state.get("run_id", "")

    papers: list[ArxivPaper] = []
    errors: list[str] = []
    start = time.perf_counter()
    success = True

    try:
        async with MultiServerMCPClient(
            {
                "arxiv": {
                    "url": settings.arxiv_mcp_url,
                    "transport": "sse",
                    "headers": {"Authorization": f"Bearer {get_jwt_token()}"},
                }
            }
        ) as client:  # type: ignore[misc]
            tools = await client.get_tools()
            fetch_tool = next(t for t in tools if t.name == "fetch_papers")

            async def _call_with_cb():
                async def _inner():
                    return await fetch_tool.ainvoke(
                        {"query": subquestion, "max_papers": max_papers}
                    )

                return await circuit_breakers["fetch_papers"].call(_inner())

            raw = await retry_with_policy("fetch_papers", _call_with_cb)

            if isinstance(raw, list):
                papers = [ArxivPaper(**p) for p in raw]

            # PII scrub string fields after model construction
            for paper in papers:
                paper.title = filter_pii_simple(paper.title)
                paper.abstract = filter_pii_simple(paper.abstract)
    except ToolDegradedError as e:
        # Non-critical tool — degrade gracefully per policy
        errors.append(e.failure_note)
        success = False
    except Exception as e:
        error_msg = f"fetch_papers [{type(e).__name__}]: {str(e)[:200]}"
        errors.append(error_msg)
        success = False

    latency_ms = (time.perf_counter() - start) * 1000

    # ── Observability ──────────────────────────────────────────────────────────
    with contextlib.suppress(Exception):
        await get_tracer().log_tool_call(
            ToolCallRecord(
                run_id=run_id,
                node_name="arxiv_agent",
                tool_name="fetch_papers",
                input_summary=subquestion[:200],
                success=success,
                latency_ms=round(latency_ms, 2),
                error_message=errors[0] if errors else None,
            )
        )

    findings = ResearchFindings(
        subquestion=subquestion,
        papers=papers,
        tool_errors=errors,
    )

    status = f"DEGRADED: {errors[0]}" if errors else f"{len(papers)} papers"
    return {
        "findings": [findings],
        "thought_log": [f"[ArxivAgent] '{subquestion[:60]}...' → {status}"],
    }
