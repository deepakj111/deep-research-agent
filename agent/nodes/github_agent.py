import asyncio
import contextlib
import time

from agent.circuit_breaker import circuit_breakers
from agent.middleware.pii_filter import filter_pii_simple
from agent.retry_policy import ToolDegradedError, retry_with_policy
from agent.state import GitHubRepo, ResearchFindings, ResearchState
from config.profiles import load_profile
from config.settings import settings
from observability.tracer import ToolCallRecord, get_tracer
from utils.auth import get_jwt_token
from utils.mcp_helpers import get_mcp_tool, parse_mcp_raw_results

# Limit concurrent GitHub MCP calls across all parallel agents
_semaphore = asyncio.Semaphore(settings.agent_max_concurrency)


async def run(state: ResearchState) -> dict:
    # Supervisor's Send() always passes exactly one subquestion per agent invocation.
    subquestions = state.get("subquestions", [])
    subquestion = subquestions[0] if subquestions else ""

    profile = load_profile(state.get("profile", "fast"))
    max_repos = profile.get("max_github_repos", 3)
    run_id = state.get("run_id", "")

    repos: list[GitHubRepo] = []
    errors: list[str] = []
    start = time.perf_counter()
    success = True

    try:
        async with _semaphore:
            search_tool = await get_mcp_tool(
                "github", settings.github_mcp_url, get_jwt_token(), "search_repos"
            )

            async def _call_with_cb():
                async def _inner():
                    return await asyncio.wait_for(
                        search_tool.ainvoke({"topic": subquestion, "max_repos": max_repos}),
                        timeout=12.0,
                    )

                return await circuit_breakers["search_repos"].call(_inner())

            raw = await retry_with_policy("search_repos", _call_with_cb)

            parsed_list = parse_mcp_raw_results(raw)
            repos = [GitHubRepo(**r) for r in parsed_list]

            # PII scrub string fields after model construction
            for repo in repos:
                repo.description = filter_pii_simple(repo.description)
    except ToolDegradedError as e:
        # Non-critical tool — degrade gracefully per policy
        errors.append(e.failure_note)
        success = False
    except Exception as e:
        error_msg = f"search_repos [{type(e).__name__}]: {str(e)[:200]}"
        errors.append(error_msg)
        success = False

    latency_ms = (time.perf_counter() - start) * 1000

    # ── Observability ──────────────────────────────────────────────────────────
    with contextlib.suppress(Exception):
        await get_tracer().log_tool_call(
            ToolCallRecord(
                run_id=run_id,
                node_name="github_agent",
                tool_name="search_repos",
                input_summary=subquestion[:200],
                success=success,
                latency_ms=round(latency_ms, 2),
                error_message=errors[0] if errors else None,
            )
        )

    findings = ResearchFindings(
        subquestion=subquestion,
        repos=repos,
        tool_errors=errors,
    )

    status = f"DEGRADED: {errors[0]}" if errors else f"{len(repos)} repos"
    return {
        "findings": [findings],
        "thought_log": [f"[GitHubAgent] '{subquestion[:60]}...' → {status}"],
    }
