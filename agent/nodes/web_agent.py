from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.circuit_breaker import circuit_breakers
from agent.state import ResearchFindings, ResearchState, WebResult
from config.settings import settings
from utils.cost_estimator import get_jwt_token

_profile_cache: dict = {}


def load_profile(name: str) -> dict:
    import yaml

    if name not in _profile_cache:
        with open(f"config/profiles/{name}.yaml") as f:
            _profile_cache[name] = yaml.safe_load(f)
    return _profile_cache[name]


async def run(state: ResearchState) -> dict:
    covered = len(state.get("findings", []))
    subquestion = state["subquestions"][covered]
    profile = load_profile(state.get("profile", "fast"))
    max_results = profile.get("max_web_results", 3)

    results: list[WebResult] = []
    errors: list[str] = []

    try:
        async with MultiServerMCPClient(
            {
                "web_search": {
                    "url": settings.web_search_mcp_url,
                    "transport": "sse",
                    "headers": {"Authorization": f"Bearer {get_jwt_token()}"},
                }
            }
        ) as client:
            tools = client.get_tools()
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
