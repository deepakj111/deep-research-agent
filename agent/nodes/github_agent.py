# agent/nodes/github_agent.py
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.circuit_breaker import circuit_breakers
from agent.state import GitHubRepo, ResearchFindings, ResearchState
from config.settings import settings
from utils.cost_estimator import get_jwt_token


def load_profile(name: str) -> dict:
    import yaml

    with open(f"config/profiles/{name}.yaml") as f:
        return yaml.safe_load(f)


async def run(state: ResearchState) -> dict:
    subquestions = state.get("subquestions", [])
    if len(subquestions) == 1:
        subquestion = subquestions[0]
    else:
        covered = len(state.get("findings", []))
        subquestion = subquestions[covered]

    profile = load_profile(state.get("profile", "fast"))
    max_repos = profile.get("max_github_repos", 3)

    repos: list[GitHubRepo] = []
    errors: list[str] = []

    try:
        async with MultiServerMCPClient(
            {
                "github": {
                    "url": settings.github_mcp_url,
                    "transport": "sse",
                    "headers": {"Authorization": f"Bearer {get_jwt_token()}"},
                }
            }
        ) as client:
            tools = client.get_tools()
            search_tool = next(t for t in tools if t.name == "search_repos")

            async def _call():
                return await search_tool.ainvoke({"topic": subquestion, "max_repos": max_repos})

            raw = await circuit_breakers["search_repos"].call(_call())
            if isinstance(raw, list):
                repos = [GitHubRepo(**r) for r in raw]
    except Exception as e:
        error_msg = f"search_repos [{type(e).__name__}]: {str(e)[:200]}"
        errors.append(error_msg)

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
