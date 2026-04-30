# mcp_servers/github/server.py
import os
import sys

import httpx
from fastmcp import FastMCP
from starlette.responses import JSONResponse

# Ensure the shared directory is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.auth import require_auth  # noqa: E402
from shared.cache import CacheLayer  # noqa: E402

mcp = FastMCP("github-server")
cache = CacheLayer(db_path=".github_cache.db", ttl_seconds=7200)


@mcp.tool()
@require_auth
async def search_repos(ctx, topic: str, max_repos: int = 5) -> list[dict]:
    """
    Search GitHub repositories by topic/keyword.

    Returns a list of repo dicts with fields matching the GitHubRepo
    Pydantic model: name, url, description, stars, language,
    last_updated, trust_score.
    """
    cache_key = f"github:{topic}:{max_repos}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    github_token = os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.github.com/search/repositories",
            params={
                "q": topic,
                "sort": "stars",
                "order": "desc",
                "per_page": max_repos,
            },
            headers=headers,
            timeout=15.0,
        )
        response.raise_for_status()
        items = response.json().get("items", [])

    repos = [
        {
            "name": r["full_name"],
            "url": r["html_url"],
            "description": r.get("description") or "",
            "stars": r.get("stargazers_count", 0),
            "language": r.get("language"),
            "last_updated": (r.get("updated_at") or "")[:10],
            "trust_score": 0.0,
        }
        for r in items
    ]

    cache.set(cache_key, repos)
    return repos


@mcp.custom_route("/health", methods=["GET"])
async def health(request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "server": "github-mcp"})


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8003)  # nosec B104
