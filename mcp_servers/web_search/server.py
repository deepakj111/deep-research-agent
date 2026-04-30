# mcp_servers/web_search/server.py
import os
import sys

import httpx
from fastmcp import FastMCP
from starlette.responses import JSONResponse

# Ensure the shared directory is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.auth import require_auth  # noqa: E402
from shared.cache import CacheLayer  # noqa: E402

mcp = FastMCP("web-search-server")
cache = CacheLayer(ttl_seconds=3600)


@mcp.tool()
@require_auth
async def search_web(ctx, query: str, max_results: int = 5) -> list[dict]:
    """
    Search the live web using Tavily and return structured results.

    Each result dict contains fields that match the WebResult Pydantic model:
      - url: str
      - title: str
      - snippet: str
      - relevance_score: float (0-1)

    Tavily's raw response uses 'content' and 'score' — we normalise here
    so the agent's WebResult(**r) construction never fails validation.
    """
    cache_key = f"web:{query}:{max_results}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        raise RuntimeError("TAVILY_API_KEY environment variable is not set.")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        raw_results: list[dict] = response.json().get("results", [])

    # Normalise Tavily field names to match WebResult Pydantic model
    # Tavily returns: url, title, content, score
    # WebResult expects: url, title, snippet, relevance_score
    normalised = [
        {
            "url": r.get("url", ""),
            "title": r.get("title", ""),
            "snippet": r.get("content", r.get("snippet", "")),
            "relevance_score": float(r.get("score", r.get("relevance_score", 0.5))),
        }
        for r in raw_results
    ]

    cache.set(cache_key, normalised)
    return normalised


@mcp.custom_route("/health", methods=["GET"])
async def health(request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "server": "web-search-mcp"})


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)  # nosec B104
