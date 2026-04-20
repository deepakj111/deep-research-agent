import os

import httpx
from cache import CacheLayer
from fastmcp import FastMCP

mcp = FastMCP("web-search-server")
cache = CacheLayer(ttl_seconds=3600)


@mcp.tool()
async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the live web using Tavily. Returns structured results with
    url, title, snippet, and relevance_score."""
    cache_key = f"web:{query}:{max_results}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.environ["TAVILY_API_KEY"],
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        results = response.json()["results"]

    cache.set(cache_key, results)
    return results


@mcp.custom_route("/health", methods=["GET"])
async def health(request) -> dict:
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "healthy", "server": "web-search-mcp"})


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)  # nosec B104
