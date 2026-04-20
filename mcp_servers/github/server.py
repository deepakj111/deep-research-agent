import json
import os
import sqlite3
import time

import httpx
from fastmcp import FastMCP

mcp = FastMCP("github-server")

_conn = sqlite3.connect(".github_cache.db", check_same_thread=False)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
)
_conn.commit()

TTL = 7200  # 2 hours


def _cache_get(key: str) -> list | None:
    row = _conn.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,)).fetchone()
    if row and time.time() < row[1]:
        return json.loads(row[0])
    return None


def _cache_set(key: str, value: list) -> None:
    _conn.execute(
        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
        (key, json.dumps(value), time.time() + TTL),
    )
    _conn.commit()


@mcp.tool()
async def search_repos(topic: str, max_repos: int = 5) -> list[dict]:
    """Search GitHub repositories by topic/keyword. Returns structured repo metadata."""
    cache_key = f"github:{topic}:{max_repos}"
    cached = _cache_get(cache_key)
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

    _cache_set(cache_key, repos)
    return repos


@mcp.custom_route("/health", methods=["GET"])
async def health(request) -> dict:
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "healthy", "server": "github-mcp"})


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8003)  # nosec B104
