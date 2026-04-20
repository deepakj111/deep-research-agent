import json
import sqlite3
import time
import xml.etree.ElementTree as ET

import httpx
from fastmcp import FastMCP

mcp = FastMCP("arxiv-server")

_conn = sqlite3.connect(".arxiv_cache.db", check_same_thread=False)
_conn.execute(
    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
)
_conn.commit()

ARXIV_NS = "{http://www.w3.org/2005/Atom}"
TTL = 86400  # 24 hours


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


def _parse_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)  # nosec B314
    papers = []
    for entry in root.findall(f"{ARXIV_NS}entry"):
        arxiv_id_raw = (entry.findtext(f"{ARXIV_NS}id") or "").strip()
        arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
        authors = [a.findtext(f"{ARXIV_NS}name") or "" for a in entry.findall(f"{ARXIV_NS}author")]
        published = (entry.findtext(f"{ARXIV_NS}published") or "")[:10]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": (entry.findtext(f"{ARXIV_NS}title") or "").strip().replace("\n", " "),
                "abstract": (entry.findtext(f"{ARXIV_NS}summary") or "").strip().replace("\n", " "),
                "authors": authors,
                "published_date": published,
                "url": arxiv_id_raw,
                "citation_count": 0,
                "trust_score": 0.0,
            }
        )
    return papers


@mcp.tool()
async def fetch_papers(query: str, max_papers: int = 5) -> list[dict]:
    """Search arXiv for academic papers. Returns structured paper metadata."""
    cache_key = f"arxiv:{query}:{max_papers}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": max_papers,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=20.0,
        )
        response.raise_for_status()
        papers = _parse_atom(response.text)

    _cache_set(cache_key, papers)
    return papers


@mcp.custom_route("/health", methods=["GET"])
async def health(request) -> dict:
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "healthy", "server": "arxiv-mcp"})


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8002)  # nosec B104
