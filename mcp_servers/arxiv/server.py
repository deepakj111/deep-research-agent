# mcp_servers/arxiv/server.py
import os
import sys
import xml.etree.ElementTree as ET

import httpx
from fastmcp import FastMCP
from starlette.responses import JSONResponse

# Ensure the shared directory is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.auth import require_auth  # noqa: E402
from shared.cache import CacheLayer  # noqa: E402

mcp = FastMCP("arxiv-server")
cache = CacheLayer(db_path=".arxiv_cache.db", ttl_seconds=86400)

ARXIV_NS = "{http://www.w3.org/2005/Atom}"


def _parse_atom(xml_text: str) -> list[dict]:
    """Parse arXiv Atom XML feed into a list of paper dicts."""
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
@require_auth
async def fetch_papers(ctx, query: str, max_papers: int = 5) -> list[dict]:
    """
    Search arXiv for academic papers matching the query.

    Returns a list of paper dicts with fields matching the ArxivPaper
    Pydantic model: arxiv_id, title, abstract, authors, published_date,
    url, citation_count, trust_score.
    """
    cache_key = f"arxiv:{query}:{max_papers}"
    cached = cache.get(cache_key)
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

    cache.set(cache_key, papers)
    return papers


@mcp.custom_route("/health", methods=["GET"])
async def health(request) -> JSONResponse:
    return JSONResponse({"status": "healthy", "server": "arxiv-mcp"})


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8002)  # nosec B104
