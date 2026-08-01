import asyncio
import json
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


def parse_mcp_raw_results(raw: Any) -> list[dict]:
    """
    Safely parse output from MCP tool invocation into a list of dictionaries.

    Handles:
    - List of dicts: [{'url': '...', ...}]
    - List of MCP text content blocks: [{'type': 'text', 'text': '[{"url": ...}]'}]
    - JSON string: '[{"url": ...}]'
    - Dict with results/papers/repos key: {'results': [{'url': ...}]}
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []

    if isinstance(raw, list):
        if not raw:
            return []
        first = raw[0]
        # Check if raw is a list of MCP content blocks (e.g. [{'type': 'text', 'text': '...'}]
        if isinstance(first, dict) and "type" in first and "text" in first:
            extracted: list[dict] = []
            for block in raw:
                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                    text_val = block["text"]
                    if isinstance(text_val, str):
                        try:
                            parsed_text = json.loads(text_val)
                            if isinstance(parsed_text, list):
                                extracted.extend(parsed_text)
                            elif isinstance(parsed_text, dict):
                                extracted.append(parsed_text)
                        except Exception:
                            pass
                    elif isinstance(text_val, list):
                        extracted.extend(text_val)
            return extracted
        return [r for r in raw if isinstance(r, dict)]

    if isinstance(raw, dict):
        for key in ("results", "papers", "repos"):
            if key in raw and isinstance(raw[key], list):
                return [r for r in raw[key] if isinstance(r, dict)]

    return []


_mcp_cache: dict[str, tuple[MultiServerMCPClient, list[Any]]] = {}
_cache_lock: asyncio.Lock | None = None


def _get_cache_lock() -> asyncio.Lock:
    global _cache_lock
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    return _cache_lock


async def get_mcp_tool(server_name: str, url: str, jwt_token: str, tool_name: str) -> Any:
    """
    Get a specific tool from an MCP server using a cached client instance.

    Reuses existing MultiServerMCPClient connections to avoid establishing new
    SSE handshakes and tool schema discovery on every subquestion execution.
    """
    cache_key = f"{server_name}:{url}"
    lock = _get_cache_lock()

    async with lock:
        if cache_key in _mcp_cache:
            _, tools = _mcp_cache[cache_key]
            tool = next((t for t in tools if getattr(t, "name", None) == tool_name), None)
            if tool is not None:
                return tool

        # Connect and discover tools
        client = MultiServerMCPClient(
            {
                server_name: {
                    "url": url,
                    "transport": "sse",
                    "headers": {"Authorization": f"Bearer {jwt_token}"},
                }
            }
        )
        tools = await client.get_tools()
        _mcp_cache[cache_key] = (client, tools)

        tool = next((t for t in tools if getattr(t, "name", None) == tool_name), None)
        if tool is not None:
            return tool
        raise RuntimeError(f"Tool '{tool_name}' not found on MCP server '{server_name}'")


def invalidate_mcp_cache(server_name: str, url: str) -> None:
    """Invalidate cache entry for a server if connection drops."""
    cache_key = f"{server_name}:{url}"
    _mcp_cache.pop(cache_key, None)
