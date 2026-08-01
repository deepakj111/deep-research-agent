import json
from typing import Any


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
