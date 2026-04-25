# utils/cost_estimator.py
"""
Dynamic LLM cost estimation using the community-maintained LiteLLM pricing
database (https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

Pricing data is fetched on first use and cached locally at
``$XDG_CACHE_HOME/deep-research-agent/model_prices.json`` (default:
``~/.cache/deep-research-agent/model_prices.json``).  The cache is refreshed
automatically when it is older than 7 days.

This approach avoids depending on the ``litellm`` package at runtime (which has
aggressive dependency pins that conflict with the rest of the stack) while still
providing dynamic, community-maintained pricing for 2,600+ models.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import jwt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

# XDG-compliant cache location
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "deep-research-agent"

_CACHE_FILE = _CACHE_DIR / "model_prices.json"

# Refresh cache after 7 days
_CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60

# Network timeout — keeps CI and offline starts fast
_FETCH_TIMEOUT_SECONDS = 10

# Conservative fallback — matches gpt-4o so unknown models are never underestimated.
_DEFAULT_INPUT_COST_PER_TOKEN = 2.50 / 1_000_000
_DEFAULT_OUTPUT_COST_PER_TOKEN = 10.00 / 1_000_000

# ---------------------------------------------------------------------------
# Lazy-loaded pricing map (module-level singleton)
# ---------------------------------------------------------------------------

_cost_map: dict[str, dict] | None = None


def _cache_is_fresh() -> bool:
    """Return True if the local cache file exists and is younger than _CACHE_MAX_AGE_SECONDS."""
    try:
        age = time.time() - _CACHE_FILE.stat().st_mtime
        return age < _CACHE_MAX_AGE_SECONDS
    except (FileNotFoundError, OSError):
        return False


def _fetch_and_cache() -> dict[str, dict]:
    """
    Download the LiteLLM community pricing JSON and write it to the local cache.

    Uses httpx (already a project dependency) with a short timeout so the
    call never blocks CI or offline environments for long.  Returns the parsed
    dict on success, or an empty dict on any failure.
    """
    try:
        import httpx

        resp = httpx.get(_PRICING_URL, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
        resp.raise_for_status()
        data: dict[str, dict] = resp.json()
        # Persist to cache
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(resp.text, encoding="utf-8")
        logger.info("Fetched and cached pricing data for %d models", len(data))
        return data
    except Exception:
        logger.debug("Failed to fetch pricing data from %s", _PRICING_URL, exc_info=True)
        return {}


def _load_cache() -> dict[str, dict]:
    """Load pricing data from the local cache file, returning {} on any error."""
    try:
        with open(_CACHE_FILE, encoding="utf-8") as fh:
            data: dict[str, dict] = json.load(fh)
        logger.debug("Loaded cached pricing data for %d models", len(data))
        return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _get_cost_map() -> dict[str, dict]:
    """
    Return the pricing map, fetching / refreshing as needed.

    Resolution order:
      1. In-process cache (instant)
      2. Fresh local disk cache (< 7 days old)
      3. Network fetch from LiteLLM GitHub → write to disk cache
      4. Stale local disk cache (better than nothing)
      5. Empty dict (all models get conservative fallback)
    """
    global _cost_map
    if _cost_map is not None:
        return _cost_map

    # Try fresh cache first
    if _cache_is_fresh():
        _cost_map = _load_cache()
        if _cost_map:
            return _cost_map

    # Cache missing or stale — fetch from network
    _cost_map = _fetch_and_cache()
    if _cost_map:
        return _cost_map

    # Network failed — try stale cache as a last resort
    _cost_map = _load_cache()
    if _cost_map:
        logger.info("Using stale cached pricing data (%d models)", len(_cost_map))
        return _cost_map

    # No data at all — fallback pricing will be used per-call
    logger.warning("No pricing data available — using conservative GPT-4o fallback rates")
    _cost_map = {}
    return _cost_map


def _lookup_model(model: str) -> tuple[float, float]:
    """
    Look up per-token costs for *model* in the pricing database.

    Tries an exact match first, then falls back to substring matching
    (longest match wins) so versioned names like ``gpt-4o-2024-11-20``
    resolve correctly.  Returns ``(input_cost_per_token, output_cost_per_token)``.
    """
    cost_map = _get_cost_map()
    key = model.lower()

    # 1. Exact match (fast path — covers the vast majority of cases)
    info = cost_map.get(key)
    if info and "input_cost_per_token" in info:
        return info["input_cost_per_token"], info["output_cost_per_token"]

    # 2. Substring match — try every known model that appears inside the key
    #    Sort by length descending so "gpt-4o-mini" beats "gpt-4o".
    matches = sorted(
        [(m, cost_map[m]) for m in cost_map if m in key and "input_cost_per_token" in cost_map[m]],
        key=lambda x: len(x[0]),
        reverse=True,
    )
    if matches:
        info = matches[0][1]
        return info["input_cost_per_token"], info["output_cost_per_token"]

    # 3. Fallback — conservative GPT-4o pricing
    return _DEFAULT_INPUT_COST_PER_TOKEN, _DEFAULT_OUTPUT_COST_PER_TOKEN


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate USD cost for a single LLM call.

    Uses the community-maintained LiteLLM pricing database (2,600+ models),
    fetched and cached automatically.  Falls back to GPT-4o pricing for
    unknown models (conservative estimate).
    """
    input_rate, output_rate = _lookup_model(model)
    return input_tokens * input_rate + output_tokens * output_rate


def get_jwt_token() -> str:
    """
    Generate a short-lived HS256 JWT for authenticating against the MCP servers.

    Token has a 1-hour expiry. MCP_JWT_SECRET must match what is configured
    in docker-compose.yml on the MCP server side.
    """
    secret = os.environ.get("MCP_JWT_SECRET", "")
    now = int(time.time())
    payload = {
        "sub": "agent",
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(payload, secret, algorithm="HS256")
