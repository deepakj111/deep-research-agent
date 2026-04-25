# utils/cost_estimator.py
"""
Dynamic LLM cost estimation using the community-maintained LiteLLM pricing
database (https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

The pricing data is shipped as ``utils/model_prices.json`` and loaded lazily on
first call.  To refresh prices run::

    make update-model-prices

This approach avoids depending on the ``litellm`` package at runtime (which has
aggressive dependency pins that conflict with the rest of the stack) while still
providing dynamic, community-maintained pricing for 2,600+ models.
"""

import json
import logging
import os
import time
from pathlib import Path

import jwt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded pricing map
# ---------------------------------------------------------------------------

_cost_map: dict[str, dict] | None = None

_PRICES_PATH = Path(__file__).parent / "model_prices.json"

# Conservative fallback — matches gpt-4o so unknown models are never underestimated.
_DEFAULT_INPUT_COST_PER_TOKEN = 2.50 / 1_000_000
_DEFAULT_OUTPUT_COST_PER_TOKEN = 10.00 / 1_000_000


def _get_cost_map() -> dict[str, dict]:
    """Load the LiteLLM community pricing JSON (lazy, cached in-process)."""
    global _cost_map
    if _cost_map is None:
        try:
            with open(_PRICES_PATH, encoding="utf-8") as fh:
                _cost_map = json.load(fh)
            logger.debug("Loaded pricing data for %d models", len(_cost_map))
        except FileNotFoundError:
            logger.warning(
                "model_prices.json not found at %s — falling back to default pricing",
                _PRICES_PATH,
            )
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

    Uses the community-maintained LiteLLM pricing database shipped at
    ``utils/model_prices.json`` (2,600+ models).  Falls back to GPT-4o
    pricing for unknown models (conservative estimate).
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
