# utils/cost_estimator.py
import os
import time

import jwt

# Static pricing table (USD per 1M tokens, input / output).
# Update this when providers change rates — no external dependency needed.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
}

# Conservative fallback — matches gpt-4o so unknown models are never underestimated.
_DEFAULT_INPUT_PER_1M = 2.50
_DEFAULT_OUTPUT_PER_1M = 10.00


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate USD cost for a single LLM call using a static pricing table.

    Matches on a case-insensitive substring of the model name so that
    versioned suffixes (e.g. 'gpt-4o-2024-11-20') still resolve correctly.
    Falls back to GPT-4o pricing for unknown models (conservative estimate).
    """
    key = model.lower()
    input_rate, output_rate = next(
        (rates for model_id, rates in _MODEL_PRICING.items() if model_id in key),
        (_DEFAULT_INPUT_PER_1M, _DEFAULT_OUTPUT_PER_1M),
    )
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


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
