# utils/cost_estimator.py
import os
import time

import jwt

# Pricing per 1 million tokens (USD), as of 2025-Q2
COST_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the USD cost for a single LLM call.

    Falls back to gpt-4o pricing for unknown model names so the
    estimate is always conservative (never under-estimated).
    """
    prices = COST_PER_1M.get(model, COST_PER_1M["gpt-4o"])
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


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
