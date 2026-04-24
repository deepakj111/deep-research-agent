# utils/cost_estimator.py
import os
import time

import jwt


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the USD cost for a single LLM call.

    Uses `litellm` to dynamically fetch live API costs.
    Falls back to conservative generic pricing (e.g., $2.5/$10) if the model is not found in litellm's registry.
    """
    import contextlib

    from litellm import cost_per_token

    with contextlib.suppress(Exception):
        cost_tuple = cost_per_token(
            model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
        if cost_tuple:
            return float(cost_tuple[0])

    # Fallback conservative pricing if litellm doesn't recognize the model or there's an error
    input_cost = (input_tokens / 1_000_000) * 2.50
    output_cost = (output_tokens / 1_000_000) * 10.00
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
