import os
import time

import jwt

# GPT-4o pricing (per 1M tokens, as of 2025)
COST_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = COST_PER_1M.get(model, {"input": 2.50, "output": 10.00})
    return (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices[
        "output"
    ]


def get_jwt_token() -> str:
    secret = os.environ.get("MCP_JWT_SECRET", "")
    payload = {"sub": "agent", "iat": int(time.time()), "exp": int(time.time()) + 3600}
    return jwt.encode(payload, secret, algorithm="HS256")
