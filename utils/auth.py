# utils/auth.py
"""
Authentication utilities for the DeepResearch Agent.
"""

from __future__ import annotations

import os
import time

import jwt


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
