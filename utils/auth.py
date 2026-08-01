# utils/auth.py
"""
Authentication utilities for the DeepResearch Agent.
"""

from __future__ import annotations

import time

import jwt


def get_jwt_token() -> str:
    """
    Generate a short-lived HS256 JWT for authenticating against the MCP servers.

    Token has a 1-hour expiry. MCP_JWT_SECRET must match what is configured
    in docker-compose.yml on the MCP server side.
    """
    import os

    from config.settings import settings

    secret = (
        os.environ.get("MCP_JWT_SECRET")
        or settings.mcp_jwt_secret
        or "deep-research-agent-mcp-jwt-secret-key-2026"
    )
    now = int(time.time())
    payload = {
        "sub": "agent",
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(payload, secret, algorithm="HS256")
