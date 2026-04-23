# mcp_servers/web_search/auth.py
import os
import typing
from functools import wraps

import jwt

SECRET = os.environ.get("MCP_JWT_SECRET", "")


def require_auth(
    func: typing.Callable[..., typing.Awaitable[typing.Any]],
) -> typing.Callable[..., typing.Awaitable[typing.Any]]:
    """
    FastMCP tool decorator that validates the incoming JWT Bearer token.

    FastMCP passes a `ctx` (Context) object as the first positional argument
    to every tool that declares it. The Authorization header is read from
    ctx.request_context.request.headers (Starlette Request object).

    If the token is missing or invalid, raises PermissionError which FastMCP
    surfaces as an MCP error response — the agent node catches this and writes
    it to tool_errors so the pipeline degrades gracefully.
    """

    @wraps(func)
    async def wrapper(ctx: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        try:
            raw = ctx.request_context.request.headers.get("Authorization", "")
        except AttributeError:
            raw = ""

        token = raw.replace("Bearer ", "").strip()

        if not token:
            raise PermissionError("Missing Authorization header — Bearer token required.")

        try:
            jwt.decode(token, SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError as e:
            raise PermissionError("JWT token has expired.") from e
        except jwt.InvalidTokenError as e:
            raise PermissionError(f"Invalid JWT token: {e}") from e

        return await func(ctx, *args, **kwargs)

    return wrapper
