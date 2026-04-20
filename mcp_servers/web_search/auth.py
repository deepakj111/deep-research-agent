import os
from functools import wraps

import jwt

SECRET = os.environ.get("MCP_JWT_SECRET", "")


def require_auth(func):
    @wraps(func)
    async def wrapper(ctx, *args, **kwargs):
        token = ctx.request_context.headers.get("Authorization", "").replace("Bearer ", "").strip()
        try:
            jwt.decode(token, SECRET, algorithms=["HS256"])
        except jwt.InvalidTokenError as e:
            raise PermissionError(f"Invalid or expired JWT token: {e}") from e
        return await func(ctx, *args, **kwargs)

    return wrapper
