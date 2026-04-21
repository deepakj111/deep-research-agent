"""
agent/retry_policy.py

Per-tool retry policies with configurable backoff strategies.

Each MCP tool has an explicit error policy defining:
  - max_retries: how many times to retry before giving up
  - backoff: "exponential" (base 2) or "linear" (fixed delay)
  - base_delay_seconds: initial delay between retries
  - fallback: what to do when all retries are exhausted
      - None: raise the exception (tool is critical)
      - "skip_and_note": swallow the error, return a degradation note
  - failure_note: human-readable message for the report when the tool fails

The retry_with_policy() coroutine wraps any async callable and applies
the appropriate policy. It composes with CircuitBreaker — retries happen
*inside* the circuit breaker window.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolErrorPolicy:
    """Immutable error policy for a single MCP tool."""

    max_retries: int = 2
    backoff: str = "exponential"  # "exponential" | "linear"
    base_delay_seconds: float = 1.0
    fallback: str | None = None  # None | "skip_and_note"
    failure_note: str = "[Tool unavailable]"


# ─────────────────────── Policy Registry ─────────────────────────────────────

TOOL_ERROR_POLICIES: dict[str, ToolErrorPolicy] = {
    "search_web": ToolErrorPolicy(
        max_retries=3,
        backoff="exponential",
        base_delay_seconds=1.0,
        fallback=None,  # web search is critical — propagate failure
        failure_note="[Web search unavailable for this sub-question]",
    ),
    "fetch_papers": ToolErrorPolicy(
        max_retries=2,
        backoff="linear",
        base_delay_seconds=2.0,
        fallback="skip_and_note",
        failure_note="[arXiv unavailable — academic sources omitted from this section]",
    ),
    "search_repos": ToolErrorPolicy(
        max_retries=2,
        backoff="linear",
        base_delay_seconds=2.0,
        fallback="skip_and_note",
        failure_note="[GitHub unavailable — code references omitted from this section]",
    ),
}


def _compute_delay(policy: ToolErrorPolicy, attempt: int) -> float:
    """Compute delay in seconds for the given attempt number (0-indexed)."""
    if policy.backoff == "exponential":
        return policy.base_delay_seconds * (2**attempt)
    # linear
    return policy.base_delay_seconds


async def retry_with_policy(
    tool_name: str,
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
) -> Any:
    """
    Execute a tool call with retry and backoff according to its error policy.

    Args:
        tool_name:     Key into TOOL_ERROR_POLICIES.
        coro_factory:  A zero-arg callable that returns a fresh coroutine on each
                       invocation. Must create a new coroutine per retry — you
                       cannot re-await the same coroutine.

    Returns:
        The tool result on success.

    Raises:
        The last exception if all retries exhausted and fallback is None.
        ToolDegradedError if fallback is "skip_and_note".
    """
    policy = TOOL_ERROR_POLICIES.get(tool_name, ToolErrorPolicy())
    last_error: Exception | None = None

    for attempt in range(policy.max_retries + 1):  # +1 for the initial attempt
        try:
            return await coro_factory()
        except Exception as exc:
            last_error = exc
            if attempt < policy.max_retries:
                delay = _compute_delay(policy, attempt)
                logger.warning(
                    "[RetryPolicy:%s] Attempt %d/%d failed (%s). Retrying in %.1fs...",
                    tool_name,
                    attempt + 1,
                    policy.max_retries + 1,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

    # All retries exhausted
    if policy.fallback == "skip_and_note":
        raise ToolDegradedError(tool_name, policy.failure_note, last_error)

    # No fallback — propagate the original exception
    raise last_error  # type: ignore[misc]


class ToolDegradedError(Exception):
    """Raised when a non-critical tool fails and the pipeline should degrade gracefully."""

    def __init__(
        self,
        tool_name: str,
        failure_note: str,
        cause: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.failure_note = failure_note
        self.cause = cause
        super().__init__(f"[{tool_name}] {failure_note}")
