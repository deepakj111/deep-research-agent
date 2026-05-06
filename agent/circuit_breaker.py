"""
agent/circuit_breaker.py

Async-safe circuit breaker for MCP tool calls.

Each tool gets its own CircuitBreaker instance with an asyncio.Lock
to protect mutable state (failure_count, state, last_failure_time)
from concurrent access during LangGraph's Send-based parallel fan-out.

State transitions follow the standard closed → open → half-open pattern:
  - CLOSED: calls proceed normally; failures are counted.
  - OPEN:   calls are rejected immediately until recovery_timeout elapses.
  - HALF_OPEN: one probe call is allowed; success → CLOSED, failure → OPEN.
"""

import asyncio
import time
import typing
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
    ) -> None:
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time: float | None = None
        # Serialize state reads/writes across concurrent coroutines.
        # LangGraph's Send API dispatches multiple agents in parallel,
        # so without this lock, failure_count increments can race.
        self._lock = asyncio.Lock()

    async def call(self, coro: typing.Awaitable[typing.Any]) -> typing.Any:
        # ── Pre-call gate (under lock) ────────────────────────────────────
        async with self._lock:
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - (self.last_failure_time or 0)
                if elapsed > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                else:
                    remaining = self.recovery_timeout - elapsed
                    raise RuntimeError(
                        f"CircuitBreaker[{self.name}] OPEN — skipping tool. "
                        f"Will retry in {remaining:.0f}s"
                    )

        # ── Actual call (outside lock — don't hold it during I/O) ─────────
        try:
            result = await coro
        except Exception:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
            raise

        # ── Success path (under lock) ─────────────────────────────────────
        async with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
        return result


# Module-level singletons — one per tool
circuit_breakers: dict[str, CircuitBreaker] = {
    "search_web": CircuitBreaker("search_web", failure_threshold=3, recovery_timeout=60),
    "fetch_papers": CircuitBreaker("fetch_papers", failure_threshold=2, recovery_timeout=60),
    "search_repos": CircuitBreaker("search_repos", failure_threshold=2, recovery_timeout=60),
}
