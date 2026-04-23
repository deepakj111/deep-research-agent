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

    async def call(self, coro: typing.Awaitable[typing.Any]) -> typing.Any:
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
        try:
            result = await coro
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            raise e


# Module-level singletons — one per tool
circuit_breakers: dict[str, CircuitBreaker] = {
    "search_web": CircuitBreaker("search_web", failure_threshold=3, recovery_timeout=60),
    "fetch_papers": CircuitBreaker("fetch_papers", failure_threshold=2, recovery_timeout=60),
    "search_repos": CircuitBreaker("search_repos", failure_threshold=2, recovery_timeout=60),
}
