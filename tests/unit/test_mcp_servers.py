# tests/unit/test_mcp_servers.py
import pytest


class TestCacheLayer:
    def test_cache_miss_returns_none(self, tmp_path):
        from mcp_servers.web_search.cache import CacheLayer

        cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        assert cache.get("nonexistent_key") is None

    def test_cache_set_and_get(self, tmp_path):
        from mcp_servers.web_search.cache import CacheLayer

        cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        cache.set("key1", [{"url": "https://example.com", "title": "Test"}])
        result = cache.get("key1")
        assert result is not None
        assert result[0]["url"] == "https://example.com"

    def test_cache_expired_returns_none(self, tmp_path):
        from mcp_servers.web_search.cache import CacheLayer

        cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=-1)
        cache.set("expired_key", [{"data": "old"}])
        assert cache.get("expired_key") is None


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_closed_state_passes_through(self):
        from agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)

        async def mock_coro():
            return "success"

        result = await cb.call(mock_coro())
        assert result == "success"

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        from agent.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=60)

        async def failing_coro():
            raise RuntimeError("tool failure")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(failing_coro())

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_open_raises_immediately(self):
        from agent.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=9999)

        async def failing_coro():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await cb.call(failing_coro())

        coro = failing_coro()
        with pytest.raises(RuntimeError, match="OPEN"):
            await cb.call(coro)
        coro.close()
