# tests/unit/test_mcp_servers.py
import pytest


class TestCacheLayer:
    def test_cache_miss_returns_none(self, tmp_path):
        from mcp_servers.shared.cache import CacheLayer

        cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        assert cache.get("nonexistent_key") is None

    def test_cache_set_and_get(self, tmp_path):
        from mcp_servers.shared.cache import CacheLayer

        cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        cache.set("key1", [{"url": "https://example.com", "title": "Test"}])
        result = cache.get("key1")
        assert result is not None
        assert result[0]["url"] == "https://example.com"

    def test_cache_expired_returns_none(self, tmp_path):
        from mcp_servers.shared.cache import CacheLayer

        cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=-1)
        cache.set("expired_key", [{"data": "old"}])
        assert cache.get("expired_key") is None

    def test_cache_overwrite_with_same_key(self, tmp_path):
        from mcp_servers.shared.cache import CacheLayer

        cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        cache.set("key1", [{"v": 1}])
        cache.set("key1", [{"v": 2}])
        result = cache.get("key1")
        assert result is not None
        assert result[0]["v"] == 2

    def test_purge_expired_removes_old_rows(self, tmp_path):
        from mcp_servers.shared.cache import CacheLayer

        # Write a stale entry with ttl=-1 (immediately expired)
        stale_cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=-1)
        stale_cache.set("stale", [{"x": 1}])

        # Write a fresh entry with ttl=60
        fresh_cache = CacheLayer(db_path=str(tmp_path / "test.db"), ttl_seconds=60)
        fresh_cache.set("fresh", [{"x": 2}])
        fresh_cache.purge_expired()

        assert fresh_cache.get("stale") is None
        assert fresh_cache.get("fresh") is not None


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

    @pytest.mark.asyncio
    async def test_failure_count_resets_on_success(self):
        from agent.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)

        async def fail():
            raise RuntimeError("fail")

        async def succeed():
            return "ok"

        with pytest.raises(RuntimeError):
            await cb.call(fail())
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

        await cb.call(succeed())
        assert cb.failure_count == 0


class TestArxivParser:
    def test_parse_atom_returns_correct_fields(self):
        from mcp_servers.arxiv.server import _parse_atom

        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2401.00001v1</id>
            <title>Test Paper Title</title>
            <summary>This is the abstract of the test paper.</summary>
            <author><name>Alice Smith</name></author>
            <author><name>Bob Jones</name></author>
            <published>2024-01-15T00:00:00Z</published>
          </entry>
        </feed>"""

        papers = _parse_atom(sample_xml)
        assert len(papers) == 1
        p = papers[0]
        assert p["arxiv_id"] == "2401.00001v1"
        assert p["title"] == "Test Paper Title"
        assert p["authors"] == ["Alice Smith", "Bob Jones"]
        assert p["published_date"] == "2024-01-15"
        assert p["url"] == "http://arxiv.org/abs/2401.00001v1"
        assert p["citation_count"] == 0
        assert p["trust_score"] == 0.0

    def test_parse_atom_empty_feed_returns_empty_list(self):
        from mcp_servers.arxiv.server import _parse_atom

        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        papers = _parse_atom(sample_xml)
        assert papers == []


class TestGitHubServerNormalisation:
    def test_github_repo_dict_matches_model_fields(self):
        from agent.state import GitHubRepo

        repo_dict = {
            "name": "openai/gpt-4",
            "url": "https://github.com/openai/gpt-4",
            "description": "A large language model",
            "stars": 15000,
            "language": "Python",
            "last_updated": "2024-11-01",
            "trust_score": 0.0,
        }
        repo = GitHubRepo(**repo_dict)
        assert repo.name == "openai/gpt-4"
        assert repo.stars == 15000
        assert repo.language == "Python"
