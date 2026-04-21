# tests/unit/test_retry_policy.py
"""
Unit tests for the per-tool retry policy module.

Tests cover:
  - Successful call (no retries needed)
  - Exponential vs linear backoff delay computation
  - Max retries exhausted with fallback=None (raises original exception)
  - Max retries exhausted with fallback="skip_and_note" (raises ToolDegradedError)
  - Retry succeeds on second attempt
  - Unknown tool falls back to default policy
"""

import asyncio

import pytest

from agent.retry_policy import (
    ToolDegradedError,
    ToolErrorPolicy,
    _compute_delay,
    retry_with_policy,
)


class TestComputeDelay:
    def test_exponential_backoff_attempt_0(self) -> None:
        policy = ToolErrorPolicy(backoff="exponential", base_delay_seconds=1.0)
        assert _compute_delay(policy, 0) == 1.0

    def test_exponential_backoff_attempt_1(self) -> None:
        policy = ToolErrorPolicy(backoff="exponential", base_delay_seconds=1.0)
        assert _compute_delay(policy, 1) == 2.0

    def test_exponential_backoff_attempt_2(self) -> None:
        policy = ToolErrorPolicy(backoff="exponential", base_delay_seconds=1.0)
        assert _compute_delay(policy, 2) == 4.0

    def test_linear_backoff_is_constant(self) -> None:
        policy = ToolErrorPolicy(backoff="linear", base_delay_seconds=2.0)
        assert _compute_delay(policy, 0) == 2.0
        assert _compute_delay(policy, 1) == 2.0
        assert _compute_delay(policy, 5) == 2.0


class TestRetryWithPolicy:
    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_policy("search_web", _factory)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, monkeypatch) -> None:
        # Patch asyncio.sleep to avoid real delays in tests
        async def _noop_sleep(_):
            pass

        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient failure")
            return "recovered"

        result = await retry_with_policy("search_web", _factory)
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_original(self, monkeypatch) -> None:
        async def _noop_sleep(_):
            pass

        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        async def _factory():
            raise ValueError("persistent failure")

        # search_web has fallback=None → should raise the original ValueError
        with pytest.raises(ValueError, match="persistent failure"):
            await retry_with_policy("search_web", _factory)

    @pytest.mark.asyncio
    async def test_degraded_fallback_raises_tool_degraded_error(self, monkeypatch) -> None:
        async def _noop_sleep(_):
            pass

        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        async def _factory():
            raise ConnectionError("arXiv down")

        # fetch_papers has fallback="skip_and_note"
        with pytest.raises(ToolDegradedError) as exc_info:
            await retry_with_policy("fetch_papers", _factory)

        assert exc_info.value.tool_name == "fetch_papers"
        assert "arXiv unavailable" in exc_info.value.failure_note

    @pytest.mark.asyncio
    async def test_unknown_tool_uses_default_policy(self, monkeypatch) -> None:
        async def _noop_sleep(_):
            pass

        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        call_count = 0

        async def _factory():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("fail")
            return "ok"

        # Default policy has max_retries=2, fallback=None
        result = await retry_with_policy("unknown_tool", _factory)
        assert result == "ok"
        assert call_count == 3  # initial + 2 retries


class TestToolDegradedError:
    def test_error_attributes(self) -> None:
        cause = RuntimeError("root cause")
        err = ToolDegradedError("fetch_papers", "arXiv unavailable", cause)
        assert err.tool_name == "fetch_papers"
        assert err.failure_note == "arXiv unavailable"
        assert err.cause is cause
        assert "fetch_papers" in str(err)
