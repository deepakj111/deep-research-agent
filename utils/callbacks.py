"""
utils/callbacks.py

LangChain callback handler that accumulates token usage and estimated USD cost
across multiple LLM calls within a single agent run.

Attach one TokenCostCallback instance to a run and read .summary when the
run completes.  Handles both OpenAI and Anthropic token-usage response shapes.

Usage::

    from utils.callbacks import TokenCostCallback

    cb = TokenCostCallback()
    result = await llm.ainvoke(prompt, config={"callbacks": [cb]})
    print(cb.summary)
    # {"total_input_tokens": 1234, "total_output_tokens": 567, "total_cost_usd": 0.003...}
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from utils.cost_estimator import estimate_cost


class TokenCostCallback(BaseCallbackHandler):
    """
    Accumulates token usage and estimated USD cost across LLM calls.

    Thread-safe for sequential use within an async event loop.  Do not share
    one instance across concurrent LLM calls unless you accept approximate
    counts (the adds are atomic for CPython's GIL, but ordering is not
    guaranteed under true parallelism).
    """

    def __init__(self) -> None:
        super().__init__()
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self._call_start_times: dict[str, float] = {}

    # ── LangChain hook overrides ──────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._call_start_times[str(run_id)] = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._call_start_times.pop(str(run_id), None)
        input_tokens, output_tokens = self._extract_token_counts(response)
        model = self._extract_model_name(response, kwargs)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += estimate_cost(model, input_tokens, output_tokens)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # Ensure start time is cleaned up even on error
        self._call_start_times.pop(str(run_id), None)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all counters — useful when reusing an instance across runs."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self._call_start_times.clear()

    @property
    def summary(self) -> dict[str, Any]:
        """Return a snapshot of accumulated usage metrics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_token_counts(response: LLMResult) -> tuple[int, int]:
        """
        Extract (input_tokens, output_tokens) from an LLMResult.

        Handles three common response shapes:
          • OpenAI: llm_output["token_usage"]["prompt_tokens"] / "completion_tokens"
          • Anthropic: llm_output["usage"]["input_tokens"] / "output_tokens"
          • Fallback: generation-level generation_info (some older adapters)
        """
        usage: dict[str, Any] = {}

        # Primary: OpenAI / most providers
        if response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}

        # Fallback: check generation-level metadata
        if not usage and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        usage = gen.generation_info.get("usage", {})
                        if usage:
                            break
                if usage:
                    break

        # OpenAI keys
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Anthropic keys (override if present and non-zero)
        if not input_tokens:
            input_tokens = usage.get("input_tokens", 0)
        if not output_tokens:
            output_tokens = usage.get("output_tokens", 0)

        return int(input_tokens), int(output_tokens)

    @staticmethod
    def _extract_model_name(response: LLMResult, kwargs: dict[str, Any]) -> str:
        """Best-effort model name extraction for cost lookup."""
        if response.llm_output:
            name = response.llm_output.get("model_name") or response.llm_output.get("model", "")
            if name:
                return name
        # Some providers embed model name in invocation params
        invocation = kwargs.get("invocation_params", {})
        return invocation.get("model", invocation.get("model_name", "gpt-4o"))
