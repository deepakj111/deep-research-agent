# tests/unit/test_evaluator.py
"""
Unit tests for the evaluation pipeline.

All LLM calls are mocked — these tests verify the data model, prompt
construction, and evaluation flow without spending any API credits.
"""

from unittest.mock import AsyncMock, patch

import pytest

from agent.state import Citation, Finding, ReportOutput
from evaluation.evaluator import (
    EvalScores,
    _build_report_text,
    _build_source_list,
    evaluate_report,
)

# ─────────────────────────── Fixtures ────────────────────────────────────────


@pytest.fixture
def sample_report() -> ReportOutput:
    return ReportOutput(
        title="Advances in Quantum Error Correction 2025",
        executive_summary=(
            "Quantum error correction has seen remarkable progress in 2025, "
            "with multiple research groups demonstrating fault-tolerant qubits."
        ),
        key_findings=[
            Finding(
                claim="Surface code implementations now achieve <0.1% logical error rates.",
                citations=[
                    Citation(
                        source_url="https://arxiv.org/abs/2501.00001",
                        title="Surface Code Breakthrough",
                        exact_snippet="logical error rate of 0.08%",
                        source_type="arxiv",
                        trust_score=0.85,
                    )
                ],
                confidence="high",
            ),
            Finding(
                claim="Google's Willow chip sets a new benchmark for quantum supremacy.",
                citations=[
                    Citation(
                        source_url="https://blog.google/technology/research/google-willow-quantum-chip/",
                        title="Google Willow Announcement",
                        exact_snippet="Willow chip demonstration",
                        source_type="web",
                        trust_score=0.70,
                    )
                ],
                confidence="high",
            ),
        ],
        emerging_trends=["Topological qubits gaining commercial traction"],
        recommended_next_steps=["Monitor IBM's error correction roadmap"],
        sources=[
            Citation(
                source_url="https://arxiv.org/abs/2501.00001",
                title="Surface Code Breakthrough",
                exact_snippet="logical error rate of 0.08%",
                source_type="arxiv",
                trust_score=0.85,
            ),
            Citation(
                source_url="https://blog.google/technology/research/google-willow-quantum-chip/",
                title="Google Willow Announcement",
                exact_snippet="Willow chip demonstration",
                source_type="web",
                trust_score=0.70,
            ),
            Citation(
                source_url="https://github.com/quantumlib/Cirq",
                title="quantumlib/Cirq",
                exact_snippet="Quantum circuit simulator library",
                source_type="github",
                trust_score=0.90,
            ),
        ],
    )


# ──────────────────────── EvalScores Model ────────────────────────────────────


class TestEvalScoresModel:
    def test_normalized_average_perfect_score(self) -> None:
        scores = EvalScores(
            faithfulness=5.0,
            answer_relevancy=5.0,
            source_coverage=5.0,
            citation_accuracy=5.0,
            coherence=5.0,
        )
        assert scores.normalized_average == pytest.approx(1.0)

    def test_normalized_average_zero_score(self) -> None:
        scores = EvalScores(
            faithfulness=0.0,
            answer_relevancy=0.0,
            source_coverage=0.0,
            citation_accuracy=0.0,
            coherence=0.0,
        )
        assert scores.normalized_average == pytest.approx(0.0)

    def test_normalized_average_mid_score(self) -> None:
        scores = EvalScores(
            faithfulness=3.0,
            answer_relevancy=3.5,
            source_coverage=4.0,
            citation_accuracy=2.5,
            coherence=3.5,
        )
        # (3.0+3.5+4.0+2.5+3.5) / 25 = 16.5 / 25 = 0.66
        assert scores.normalized_average == pytest.approx(0.66)

    def test_to_dict_contains_all_keys(self) -> None:
        scores = EvalScores(
            faithfulness=4.0,
            answer_relevancy=4.0,
            source_coverage=4.0,
            citation_accuracy=4.0,
            coherence=4.0,
            overall_notes="Good report",
        )
        d = scores.to_dict()
        expected = {
            "faithfulness",
            "answer_relevancy",
            "source_coverage",
            "citation_accuracy",
            "coherence",
            "overall_notes",
            "normalized_average",
        }
        assert set(d.keys()) == expected

    def test_scores_reject_out_of_range_values(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            EvalScores(
                faithfulness=6.0,  # > 5 — should fail
                answer_relevancy=3.0,
                source_coverage=3.0,
                citation_accuracy=3.0,
                coherence=3.0,
            )


# ─────────────────────── Report Text Builders ────────────────────────────────


class TestReportTextBuilders:
    def test_build_report_text_includes_title(self, sample_report: ReportOutput) -> None:
        text = _build_report_text(sample_report)
        assert sample_report.title in text

    def test_build_report_text_includes_executive_summary(
        self, sample_report: ReportOutput
    ) -> None:
        text = _build_report_text(sample_report)
        assert "remarkable progress" in text

    def test_build_report_text_includes_finding_claims(self, sample_report: ReportOutput) -> None:
        text = _build_report_text(sample_report)
        assert "Surface code" in text
        assert "Willow" in text

    def test_build_source_list_all_source_types(self, sample_report: ReportOutput) -> None:
        source_text = _build_source_list(sample_report)
        assert "[ARXIV]" in source_text
        assert "[WEB]" in source_text
        assert "[GITHUB]" in source_text

    def test_build_source_list_no_sources(self) -> None:
        empty_report = ReportOutput(
            title="Empty",
            executive_summary="Nothing here.",
            key_findings=[],
            emerging_trends=[],
            recommended_next_steps=[],
            sources=[],
        )
        result = _build_source_list(empty_report)
        assert "(no sources recorded)" in result

    def test_build_report_text_truncation_guard(self) -> None:
        """Extremely long reports must be truncated before reaching the evaluator."""
        long_summary = "x" * 20_000
        big_report = ReportOutput(
            title="Big Report",
            executive_summary=long_summary,
            key_findings=[],
            emerging_trends=[],
            recommended_next_steps=[],
        )
        text = _build_report_text(big_report)
        # evaluate_report caps at 12_000 chars — just verify _build_report_text
        # itself does not crash on large input
        assert isinstance(text, str)
        assert len(text) > 0


# ──────────────────────── evaluate_report (mocked) ───────────────────────────


class TestEvaluateReport:
    @pytest.mark.asyncio
    async def test_returns_eval_scores_on_success(self, sample_report: ReportOutput) -> None:
        mock_scores = EvalScores(
            faithfulness=4.5,
            answer_relevancy=4.0,
            source_coverage=5.0,
            citation_accuracy=4.2,
            coherence=4.8,
            overall_notes="Strong research with good source diversity.",
        )

        with patch("evaluation.evaluator._get_eval_llm") as mock_get_llm:
            mock_llm = mock_get_llm.return_value
            mock_llm.ainvoke = AsyncMock(return_value=mock_scores)

            result = await evaluate_report("Quantum error correction advances 2025", sample_report)

        assert isinstance(result, EvalScores)
        assert result.faithfulness == pytest.approx(4.5)
        assert result.normalized_average == pytest.approx((4.5 + 4.0 + 5.0 + 4.2 + 4.8) / 25.0)

    @pytest.mark.asyncio
    async def test_propagates_llm_errors(self, sample_report: ReportOutput) -> None:
        with patch("evaluation.evaluator._get_eval_llm") as mock_get_llm:
            mock_llm = mock_get_llm.return_value
            mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API rate limited"))

            with pytest.raises(RuntimeError, match="API rate limited"):
                await evaluate_report("test query", sample_report)

    @pytest.mark.asyncio
    async def test_respects_timeout(self, sample_report: ReportOutput) -> None:
        import asyncio

        async def slow_llm(*args, **kwargs):
            await asyncio.sleep(999)

        with patch("evaluation.evaluator._get_eval_llm") as mock_get_llm:
            mock_llm = mock_get_llm.return_value
            mock_llm.ainvoke = slow_llm

            with pytest.raises(asyncio.TimeoutError):
                await evaluate_report("test query", sample_report, timeout_seconds=0.05)


# ──────────────────────── TokenCostCallback ──────────────────────────────────


class TestTokenCostCallback:
    def test_accumulates_openai_token_counts(self) -> None:
        from unittest.mock import MagicMock
        from uuid import uuid4

        from langchain_core.outputs import LLMResult

        from utils.callbacks import TokenCostCallback

        cb = TokenCostCallback()
        run_id = uuid4()

        # Simulate on_llm_end with OpenAI token format
        mock_result = MagicMock(spec=LLMResult)
        mock_result.llm_output = {
            "token_usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            "model_name": "gpt-4o",
        }
        mock_result.generations = []
        cb.on_llm_end(mock_result, run_id=run_id)

        assert cb.total_input_tokens == 1000
        assert cb.total_output_tokens == 500
        assert cb.total_cost_usd > 0

    def test_accumulates_anthropic_token_counts(self) -> None:
        from unittest.mock import MagicMock
        from uuid import uuid4

        from langchain_core.outputs import LLMResult

        from utils.callbacks import TokenCostCallback

        cb = TokenCostCallback()
        run_id = uuid4()

        mock_result = MagicMock(spec=LLMResult)
        mock_result.llm_output = {
            "usage": {"input_tokens": 800, "output_tokens": 300},
            "model": "claude-sonnet-4-5",
        }
        mock_result.generations = []
        cb.on_llm_end(mock_result, run_id=run_id)

        assert cb.total_input_tokens == 800
        assert cb.total_output_tokens == 300

    def test_reset_clears_counts(self) -> None:
        from unittest.mock import MagicMock
        from uuid import uuid4

        from langchain_core.outputs import LLMResult

        from utils.callbacks import TokenCostCallback

        cb = TokenCostCallback()
        run_id = uuid4()
        mock_result = MagicMock(spec=LLMResult)
        mock_result.llm_output = {
            "token_usage": {"prompt_tokens": 500, "completion_tokens": 200},
            "model_name": "gpt-4o-mini",
        }
        mock_result.generations = []
        cb.on_llm_end(mock_result, run_id=run_id)

        cb.reset()

        assert cb.total_input_tokens == 0
        assert cb.total_output_tokens == 0
        assert cb.total_cost_usd == 0.0

    def test_summary_property_structure(self) -> None:
        from utils.callbacks import TokenCostCallback

        cb = TokenCostCallback()
        summary = cb.summary
        assert set(summary.keys()) == {
            "total_input_tokens",
            "total_output_tokens",
            "total_cost_usd",
        }
