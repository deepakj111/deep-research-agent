# tests/unit/test_agent_nodes.py
import pytest

from agent.state import (
    CritiqueOutput,
    ResearchFindings,
    ResearchState,
    RunMetadata,
    WebResult,
)


@pytest.fixture
def base_state():
    return ResearchState(
        query="Latest advances in quantum computing 2025",
        profile="fast",
        run_id="unit-test-001",
        query_difficulty="narrow",
        subquestions=["What are the latest quantum computing breakthroughs?"],
        approved_plan=True,
        findings=[],
        critique=None,
        iteration_count=0,
        final_report=None,
        run_metadata=RunMetadata(run_id="unit-test-001", profile="fast"),
        error_log=[],
        thought_log=[],
    )


class TestCriticNode:
    def test_should_continue_returns_synthesize_when_no_critique(self, base_state):
        from agent.nodes.critic import should_continue

        assert should_continue(base_state) == "synthesize"

    def test_should_continue_returns_synthesize_when_max_iterations(self, base_state):
        from agent.nodes.critic import should_continue

        base_state["run_metadata"] = RunMetadata(run_id="test", profile="fast", iteration_count=15)
        assert should_continue(base_state) == "synthesize"

    def test_should_continue_loops_when_critique_says_so(self, base_state):
        from agent.nodes.critic import should_continue

        base_state["critique"] = CritiqueOutput(
            coverage_score=0.4,
            recency_score=0.5,
            depth_score=0.5,
            source_diversity_score=0.3,
            missing_areas=["academic papers missing"],
            should_continue=True,
            reasoning="Coverage is insufficient",
        )
        base_state["run_metadata"] = RunMetadata(run_id="test", profile="fast", iteration_count=2)
        assert should_continue(base_state) == "continue"


class TestWriterNode:
    @pytest.mark.asyncio
    async def test_writer_returns_error_log_when_no_report(self, base_state):
        from agent.nodes.writer import run

        result = await run(base_state)
        assert "error_log" in result
        assert len(result["error_log"]) > 0

    @pytest.mark.asyncio
    async def test_writer_attaches_citations(self, base_state):
        from agent.nodes.writer import run
        from agent.state import Citation, Finding, ReportOutput

        report = ReportOutput(
            title="Test Report",
            executive_summary="A test summary.",
            key_findings=[
                Finding(
                    claim="Quantum computing advanced significantly",
                    citations=[
                        Citation(
                            source_url="https://arxiv.org/test",
                            title="Test Paper",
                            exact_snippet="snippet",
                            source_type="arxiv",
                            trust_score=0.8,
                        )
                    ],
                    confidence="high",
                )
            ],
            emerging_trends=["Quantum error correction improving"],
            recommended_next_steps=["Monitor arXiv for new papers"],
        )
        base_state["final_report"] = report
        base_state["findings"] = [
            ResearchFindings(
                subquestion="test",
                web_results=[WebResult(url="https://example.com", title="Test", snippet="snippet")],
            )
        ]
        result = await run(base_state)
        assert result["final_report"].sources is not None
        assert len(result["final_report"].sources) > 0


class TestSupervisorNode:
    @pytest.mark.asyncio
    async def test_fan_out_produces_correct_send_count(self, base_state):
        from langgraph.types import Command, Send

        from agent.nodes.supervisor import run

        base_state["subquestions"] = ["Q1", "Q2", "Q3"]
        base_state["findings"] = []

        result = await run(base_state)
        assert isinstance(result, Command)
        # 3 subquestions x 3 agents = 9 Send objects
        assert isinstance(result.goto, list)
        assert len(result.goto) == 9
        assert all(isinstance(s, Send) for s in result.goto)

    @pytest.mark.asyncio
    async def test_empty_subquestions_goes_to_critic(self, base_state):
        from langgraph.types import Command

        from agent.nodes.supervisor import run

        base_state["subquestions"] = []
        result = await run(base_state)
        assert isinstance(result, Command)
        assert result.goto == "critic"


class TestCostEstimator:
    """Tests for the dynamic cost estimator backed by LiteLLM community pricing."""

    # Deterministic test pricing data (matches real LiteLLM pricing values)
    _TEST_COST_MAP: dict = {
        "gpt-4o": {"input_cost_per_token": 2.5e-06, "output_cost_per_token": 1e-05},
        "gpt-4o-mini": {"input_cost_per_token": 1.5e-07, "output_cost_per_token": 6e-07},
        "claude-sonnet-4-5": {"input_cost_per_token": 3e-06, "output_cost_per_token": 1.5e-05},
    }

    def setup_method(self):
        """Reset the cached cost map and inject test data."""
        import utils.cost_estimator as mod

        mod._cost_map = self._TEST_COST_MAP.copy()

    def teardown_method(self):
        """Clear the cost map after each test."""
        import utils.cost_estimator as mod

        mod._cost_map = None

    def test_gpt4o_cost_calculation(self):
        from utils.cost_estimator import estimate_cost

        cost = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == 12.50

    def test_gpt4o_mini_cost_calculation(self):
        from utils.cost_estimator import estimate_cost

        cost = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        assert cost == 0.75

    def test_unknown_model_falls_back_to_gpt4o_pricing(self):
        from utils.cost_estimator import estimate_cost

        cost = estimate_cost("unknown-model", 1_000_000, 0)
        assert cost == 2.50

    def test_claude_sonnet_cost_calculation(self):
        from utils.cost_estimator import estimate_cost

        cost = estimate_cost("claude-sonnet-4-5", 1_000_000, 1_000_000)
        assert cost == 18.00

    def test_fetch_and_cache_populates_cost_map(self, monkeypatch):
        """Verify the fetch-and-cache mechanism works with a mocked HTTP response."""

        import utils.cost_estimator as mod

        # Reset to force a fresh fetch
        mod._cost_map = None

        # Mock _cache_is_fresh to return False (force network fetch)
        monkeypatch.setattr(mod, "_cache_is_fresh", lambda: False)

        # Mock _fetch_and_cache to return test data (no real HTTP call)
        mock_data = {
            **self._TEST_COST_MAP,
            "extra-model": {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06},
        }
        monkeypatch.setattr(mod, "_fetch_and_cache", lambda: mock_data)

        cost_map = mod._get_cost_map()
        assert len(cost_map) == 4
        assert "extra-model" in cost_map


class TestReportFormatter:
    def test_to_markdown_contains_title(self):
        from agent.state import Citation, Finding, ReportOutput
        from utils.report_formatter import to_markdown

        report = ReportOutput(
            title="Test Report Title",
            executive_summary="Summary here.",
            key_findings=[
                Finding(
                    claim="Finding one",
                    citations=[
                        Citation(
                            source_url="https://example.com",
                            title="Source",
                            exact_snippet="snippet",
                            source_type="web",
                            trust_score=0.7,
                        )
                    ],
                    confidence="high",
                )
            ],
            emerging_trends=["Trend A"],
            recommended_next_steps=["Do X"],
        )
        md = to_markdown(report)
        assert "# Test Report Title" in md
        assert "## Executive Summary" in md
        assert "## Key Findings" in md
        assert "Trend A" in md


class TestSynthesizerEdgeCases:
    @pytest.mark.asyncio
    async def test_synthesizer_returns_none_report_when_both_models_fail(self, base_state):
        from unittest.mock import AsyncMock, patch

        from agent.nodes.synthesizer import run

        base_state["findings"] = []

        with (
            patch("agent.nodes.synthesizer._gpt4o") as mock_gpt,
            patch("agent.nodes.synthesizer._claude") as mock_claude,
        ):
            mock_gpt.ainvoke = AsyncMock(side_effect=RuntimeError("GPT failed"))
            mock_claude.ainvoke = AsyncMock(side_effect=RuntimeError("Claude failed"))

            result = await run(base_state)

            # When both models fail, final_report must be None
            assert result["final_report"] is None
            assert "error_log" in result
            assert len(result["error_log"]) > 0
            assert "0/2" in result["thought_log"][0]

    @pytest.mark.asyncio
    async def test_synthesizer_uses_fallback_when_one_model_fails(self, base_state):
        from unittest.mock import AsyncMock, patch

        from agent.nodes.synthesizer import run
        from agent.state import ReportOutput

        base_state["findings"] = []

        fallback_report = ReportOutput(
            title="Fallback",
            executive_summary="Summary",
            key_findings=[],
            emerging_trends=[],
            recommended_next_steps=[],
        )

        with (
            patch("agent.nodes.synthesizer._gpt4o") as mock_gpt,
            patch("agent.nodes.synthesizer._claude") as mock_claude,
        ):
            mock_gpt.ainvoke = AsyncMock(side_effect=RuntimeError("GPT failed"))
            mock_claude.ainvoke = AsyncMock(return_value=fallback_report)

            result = await run(base_state)

            assert result["final_report"] is not None
            assert result["final_report"].title == "Fallback"
            assert "1/2" in result["thought_log"][0]
