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
