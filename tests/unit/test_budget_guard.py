# tests/unit/test_budget_guard.py
"""
Unit tests for the budget guard module.

Tests cover:
  - Budget OK → delegates to critic's decision
  - Iteration limit exceeded → forces synthesis
  - Cost limit exceeded → forces synthesis
  - Missing metadata → falls back to critic
"""

import pytest

from agent.state import CritiqueOutput, ResearchState, RunMetadata


@pytest.fixture
def base_state():
    return ResearchState(
        query="Test query",
        profile="fast",
        run_id="budget-test-001",
        query_difficulty="narrow",
        subquestions=["Q1"],
        approved_plan=True,
        findings=[],
        critique=None,
        iteration_count=0,
        final_report=None,
        run_metadata=RunMetadata(run_id="budget-test-001", profile="fast"),
        error_log=[],
        thought_log=[],
    )


class TestBudgetGuard:
    def test_budget_ok_delegates_to_critic_synthesize(self, base_state) -> None:
        from agent.budget_guard import check_budget

        # No critique and low iteration → critic says "synthesize"
        base_state["run_metadata"] = RunMetadata(
            run_id="test", profile="fast", iteration_count=1, estimated_cost_usd=0.01
        )
        result = check_budget(base_state)
        assert result == "synthesize"

    def test_budget_ok_critic_says_continue(self, base_state) -> None:
        from agent.budget_guard import check_budget

        base_state["run_metadata"] = RunMetadata(
            run_id="test", profile="fast", iteration_count=1, estimated_cost_usd=0.01
        )
        base_state["critique"] = CritiqueOutput(
            coverage_score=0.3,
            recency_score=0.3,
            depth_score=0.3,
            source_diversity_score=0.3,
            missing_areas=["more papers needed"],
            should_continue=True,
            reasoning="Insufficient coverage",
        )
        result = check_budget(base_state)
        assert result == "continue"

    def test_iteration_limit_forces_synthesis(self, base_state, monkeypatch) -> None:
        from agent.budget_guard import check_budget

        # Set iteration count to max
        monkeypatch.setattr("agent.budget_guard.settings.max_iterations", 10)
        base_state["run_metadata"] = RunMetadata(
            run_id="test", profile="fast", iteration_count=10, estimated_cost_usd=0.01
        )
        # Even if critic says continue, budget guard overrides
        base_state["critique"] = CritiqueOutput(
            coverage_score=0.3,
            recency_score=0.3,
            depth_score=0.3,
            source_diversity_score=0.3,
            should_continue=True,
            reasoning="Need more",
        )
        result = check_budget(base_state)
        assert result == "synthesize"

    def test_cost_limit_forces_synthesis(self, base_state, monkeypatch) -> None:
        from agent.budget_guard import check_budget

        monkeypatch.setattr("agent.budget_guard.settings.max_cost_per_run_usd", 1.0)
        base_state["run_metadata"] = RunMetadata(
            run_id="test", profile="fast", iteration_count=2, estimated_cost_usd=1.5
        )
        base_state["critique"] = CritiqueOutput(
            coverage_score=0.3,
            recency_score=0.3,
            depth_score=0.3,
            source_diversity_score=0.3,
            should_continue=True,
            reasoning="Need more",
        )
        result = check_budget(base_state)
        assert result == "synthesize"

    def test_no_metadata_falls_through_to_critic(self, base_state) -> None:
        from agent.budget_guard import check_budget

        # run_metadata is None
        base_state["run_metadata"] = None  # type: ignore[typeddict-item]
        result = check_budget(base_state)
        # With no meta and no critique, critic returns "synthesize"
        assert result == "synthesize"
