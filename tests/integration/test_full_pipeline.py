# tests/integration/test_full_pipeline.py
from unittest.mock import AsyncMock, patch

import pytest

from agent.state import ResearchState, RunMetadata


@pytest.mark.asyncio
@pytest.mark.integration
async def test_graph_compiles_without_error():
    from agent.graph import graph

    assert graph is not None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_classifier_node_runs():
    from agent.nodes.classifier import run

    with patch("agent.nodes.classifier._llm") as mock_llm:
        from pydantic import BaseModel

        class FakeOutput(BaseModel):
            difficulty: str = "narrow"
            reasoning: str = "Specific query"
            suggested_num_questions: int = 3

        mock_llm.ainvoke = AsyncMock(return_value=FakeOutput())

        state = ResearchState(
            query="quantum computing",
            profile="fast",
            run_id="int-test-001",
            query_difficulty="",
            subquestions=[],
            approved_plan=False,
            findings=[],
            critique=None,
            iteration_count=0,
            final_report=None,
            run_metadata=RunMetadata(run_id="int-test-001", profile="fast"),
            error_log=[],
            thought_log=[],
        )

        result = await run(state)
        assert result["query_difficulty"] == "narrow"
        assert len(result["thought_log"]) > 0
