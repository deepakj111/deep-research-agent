import asyncio

import pytest

from agent.state import ResearchFindings, ResearchState, RunMetadata, WebResult


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_web_result():
    return WebResult(
        url="https://example.com/article",
        title="Sample Article",
        snippet="This is a test snippet about the topic.",
        relevance_score=0.85,
    )


@pytest.fixture
def sample_findings(sample_web_result):
    return ResearchFindings(
        subquestion="What are the latest developments?",
        web_results=[sample_web_result],
    )


@pytest.fixture
def sample_state(sample_findings):
    return ResearchState(
        query="Test research query",
        profile="fast",
        run_id="test-run-001",
        query_difficulty="narrow",
        subquestions=["What are the latest developments?"],
        approved_plan=True,
        findings=[sample_findings],
        critique=None,
        iteration_count=0,
        final_report=None,
        run_metadata=RunMetadata(run_id="test-run-001", profile="fast"),
        error_log=[],
        thought_log=[],
    )
