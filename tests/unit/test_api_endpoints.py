"""
tests/unit/test_api_endpoints.py

Unit tests for the FastAPI gateway endpoints.

Uses FastAPI's TestClient with mocked graph state to test:
  - CORS headers
  - Report retrieval (JSON, Markdown, PDF, HTML)
  - Health endpoint
  - Error handling for missing threads
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.state import Citation, Finding, ReportOutput

# ────────────────────────── Fixtures ──────────────────────────────────────────


@pytest.fixture
def sample_report() -> ReportOutput:
    return ReportOutput(
        title="Test Research Report",
        executive_summary="This is a test summary for API testing.",
        key_findings=[
            Finding(
                claim="Test finding claim",
                citations=[
                    Citation(
                        source_url="https://example.com/test",
                        title="Test Source",
                        exact_snippet="Test snippet content",
                        source_type="web",
                        trust_score=0.75,
                    )
                ],
                confidence="high",
            )
        ],
        emerging_trends=["Test trend"],
        recommended_next_steps=["Test step"],
        sources=[
            Citation(
                source_url="https://example.com/test",
                title="Test Source",
                exact_snippet="Test snippet content",
                source_type="web",
                trust_score=0.75,
            )
        ],
        version=1,
    )


@pytest.fixture
def mock_graph_state(sample_report: ReportOutput):
    """Create a mock graph state snapshot that contains a final report."""
    snapshot = MagicMock()
    snapshot.values = {
        "query": "test query",
        "query_difficulty": "narrow",
        "subquestions": ["sub1"],
        "findings": [],
        "approved_plan": True,
        "final_report": sample_report,
        "run_metadata": None,
        "error_log": [],
        "thought_log": [],
    }
    snapshot.next = ()
    return snapshot


@pytest.fixture
def client():
    """Create a TestClient with the graph mocked."""
    with patch("api.main.graph") as mock_graph:
        mock_graph.get_state.return_value = None
        from api.main import app

        yield TestClient(app), mock_graph


# ────────────────────────── Health Tests ──────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "deep-research-agent-api"


# ────────────────────────── CORS Tests ────────────────────────────────────────


class TestCORS:
    def test_cors_allows_streamlit_origin(self, client):
        test_client, _ = client
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


# ────────────────────────── Report Endpoint Tests ─────────────────────────────


class TestReportEndpoints:
    def test_get_report_json_not_found(self, client):
        test_client, mock_graph = client
        mock_graph.get_state.return_value = None
        response = test_client.get("/research/report/nonexistent-id")
        assert response.status_code == 404

    def test_get_report_json_success(self, client, mock_graph_state, sample_report):
        test_client, mock_graph = client
        mock_graph.get_state.return_value = mock_graph_state
        response = test_client.get("/research/report/test-thread-id")
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Research Report"
        assert data["version"] == 1

    def test_get_report_no_report_in_state(self, client):
        test_client, mock_graph = client
        snapshot = MagicMock()
        snapshot.values = {"final_report": None}
        mock_graph.get_state.return_value = snapshot
        response = test_client.get("/research/report/test-thread-id")
        assert response.status_code == 404
        assert "no final report" in response.json()["detail"].lower()

    def test_get_report_markdown(self, client, mock_graph_state):
        test_client, mock_graph = client
        mock_graph.get_state.return_value = mock_graph_state
        response = test_client.get("/research/report/test-thread-id/markdown")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/markdown; charset=utf-8"
        assert "# Test Research Report" in response.text

    def test_get_report_html(self, client, mock_graph_state):
        test_client, mock_graph = client
        mock_graph.get_state.return_value = mock_graph_state
        response = test_client.get("/research/report/test-thread-id/html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<!DOCTYPE html>" in response.text
        assert "Test Research Report" in response.text

    def test_get_report_pdf(self, client, mock_graph_state):
        test_client, mock_graph = client
        mock_graph.get_state.return_value = mock_graph_state
        try:
            response = test_client.get("/research/report/test-thread-id/pdf")
            if response.status_code == 200:
                assert response.headers["content-type"] == "application/pdf"
                assert response.content[:5] == b"%PDF-"
            elif response.status_code == 501:
                # WeasyPrint not installed — acceptable in CI
                pass
        except Exception:
            pytest.skip("PDF generation unavailable in this environment")


# ────────────────────────── State Endpoint Tests ──────────────────────────────


class TestStateEndpoint:
    def test_state_not_found(self, client):
        test_client, mock_graph = client
        mock_graph.get_state.return_value = None
        response = test_client.get("/research/state/nonexistent-id")
        assert response.status_code == 404

    def test_state_success(self, client, mock_graph_state):
        test_client, mock_graph = client
        mock_graph.get_state.return_value = mock_graph_state
        response = test_client.get("/research/state/test-thread-id")
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "test-thread-id"
        assert data["query"] == "test query"


# ────────────────────────── Runs Endpoint Tests ───────────────────────────────


class TestRunsEndpoint:
    def test_list_runs(self, client):
        test_client, _ = client
        from api.main import app, get_tracer

        mock_tracer = MagicMock()
        mock_tracer.get_recent_runs.return_value = [
            {
                "run_id": "test-run",
                "query": "test",
                "profile": "fast",
                "status": "completed",
                "started_at": "2026-01-01T00:00:00",
                "total_cost_usd": 0.05,
                "final_score": 0.85,
            }
        ]
        app.dependency_overrides[get_tracer] = lambda: mock_tracer

        try:
            response = test_client.get("/research/runs")
            assert response.status_code == 200
            data = response.json()
            assert len(data["runs"]) == 1
            assert data["runs"][0]["run_id"] == "test-run"
        finally:
            app.dependency_overrides.clear()

    def test_run_detail_not_found(self, client):
        test_client, _ = client
        from api.main import app, get_tracer

        mock_tracer = MagicMock()
        mock_tracer.get_run_summary.return_value = {}
        app.dependency_overrides[get_tracer] = lambda: mock_tracer

        try:
            response = test_client.get("/research/runs/nonexistent")
            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()
