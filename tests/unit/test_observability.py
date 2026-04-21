# tests/unit/test_observability.py
"""
Unit tests for the observability tracer.

All tests use tmp_path-scoped SQLite databases to avoid cross-test
contamination and to keep the working directory clean.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from observability.tracer import (
    EvalScoreRecord,
    NodeExecutionRecord,
    ResearchTracer,
    ToolCallRecord,
    trace_tool_call,
)

# ─────────────────────────── Fixtures ────────────────────────────────────────


@pytest.fixture
def tracer(tmp_path: Path) -> ResearchTracer:
    """Return a fresh tracer backed by a temporary SQLite file."""
    return ResearchTracer(db_path=tmp_path / "test_runs.db")


# ─────────────────────────── Schema / Initialization ─────────────────────────


class TestTracerInit:
    def test_db_file_is_created(self, tmp_path: Path) -> None:
        db = tmp_path / "init_test.db"
        ResearchTracer(db_path=db)
        assert db.exists()

    def test_tables_exist_after_init(self, tracer: ResearchTracer) -> None:
        tables = {
            r[0]
            for r in tracer._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"runs", "tool_calls", "node_executions", "eval_scores"} <= tables


# ──────────────────────── Run Lifecycle ──────────────────────────────────────


class TestRunLifecycle:
    @pytest.mark.asyncio
    async def test_start_run_creates_record(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("run-001", "test query", "fast")
        row = tracer._conn.execute(
            "SELECT run_id, query, profile, status FROM runs WHERE run_id = 'run-001'"
        ).fetchone()
        assert row is not None
        assert row[0] == "run-001"
        assert row[1] == "test query"
        assert row[2] == "fast"
        assert row[3] == "running"

    @pytest.mark.asyncio
    async def test_end_run_updates_record(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("run-002", "another query", "deep")
        await tracer.end_run(
            "run-002",
            status="completed",
            total_cost_usd=0.05,
            total_latency_ms=12345.0,
            iteration_count=3,
            findings_count=9,
            final_score=0.82,
        )
        row = tracer._conn.execute(
            "SELECT status, total_cost_usd, total_latency_ms, "
            "iteration_count, findings_count, final_score "
            "FROM runs WHERE run_id = 'run-002'"
        ).fetchone()
        assert row[0] == "completed"
        assert row[1] == pytest.approx(0.05)
        assert row[2] == pytest.approx(12345.0)
        assert row[3] == 3
        assert row[4] == 9
        assert row[5] == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_start_run_idempotent(self, tracer: ResearchTracer) -> None:
        """Calling start_run twice for the same run_id must not raise or duplicate."""
        await tracer.start_run("run-003", "q", "fast")
        await tracer.start_run("run-003", "q", "fast")  # second call — INSERT OR IGNORE
        count = tracer._conn.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = 'run-003'"
        ).fetchone()[0]
        assert count == 1

    def test_get_run_summary_returns_empty_for_unknown(self, tracer: ResearchTracer) -> None:
        assert tracer.get_run_summary("nonexistent") == {}

    @pytest.mark.asyncio
    async def test_get_run_summary_contains_expected_keys(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("run-004", "query text", "fast")
        summary = tracer.get_run_summary("run-004")
        expected_keys = {
            "run_id",
            "query",
            "profile",
            "status",
            "started_at",
            "completed_at",
            "total_cost_usd",
            "total_latency_ms",
            "iteration_count",
            "findings_count",
            "final_score",
            "tool_call_count",
            "tool_success_rate",
        }
        assert expected_keys == set(summary.keys())


# ─────────────────────────── Tool Calls ──────────────────────────────────────


class TestToolCallLogging:
    @pytest.mark.asyncio
    async def test_successful_tool_call_is_recorded(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("r1", "q", "fast")
        await tracer.log_tool_call(
            ToolCallRecord(
                run_id="r1",
                node_name="web_agent",
                tool_name="search_web",
                input_summary="quantum computing 2025",
                success=True,
                latency_ms=423.5,
            )
        )
        row = tracer._conn.execute(
            "SELECT tool_name, success, latency_ms FROM tool_calls WHERE run_id = 'r1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "search_web"
        assert row[1] == 1
        assert row[2] == pytest.approx(423.5)

    @pytest.mark.asyncio
    async def test_failed_tool_call_records_error(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("r2", "q", "fast")
        await tracer.log_tool_call(
            ToolCallRecord(
                run_id="r2",
                node_name="arxiv_agent",
                tool_name="fetch_papers",
                input_summary="test",
                success=False,
                latency_ms=50.0,
                error_message="ConnectionError: timeout",
            )
        )
        row = tracer._conn.execute(
            "SELECT success, error_message FROM tool_calls WHERE run_id = 'r2'"
        ).fetchone()
        assert row[0] == 0
        assert "ConnectionError" in row[1]

    @pytest.mark.asyncio
    async def test_get_tool_call_stats_aggregates_correctly(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("r3", "q", "fast")
        for i in range(3):
            await tracer.log_tool_call(
                ToolCallRecord(
                    run_id="r3",
                    node_name="web_agent",
                    tool_name="search_web",
                    input_summary=f"q{i}",
                    success=(i < 2),  # 2 success, 1 failure
                    latency_ms=100.0 * (i + 1),
                )
            )
        stats = tracer.get_tool_call_stats("r3")
        assert len(stats) == 1
        row = stats[0]
        assert row["tool_name"] == "search_web"
        assert row["total_calls"] == 3
        assert row["success_count"] == 2
        assert row["avg_latency_ms"] == pytest.approx(200.0)  # (100+200+300)/3


# ─────────────────────────── Node Executions ─────────────────────────────────


class TestNodeExecutionLogging:
    @pytest.mark.asyncio
    async def test_node_execution_is_recorded(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("r4", "q", "fast")
        await tracer.log_node_execution(
            NodeExecutionRecord(
                run_id="r4",
                node_name="critic",
                started_at=datetime.now(UTC).isoformat(),
                latency_ms=800.0,
                input_tokens=500,
                output_tokens=200,
                estimated_cost_usd=0.0035,
                model_name="gpt-4o",
            )
        )
        timings = tracer.get_node_timings("r4")
        assert len(timings) == 1
        assert timings[0]["node_name"] == "critic"
        assert timings[0]["input_tokens"] == 500
        assert timings[0]["model_name"] == "gpt-4o"


# ─────────────────────────── Eval Scores ────────────────────────────────────


class TestEvalScoreLogging:
    @pytest.mark.asyncio
    async def test_eval_scores_persisted_and_final_score_updated(
        self, tracer: ResearchTracer
    ) -> None:
        await tracer.start_run("r5", "q", "fast")
        await tracer.log_eval_scores(
            EvalScoreRecord(
                run_id="r5",
                faithfulness=4.5,
                answer_relevancy=4.0,
                source_coverage=3.5,
                citation_accuracy=4.2,
                coherence=4.8,
                normalized_average=0.84,
                overall_notes="Good report with minor citation gaps.",
            )
        )
        row = tracer._conn.execute(
            "SELECT faithfulness, normalized_average FROM eval_scores WHERE run_id = 'r5'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(4.5)
        assert row[1] == pytest.approx(0.84)

        # Ensure final_score was propagated to the runs table
        run_row = tracer._conn.execute(
            "SELECT final_score FROM runs WHERE run_id = 'r5'"
        ).fetchone()
        assert run_row[0] == pytest.approx(0.84)


# ─────────────────────── trace_tool_call Context Manager ─────────────────────


class TestTraceToolCallContextManager:
    @pytest.mark.asyncio
    async def test_records_successful_call(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("r6", "q", "fast")

        async with trace_tool_call(tracer, "r6", "web_agent", "search_web", "test"):
            await asyncio.sleep(0)  # simulate async work

        row = tracer._conn.execute(
            "SELECT success, tool_name FROM tool_calls WHERE run_id = 'r6'"
        ).fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] == "search_web"

    @pytest.mark.asyncio
    async def test_records_failed_call_and_reraises(self, tracer: ResearchTracer) -> None:
        await tracer.start_run("r7", "q", "fast")

        with pytest.raises(RuntimeError, match="tool broke"):
            async with trace_tool_call(tracer, "r7", "arxiv_agent", "fetch_papers", "test"):
                raise RuntimeError("tool broke")

        row = tracer._conn.execute(
            "SELECT success, error_message FROM tool_calls WHERE run_id = 'r7'"
        ).fetchone()
        assert row is not None
        assert row[0] == 0
        assert "RuntimeError" in row[1]

    @pytest.mark.asyncio
    async def test_tracer_failure_does_not_propagate(self, tmp_path: Path) -> None:
        """Closing the DB connection before use simulates a broken tracer; the
        context manager must still re-raise the original error, not the SQLite one."""
        broken_tracer = ResearchTracer(db_path=tmp_path / "broken.db")
        broken_tracer._conn.close()  # deliberately break the connection

        # Should not raise SQLite errors — observability is best-effort
        with pytest.raises(RuntimeError, match="original"):
            async with trace_tool_call(broken_tracer, "r8", "web_agent", "search_web", "test"):
                raise RuntimeError("original")


# ───────────────────────── Recent Runs Query ─────────────────────────────────


class TestGetRecentRuns:
    @pytest.mark.asyncio
    async def test_recent_runs_returns_correct_count(self, tracer: ResearchTracer) -> None:
        for i in range(5):
            await tracer.start_run(f"run-{i}", f"query {i}", "fast")

        runs = tracer.get_recent_runs(limit=3)
        assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_recent_runs_empty_when_no_runs(self, tracer: ResearchTracer) -> None:
        assert tracer.get_recent_runs() == []
