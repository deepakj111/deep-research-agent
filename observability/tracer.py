"""
observability/tracer.py

Async-safe, SQLite-backed research run tracer.

Every MCP tool call, LLM node execution, and run lifecycle event is recorded
here. Designed around three hard constraints:

  1. Non-blocking  — all DB writes run in a thread pool via asyncio.to_thread
                     so they never stall the LangGraph event loop.
  2. Non-fatal     — every public method swallows exceptions; observability
                     must never propagate errors back into the agent.
  3. Singleton     — get_tracer() returns a process-level instance so all
                     nodes share one connection without re-opening the file.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import typing
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ─────────────────────────────── Data Classes ────────────────────────────────


@dataclass
class RunRecord:
    run_id: str
    query: str
    profile: str
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "running"


@dataclass
class ToolCallRecord:
    run_id: str
    node_name: str
    tool_name: str
    input_summary: str
    success: bool
    latency_ms: float
    estimated_cost_usd: float = 0.0
    error_message: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class NodeExecutionRecord:
    run_id: str
    node_name: str
    started_at: str
    completed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model_name: str = ""
    success: bool = True
    error_message: str | None = None


@dataclass
class EvalScoreRecord:
    run_id: str
    faithfulness: float
    answer_relevancy: float
    source_coverage: float
    citation_accuracy: float
    coherence: float
    normalized_average: float
    overall_notes: str
    evaluated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ──────────────────────────────── Tracer ─────────────────────────────────────


class ResearchTracer:
    """
    Thread-safe SQLite tracer for agent run observability.

    Instantiate once via get_tracer() and share across all nodes.
    Pass a custom db_path in tests (e.g. from tmp_path fixture) to
    avoid polluting the working directory.
    """

    def __init__(self, db_path: str | Path = ".research_runs.db") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            self._conn.executescript(schema_path.read_text())
        else:
            self._conn.executescript(_INLINE_SCHEMA)
        self._conn.commit()

    # ── Synchronous write helpers (dispatched via asyncio.to_thread) ──────────

    def _start_run_sync(self, record: RunRecord) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, query, profile, status, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.run_id,
                record.query[:2000],
                record.profile,
                record.status,
                record.started_at,
            ),
        )
        self._conn.commit()

    def _end_run_sync(
        self,
        run_id: str,
        status: str,
        total_cost_usd: float,
        total_latency_ms: float,
        iteration_count: int,
        findings_count: int,
        final_score: float | None,
    ) -> None:
        self._conn.execute(
            """UPDATE runs SET
                 status           = ?,
                 completed_at     = ?,
                 total_cost_usd   = ?,
                 total_latency_ms = ?,
                 iteration_count  = ?,
                 findings_count   = ?,
                 final_score      = ?
               WHERE run_id = ?""",
            (
                status,
                datetime.now(UTC).isoformat(),
                round(total_cost_usd, 6),
                round(total_latency_ms, 2),
                iteration_count,
                findings_count,
                final_score,
                run_id,
            ),
        )
        self._conn.commit()

    def _log_tool_call_sync(self, record: ToolCallRecord) -> None:
        self._conn.execute(
            """INSERT INTO tool_calls
               (run_id, node_name, tool_name, input_summary, success,
                latency_ms, estimated_cost_usd, error_message, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.run_id,
                record.node_name,
                record.tool_name,
                record.input_summary[:500],
                int(record.success),
                round(record.latency_ms, 2),
                round(record.estimated_cost_usd, 6),
                record.error_message,
                record.timestamp,
            ),
        )
        self._conn.commit()

    def _log_node_sync(self, record: NodeExecutionRecord) -> None:
        self._conn.execute(
            """INSERT INTO node_executions
               (run_id, node_name, started_at, completed_at, latency_ms,
                input_tokens, output_tokens, estimated_cost_usd,
                model_name, success, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.run_id,
                record.node_name,
                record.started_at,
                record.completed_at,
                round(record.latency_ms, 2),
                record.input_tokens,
                record.output_tokens,
                round(record.estimated_cost_usd, 6),
                record.model_name,
                int(record.success),
                record.error_message,
            ),
        )
        self._conn.commit()

    def _log_eval_sync(self, record: EvalScoreRecord) -> None:
        self._conn.execute(
            """INSERT INTO eval_scores
               (run_id, evaluated_at, faithfulness, answer_relevancy,
                source_coverage, citation_accuracy, coherence,
                normalized_average, overall_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.run_id,
                record.evaluated_at,
                record.faithfulness,
                record.answer_relevancy,
                record.source_coverage,
                record.citation_accuracy,
                record.coherence,
                record.normalized_average,
                record.overall_notes[:2000],
            ),
        )
        # Also update the run's final_score for quick dashboard queries
        self._conn.execute(
            "UPDATE runs SET final_score = ? WHERE run_id = ?",
            (record.normalized_average, record.run_id),
        )
        self._conn.commit()

    # ── Public async API ──────────────────────────────────────────────────────

    async def start_run(self, run_id: str, query: str, profile: str) -> None:
        """Record that a new research run has started."""
        record = RunRecord(run_id=run_id, query=query, profile=profile)
        await self._safe(self._start_run_sync, record)

    async def end_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        total_cost_usd: float = 0.0,
        total_latency_ms: float = 0.0,
        iteration_count: int = 0,
        findings_count: int = 0,
        final_score: float | None = None,
    ) -> None:
        """Finalise a run record with outcome metrics."""
        await self._safe(
            self._end_run_sync,
            run_id,
            status,
            total_cost_usd,
            total_latency_ms,
            iteration_count,
            findings_count,
            final_score,
        )

    async def log_tool_call(self, record: ToolCallRecord) -> None:
        """Record a single MCP tool call with its latency and outcome."""
        await self._safe(self._log_tool_call_sync, record)

    async def log_node_execution(self, record: NodeExecutionRecord) -> None:
        """Record an LLM node execution with token counts and cost."""
        await self._safe(self._log_node_sync, record)

    async def log_eval_scores(self, record: EvalScoreRecord) -> None:
        """Persist LLM-as-judge evaluation scores for a completed run."""
        await self._safe(self._log_eval_sync, record)

    # ── Query helpers (synchronous — safe to call from non-async context) ─────

    def get_run_summary(self, run_id: str) -> dict[str, typing.Any]:
        """Return aggregated metrics for a single run."""
        row = self._conn.execute(
            "SELECT run_id, query, profile, status, started_at, completed_at, "
            "total_cost_usd, total_latency_ms, iteration_count, findings_count, final_score "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return {}

        calls = self._conn.execute(
            "SELECT success, latency_ms FROM tool_calls WHERE run_id = ?",
            (run_id,),
        ).fetchall()

        total_calls = len(calls)
        success_rate = sum(c[0] for c in calls) / total_calls if total_calls else 0.0

        return {
            "run_id": row[0],
            "query": row[1],
            "profile": row[2],
            "status": row[3],
            "started_at": row[4],
            "completed_at": row[5],
            "total_cost_usd": row[6],
            "total_latency_ms": row[7],
            "iteration_count": row[8],
            "findings_count": row[9],
            "final_score": row[10],
            "tool_call_count": total_calls,
            "tool_success_rate": round(success_rate, 3),
        }

    def get_recent_runs(self, limit: int = 20) -> list[dict[str, typing.Any]]:
        """Return the most recent runs ordered by start time."""
        rows = self._conn.execute(
            "SELECT run_id, query, profile, status, started_at, "
            "total_cost_usd, final_score "
            "FROM runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "run_id": r[0],
                "query": r[1],
                "profile": r[2],
                "status": r[3],
                "started_at": r[4],
                "total_cost_usd": r[5],
                "final_score": r[6],
            }
            for r in rows
        ]

    def get_tool_call_stats(self, run_id: str) -> list[dict[str, typing.Any]]:
        """Return per-tool aggregated stats for a run (useful for dashboards)."""
        rows = self._conn.execute(
            """SELECT
                 tool_name,
                 COUNT(*)                  AS total_calls,
                 SUM(success)              AS success_count,
                 AVG(latency_ms)           AS avg_latency_ms,
                 SUM(estimated_cost_usd)   AS total_cost_usd
               FROM tool_calls
               WHERE run_id = ?
               GROUP BY tool_name""",
            (run_id,),
        ).fetchall()
        return [
            {
                "tool_name": r[0],
                "total_calls": r[1],
                "success_count": r[2],
                "avg_latency_ms": round(r[3] or 0.0, 1),
                "total_cost_usd": round(r[4] or 0.0, 6),
            }
            for r in rows
        ]

    def get_node_timings(self, run_id: str) -> list[dict[str, typing.Any]]:
        """Return per-node latency breakdown for a run."""
        rows = self._conn.execute(
            """SELECT node_name, latency_ms, input_tokens, output_tokens,
                      estimated_cost_usd, model_name, success
               FROM node_executions
               WHERE run_id = ?
               ORDER BY started_at""",
            (run_id,),
        ).fetchall()
        return [
            {
                "node_name": r[0],
                "latency_ms": r[1],
                "input_tokens": r[2],
                "output_tokens": r[3],
                "estimated_cost_usd": r[4],
                "model_name": r[5],
                "success": bool(r[6]),
            }
            for r in rows
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _safe(self, fn: typing.Callable[..., None], *args: typing.Any) -> None:
        """
        Dispatch a blocking SQLite write to the thread pool.
        Swallows all exceptions — observability must never crash the agent.
        """
        with suppress(Exception):  # intentional: tracing is best-effort
            await asyncio.to_thread(fn, *args)


# ──────────────────────────── Context Manager ────────────────────────────────


@asynccontextmanager
async def trace_tool_call(
    tracer: ResearchTracer,
    run_id: str,
    node_name: str,
    tool_name: str,
    input_summary: str,
) -> AsyncIterator[None]:
    """
    Async context manager that records a tool call's wall-clock latency
    and whether it succeeded — even when an exception is raised.

    Example::

        async with trace_tool_call(tracer, run_id, "web_agent", "search_web", query):
            raw = await circuit_breakers["search_web"].call(_call())
    """
    start = time.perf_counter()
    error: str | None = None
    success = True
    try:
        yield
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        await tracer.log_tool_call(
            ToolCallRecord(
                run_id=run_id,
                node_name=node_name,
                tool_name=tool_name,
                input_summary=input_summary[:300],
                success=success,
                latency_ms=round(latency_ms, 2),
                error_message=error,
            )
        )


# ──────────────────────────────── Singleton ──────────────────────────────────

_tracer_instance: ResearchTracer | None = None


def get_tracer(db_path: str | Path | None = None) -> ResearchTracer:
    """
    Return the process-level tracer singleton.

    On first call, creates the SQLite file at db_path (default:
    .research_runs.db in the working directory). Subsequent calls
    ignore db_path and return the existing instance.

    In tests, create a dedicated ResearchTracer(db_path=tmp_path/...) instead
    of using this singleton to avoid cross-test contamination.
    """
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = ResearchTracer(db_path=db_path or ".research_runs.db")
    return _tracer_instance


# ─────────────────────────── Inline Schema Fallback ──────────────────────────
# Used when schema.sql is not present (e.g. running tests outside the repo root).

_INLINE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, query TEXT NOT NULL, profile TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running', started_at TEXT NOT NULL,
    completed_at TEXT, total_cost_usd REAL DEFAULT 0.0,
    total_latency_ms REAL DEFAULT 0.0, iteration_count INTEGER DEFAULT 0,
    findings_count INTEGER DEFAULT 0, final_score REAL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    node_name TEXT NOT NULL, tool_name TEXT NOT NULL, input_summary TEXT,
    success INTEGER NOT NULL DEFAULT 1, latency_ms REAL NOT NULL DEFAULT 0.0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0, error_message TEXT,
    timestamp TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS node_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    node_name TEXT NOT NULL, started_at TEXT NOT NULL, completed_at TEXT,
    latency_ms REAL, input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0, estimated_cost_usd REAL DEFAULT 0.0,
    model_name TEXT DEFAULT '', success INTEGER DEFAULT 1, error_message TEXT
);
CREATE TABLE IF NOT EXISTS eval_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
    evaluated_at TEXT NOT NULL, faithfulness REAL, answer_relevancy REAL,
    source_coverage REAL, citation_accuracy REAL, coherence REAL,
    normalized_average REAL, overall_notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_tc_run  ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_ne_run  ON node_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_es_run  ON eval_scores(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(started_at DESC);
"""
