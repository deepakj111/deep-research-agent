-- observability/schema.sql
-- Research run observability schema.
-- Applied once on tracer initialization; all CREATE statements are idempotent.

PRAGMA journal_mode = WAL;          -- better concurrent read/write throughput
PRAGMA foreign_keys = ON;

-- ─────────────────────────────────────────────────────────────────────────────
-- Top-level run record — one row per agent invocation
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT    PRIMARY KEY,
    query             TEXT    NOT NULL,
    profile           TEXT    NOT NULL,
    status            TEXT    NOT NULL DEFAULT 'running',  -- running|completed|failed|budget_exceeded
    started_at        TEXT    NOT NULL,
    completed_at      TEXT,
    total_cost_usd    REAL    DEFAULT 0.0,
    total_latency_ms  REAL    DEFAULT 0.0,
    iteration_count   INTEGER DEFAULT 0,
    findings_count    INTEGER DEFAULT 0,
    final_score       REAL                                 -- normalized 0-1 from eval pipeline
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Individual MCP tool calls (web_search, fetch_papers, search_repos)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tool_calls (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT    NOT NULL REFERENCES runs(run_id),
    node_name            TEXT    NOT NULL,
    tool_name            TEXT    NOT NULL,
    input_summary        TEXT,
    success              INTEGER NOT NULL DEFAULT 1,
    latency_ms           REAL    NOT NULL DEFAULT 0.0,
    estimated_cost_usd   REAL    NOT NULL DEFAULT 0.0,
    error_message        TEXT,
    timestamp            TEXT    NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- LLM node executions (classifier, planner, critic, synthesizer, writer)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS node_executions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT    NOT NULL REFERENCES runs(run_id),
    node_name            TEXT    NOT NULL,
    started_at           TEXT    NOT NULL,
    completed_at         TEXT,
    latency_ms           REAL,
    input_tokens         INTEGER DEFAULT 0,
    output_tokens        INTEGER DEFAULT 0,
    estimated_cost_usd   REAL    DEFAULT 0.0,
    model_name           TEXT    DEFAULT '',
    success              INTEGER DEFAULT 1,
    error_message        TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Evaluation scores — populated by the LLM-as-judge pipeline
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_scores (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT    NOT NULL REFERENCES runs(run_id),
    evaluated_at       TEXT    NOT NULL,
    faithfulness       REAL,
    answer_relevancy   REAL,
    source_coverage    REAL,
    citation_accuracy  REAL,
    coherence          REAL,
    normalized_average REAL,
    overall_notes      TEXT
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Indexes for common query patterns
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id        ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_node_executions_run_id   ON node_executions(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_scores_run_id       ON eval_scores(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at          ON runs(started_at DESC);
