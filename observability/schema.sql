-- Research runs — one row per agent run
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    profile TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    completed_at TEXT,
    total_cost_usd REAL DEFAULT 0.0,
    total_latency_ms REAL DEFAULT 0.0,
    final_score REAL
);

-- Tool calls — one row per tool invocation
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    node_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    input_summary TEXT,
    success INTEGER NOT NULL DEFAULT 1,
    latency_ms REAL DEFAULT 0.0,
    estimated_cost_usd REAL DEFAULT 0.0,
    error_message TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
