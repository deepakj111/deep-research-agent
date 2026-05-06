# agent/graph.py
from __future__ import annotations

import atexit
import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent.budget_guard import check_budget
from agent.nodes import (
    arxiv_agent,
    classifier,
    critic,
    github_agent,
    planner,
    supervisor,
    synthesizer,
    web_agent,
    writer,
)
from agent.state import ResearchState

# Threading lock protects the lazy singleton from double-initialization
# when multiple threads call get_graph() concurrently (e.g. FastAPI with
# threaded middleware or background tasks within a single worker process).
_graph_lock = threading.Lock()
_graph = None
_checkpoint_conn: sqlite3.Connection | None = None


def _build_graph():
    """Construct and compile the LangGraph research agent graph.

    Called once by get_graph(); all subsequent calls return the cached instance.
    Keeps the SQLite checkpoint connection out of module-level scope so it is
    only opened on first use — not at import time.
    """
    global _checkpoint_conn

    workflow = StateGraph(ResearchState)

    workflow.add_node("classifier", classifier.run)
    workflow.add_node("planner", planner.run)
    workflow.add_node("supervisor", supervisor.run)
    workflow.add_node("web_agent", web_agent.run)
    workflow.add_node("arxiv_agent", arxiv_agent.run)
    workflow.add_node("github_agent", github_agent.run)
    workflow.add_node("critic", critic.run)
    workflow.add_node("synthesizer", synthesizer.run)
    workflow.add_node("writer", writer.run)

    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier", "planner")
    workflow.add_edge("planner", "supervisor")

    # Agent nodes feed back to critic after each individual call
    # (Send-based fan-out means all agents run in parallel, then reconverge at critic)
    workflow.add_edge("web_agent", "critic")
    workflow.add_edge("arxiv_agent", "critic")
    workflow.add_edge("github_agent", "critic")

    # Budget guard wraps the critic's should_continue decision:
    # - checks iteration count against settings.max_iterations
    # - checks estimated_cost_usd against settings.max_cost_per_run_usd
    # - if budget OK, delegates to critic.should_continue
    workflow.add_conditional_edges(
        "critic",
        check_budget,
        {
            "continue": "planner",
            "synthesize": "synthesizer",
        },
    )

    workflow.add_edge("synthesizer", "writer")
    workflow.add_edge("writer", END)

    # SqliteSaver persists state across process restarts — required for HITL resume.
    # WAL mode enables concurrent readers without blocking writers, which is
    # essential when SSE streams read state while the graph is still writing.
    _checkpoint_conn = sqlite3.connect(".checkpoints.db", check_same_thread=False)
    _checkpoint_conn.execute("PRAGMA journal_mode=WAL")
    _checkpoint_conn.execute("PRAGMA synchronous=NORMAL")
    memory = SqliteSaver(_checkpoint_conn)

    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["planner"],
    )


def _cleanup() -> None:
    """Close the checkpoint SQLite connection on process exit."""
    global _checkpoint_conn
    if _checkpoint_conn is not None:
        _checkpoint_conn.close()
        _checkpoint_conn = None


atexit.register(_cleanup)


def get_graph():
    """Return the compiled research agent graph (thread-safe lazy singleton)."""
    global _graph
    if _graph is not None:
        return _graph
    with _graph_lock:
        # Double-check after acquiring the lock — another thread may have
        # initialized the graph while we were waiting.
        if _graph is None:
            _graph = _build_graph()
        return _graph


# Backward-compatible module-level alias so existing `from agent.graph import graph`
# continues to work. The property-like access is achieved via a module __getattr__.
def __getattr__(name: str):
    if name == "graph":
        return get_graph()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
