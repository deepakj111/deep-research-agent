# agent/graph.py
from __future__ import annotations

import atexit
import sqlite3
import threading
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
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


class HybridSqliteSaver(BaseCheckpointSaver):
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self._asaver = None
        self._aconn = None
        self._sync_saver = None
        self._sync_conn = None

    async def _get_asaver(self):
        if self._asaver is None:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self._aconn = await aiosqlite.connect(self.db_path, check_same_thread=False)
            await self._aconn.execute("PRAGMA journal_mode=WAL")
            await self._aconn.execute("PRAGMA synchronous=NORMAL")
            self._asaver = AsyncSqliteSaver(self._aconn)
            await self._asaver.setup()
        return self._asaver

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        saver = await self._get_asaver()
        return await saver.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        saver = await self._get_asaver()
        return await saver.aput_writes(config, writes, task_id, task_path)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        saver = await self._get_asaver()
        return await saver.aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        saver = await self._get_asaver()
        async for ct in saver.alist(config, filter=filter, before=before, limit=limit):
            yield ct

    @property
    def sync_saver(self):
        if self._sync_saver is None:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            self._sync_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._sync_conn.execute("PRAGMA journal_mode=WAL")
            self._sync_conn.execute("PRAGMA synchronous=NORMAL")
            self._sync_saver = SqliteSaver(self._sync_conn)
            self._sync_saver.setup()
        return self._sync_saver

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        return self.sync_saver.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.sync_saver.put_writes(config, writes, task_id, task_path)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.sync_saver.get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        return self.sync_saver.list(config, filter=filter, before=before, limit=limit)

    def close(self):
        if self._sync_conn:
            self._sync_conn.close()
            self._sync_conn = None
        # Let async connection be garbage collected


_hybrid_saver = None


def _build_graph():
    """Construct and compile the LangGraph research agent graph.

    Called once by get_graph(); all subsequent calls return the cached instance.
    Keeps the SQLite checkpoint connection out of module-level scope so it is
    only opened on first use — not at import time.
    """
    global _hybrid_saver

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
    # HybridSqliteSaver supports both sync and async state methods.
    _hybrid_saver = HybridSqliteSaver(".checkpoints.db")

    return workflow.compile(
        checkpointer=_hybrid_saver,
        interrupt_before=["supervisor"],
    )


def _cleanup() -> None:
    """Close the checkpoint SQLite connection on process exit."""
    global _hybrid_saver
    if _hybrid_saver is not None:
        _hybrid_saver.close()
        _hybrid_saver = None


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
