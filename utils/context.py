"""
utils/context.py

Async-safe context propagation for the DeepResearch Agent.

Uses Python's contextvars to propagate the current run_id through the
async call stack without threading it manually through every function
signature. All log records emitted within a run automatically include
the run_id for log aggregation and correlation.

Usage::

    from utils.context import bind_run_id, get_run_id

    # At the start of a run (e.g. in the API endpoint):
    bind_run_id("abc-123")

    # Anywhere downstream (nodes, middleware, tools):
    run_id = get_run_id()  # → "abc-123"
"""

from __future__ import annotations

from contextvars import ContextVar

# ContextVar is natively safe for asyncio — each task inherits the
# parent's value but can override it without affecting siblings.
_run_id_var: ContextVar[str] = ContextVar("run_id", default="")


def bind_run_id(run_id: str) -> None:
    """Set the current run_id for this async context."""
    _run_id_var.set(run_id)


def get_run_id() -> str:
    """Get the current run_id (empty string if not bound)."""
    return _run_id_var.get()
