"""
observability/dashboard.py

Observability dashboard API helpers for the DeepResearch Agent.

This module provides the data-access layer used by the Streamlit rendering helpers
in `app/pages/traces.py`.
"""

from __future__ import annotations

from typing import Any

from observability.tracer import get_tracer

# ─────────────────────────── Data Access Layer ───────────────────────────────


def get_run_overview(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent runs with summary statistics for the dashboard table."""
    tracer = get_tracer()
    return tracer.get_recent_runs(limit=limit)


def get_run_detail(run_id: str) -> dict[str, Any]:
    """Return full detail for a single run: summary + tool stats + node timings."""
    tracer = get_tracer()
    return {
        "summary": tracer.get_run_summary(run_id),
        "tool_stats": tracer.get_tool_call_stats(run_id),
        "node_timings": tracer.get_node_timings(run_id),
    }


def get_cost_by_profile() -> dict[str, float]:
    """
    Return average cost per run grouped by profile.
    Used for the cost comparison chart in the README cost table.
    """
    tracer = get_tracer()
    conn = tracer._conn  # noqa: SLF001  — internal access for dashboard only
    rows = conn.execute(
        "SELECT profile, AVG(total_cost_usd) FROM runs WHERE status = 'completed' GROUP BY profile"
    ).fetchall()
    return {r[0]: round(r[1] or 0.0, 4) for r in rows}


def get_tool_success_rates() -> dict[str, float]:
    """Return overall success rate per tool across all runs."""
    tracer = get_tracer()
    conn = tracer._conn  # noqa: SLF001
    rows = conn.execute(
        "SELECT tool_name, AVG(CAST(success AS REAL)) FROM tool_calls GROUP BY tool_name"
    ).fetchall()
    return {r[0]: round(r[1] or 0.0, 3) for r in rows}


# ──────────────────────────── Streamlit Page ─────────────────────────────────


def render_dashboard() -> None:
    """
    Streamlit observability dashboard entrypoint (used for standalone mode).
    For the full multi-page app, see `app/streamlit_app.py`.
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError:
        print("Streamlit not available — run `uv add streamlit` to enable the dashboard.")
        return

    st.subheader("📊 Observability Dashboard")
    st.info(
        "For the full interactive UI, run `uv run streamlit run app/streamlit_app.py`.",
        icon="🔧",
    )

    runs = get_run_overview(limit=10)
    if runs:
        st.write("**10 Most Recent Runs**")
        st.dataframe(runs, use_container_width=True)
    else:
        st.write("No runs recorded yet. Start a research query to see data here.")
