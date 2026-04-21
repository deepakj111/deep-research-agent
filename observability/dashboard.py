"""
observability/dashboard.py

Observability dashboard for the DeepResearch Agent.

This module provides the data-access layer and Streamlit rendering helpers
that will be assembled into the full Phase 5 UI.  The functions here are
fully implemented and tested; only the top-level `render_dashboard()` page
entry point is stubbed until Phase 5 wires it into the multi-page Streamlit
app.
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
    Full Streamlit observability dashboard.

    Stubbed for Phase 5 — the data helpers above are already functional.
    Phase 5 will add:
      - Run history table with status badges
      - Cost/latency breakdown charts (Plotly)
      - Per-run tool call waterfall view
      - Benchmark score trend line
      - Live tail of the thought_log for active runs
    """
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError:
        print("Streamlit not available — run `uv add streamlit` to enable the dashboard.")
        return

    st.subheader("📊 Observability Dashboard")
    st.info(
        "Full dashboard UI is implemented in **Phase 5**. "
        "The data layer is ready — run `get_run_overview()` to query it directly.",
        icon="🔧",
    )

    runs = get_run_overview(limit=10)
    if runs:
        st.write("**10 Most Recent Runs**")
        st.dataframe(runs, use_container_width=True)
    else:
        st.write("No runs recorded yet. Start a research query to see data here.")
