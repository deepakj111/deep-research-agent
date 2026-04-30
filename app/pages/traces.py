"""
app/pages/traces.py

Observability dashboard for the DeepResearch Agent.

Displays:
- Run history table (recent runs from the observability DB)
- Selected run deep-dive: node execution timeline, cost breakdown,
  tool call success rate, and source inventory.

All data is fetched from the FastAPI gateway's observability endpoints.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.auth import require_auth
from app.components.theme import (
    COLORS,
    hero_header,
    inject_theme,
    metric_card,
)

# ────────────────────────── Config ────────────────────────────────────────────

AGENT_API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8080")

st.set_page_config(page_title="Traces | DeepResearch", page_icon="📊", layout="wide")
inject_theme()
require_auth()

hero_header("📊 Observability Dashboard", "Inspect past runs, node timings, and cost breakdowns")


# ────────────────────────── API Helpers ────────────────────────────────────────


def _fetch_runs(limit: int = 20) -> list[dict]:
    try:
        resp = httpx.get(f"{AGENT_API_URL}/research/runs", params={"limit": limit}, timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("runs", [])
    except Exception:
        pass
    return []


def _fetch_run_detail(run_id: str) -> dict | None:
    try:
        resp = httpx.get(f"{AGENT_API_URL}/research/runs/{run_id}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ────────────────────────── Run History ───────────────────────────────────────

st.markdown("### 📋 Recent Runs")

runs = _fetch_runs()

if not runs:
    st.info(
        "No runs found. Start a research query on the **Research** page, "
        "or ensure the Agent API is running."
    )
    st.stop()

# Build the runs table
runs_df = pd.DataFrame(runs)
runs_df = runs_df.rename(
    columns={
        "run_id": "Run ID",
        "query": "Query",
        "profile": "Profile",
        "status": "Status",
        "started_at": "Started",
        "total_cost_usd": "Cost (USD)",
        "final_score": "Quality Score",
    }
)

# Truncate query for display
if "Query" in runs_df.columns:
    runs_df["Query"] = runs_df["Query"].apply(
        lambda x: (x[:60] + "...") if isinstance(x, str) and len(x) > 60 else x
    )

# Format cost
if "Cost (USD)" in runs_df.columns:
    runs_df["Cost (USD)"] = runs_df["Cost (USD)"].apply(
        lambda x: f"${x:.4f}" if x is not None else "—"
    )

# Format score
if "Quality Score" in runs_df.columns:
    runs_df["Quality Score"] = runs_df["Quality Score"].apply(
        lambda x: f"{x:.2f}" if x is not None else "—"
    )

st.dataframe(runs_df, use_container_width=True, hide_index=True)

# ────────────────────────── Run Selector ──────────────────────────────────────

st.markdown("---")
st.markdown("### 🔍 Run Deep-Dive")

run_ids = [r.get("run_id", "") for r in runs]
selected_run_id = st.selectbox(
    "Select a run to inspect",
    options=run_ids,
    format_func=lambda x: (
        f"{x[:8]}... — {next((r.get('query', '?')[:50] for r in runs if r.get('run_id') == x), '?')}"
    ),
)

if not selected_run_id:
    st.stop()

detail = _fetch_run_detail(selected_run_id)

if not detail:
    st.warning(f"Could not load details for run `{selected_run_id[:8]}...`")
    st.stop()

summary = detail.get("summary", {})
tool_stats = detail.get("tool_stats", [])
node_timings = detail.get("node_timings", [])

# ────────────────────────── Summary Metrics ───────────────────────────────────

st.markdown("#### 📈 Run Summary")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    metric_card("Status", summary.get("status", "unknown").upper())
with m2:
    cost = summary.get("total_cost_usd")
    metric_card("Total Cost", f"${cost:.4f}" if cost else "—")
with m3:
    latency = summary.get("total_latency_ms")
    if latency:
        metric_card("Latency", f"{latency / 1000:.1f}s")
    else:
        metric_card("Latency", "—")
with m4:
    metric_card("Iterations", str(summary.get("iteration_count", "—")))
with m5:
    score = summary.get("final_score")
    metric_card("Quality", f"{score:.2f}" if score else "—")

# ────────────────────────── Node Execution Timeline ───────────────────────────

if node_timings:
    st.markdown("#### ⏱️ Node Execution Timeline")

    nt_df = pd.DataFrame(node_timings)

    if "latency_ms" in nt_df.columns and "node_name" in nt_df.columns:
        fig_timeline = px.bar(
            nt_df,
            x="latency_ms",
            y="node_name",
            orientation="h",
            color="node_name",
            color_discrete_sequence=[
                COLORS["accent_teal"],
                COLORS["accent_blue"],
                COLORS["accent_orange"],
                COLORS["accent_purple"],
                COLORS["accent_pink"],
                COLORS["success"],
                COLORS["warning"],
            ],
            labels={"latency_ms": "Latency (ms)", "node_name": "Node"},
            title="Node Execution Latency",
        )
        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["text_primary"], "family": "Inter"},
            showlegend=False,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Cost breakdown pie chart
    if "estimated_cost_usd" in nt_df.columns:
        cost_df = nt_df.groupby("node_name")["estimated_cost_usd"].sum().reset_index()
        cost_df = cost_df[cost_df["estimated_cost_usd"] > 0]

        if not cost_df.empty:
            st.markdown("#### 💰 Cost Breakdown by Node")

            fig_cost = px.pie(
                cost_df,
                values="estimated_cost_usd",
                names="node_name",
                color_discrete_sequence=[
                    COLORS["accent_teal"],
                    COLORS["accent_blue"],
                    COLORS["accent_orange"],
                    COLORS["accent_purple"],
                    COLORS["accent_pink"],
                ],
                title="Token Cost Distribution",
            )
            fig_cost.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text_primary"], "family": "Inter"},
                margin={"l": 0, "r": 0, "t": 40, "b": 0},
            )
            fig_cost.update_traces(
                textposition="inside",
                textinfo="percent+label",
                marker={"line": {"color": COLORS["bg_primary"], "width": 2}},
            )
            st.plotly_chart(fig_cost, use_container_width=True)

# ────────────────────────── Tool Call Stats ────────────────────────────────────

if tool_stats:
    st.markdown("#### 🔧 Tool Call Statistics")

    ts_df = pd.DataFrame(tool_stats)
    ts_df = ts_df.rename(
        columns={
            "tool_name": "Tool",
            "total_calls": "Total Calls",
            "success_count": "Successes",
            "avg_latency_ms": "Avg Latency (ms)",
            "total_cost_usd": "Total Cost ($)",
        }
    )

    if "Successes" in ts_df.columns and "Total Calls" in ts_df.columns:
        ts_df["Success Rate"] = ts_df.apply(
            lambda row: (
                f"{row['Successes'] / row['Total Calls'] * 100:.0f}%"
                if row["Total Calls"] > 0
                else "—"
            ),
            axis=1,
        )

    if "Avg Latency (ms)" in ts_df.columns:
        ts_df["Avg Latency (ms)"] = ts_df["Avg Latency (ms)"].apply(
            lambda x: f"{x:.1f}" if x else "—"
        )

    if "Total Cost ($)" in ts_df.columns:
        ts_df["Total Cost ($)"] = ts_df["Total Cost ($)"].apply(lambda x: f"${x:.6f}" if x else "—")

    st.dataframe(ts_df, use_container_width=True, hide_index=True)

# ────────────────────────── No Data Fallback ──────────────────────────────────

if not node_timings and not tool_stats:
    st.info(
        "No detailed observability data available for this run. "
        "Ensure the agent nodes are logging to the tracer."
    )
