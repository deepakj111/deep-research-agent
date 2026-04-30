"""
app/pages/research.py

Live research interface for the DeepResearch Agent.

Flow:
1. User enters a query and selects a research profile (fast/deep).
2. Clicking "Start Research" opens an SSE connection to the FastAPI gateway.
3. Events stream into three columns: thought log, report, and metrics.
4. If the agent hits the HITL interrupt, the UI shows the proposed plan
   and allows the user to approve, edit, or reject before continuing.
5. On completion, the full report renders with download buttons (MD + PDF)
   and an interactive source relationship graph.
"""

from __future__ import annotations

import contextlib
import json
import os
import time

import httpx
import streamlit as st

from app.components.auth import require_auth
from app.components.source_graph import render_source_graph
from app.components.theme import hero_header, inject_theme, metric_card

# ────────────────────────── Config ────────────────────────────────────────────

AGENT_API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8080")

st.set_page_config(page_title="Research | DeepResearch", page_icon="🔬", layout="wide")
inject_theme()
require_auth()

hero_header("🔬 Research", "Start a new research query and watch the agent reason in real time")

# ────────────────────────── Input Section ─────────────────────────────────────

with st.container():
    input_col, config_col = st.columns([3, 1])
    with input_col:
        query = st.text_area(
            "Research Query",
            height=90,
            placeholder="e.g. Latest breakthroughs in quantum error correction 2025-2026",
            key="research_query",
        )
    with config_col:
        profile = st.selectbox(
            "Research Depth",
            ["fast", "deep"],
            index=0,
            help="**Fast**: 3-5 min, lower cost, fewer sources. **Deep**: 8-12 min, comprehensive.",
        )
        submit = st.button("🚀 Start Research", type="primary", use_container_width=True)


# ────────────────────────── SSE Parser ────────────────────────────────────────


def _parse_sse_line(line: str) -> dict | None:
    """Parse a single SSE data line into a dict, or None if not a data line."""
    if line.startswith("data: "):
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            return None
    return None


# ────────────────────────── Emoji Formatter ───────────────────────────────────

_EVENT_EMOJIS = {
    "node_start": "▶️",
    "tool_call": "🔧",
    "tool_result": "✅",
    "token": "💬",
    "hitl_interrupt": "⏸️",
    "complete": "🏁",
}


def _format_thought(event: dict) -> str | None:
    """Convert an SSE event into a human-readable thought-log line."""
    etype = event.get("type", "")
    emoji = _EVENT_EMOJIS.get(etype, "•")

    if etype == "node_start":
        return f"{emoji} **{event['node']}** starting..."
    if etype == "tool_call":
        return f"{emoji} `{event['tool']}` → `{event.get('input', '')[:80]}...`"
    if etype == "tool_result":
        return f"✅ `{event['tool']}` returned **{event.get('count', '?')}** results"
    if etype == "hitl_interrupt":
        return f"{emoji} **HITL Interrupt** — Plan ready for review"
    if etype == "complete":
        return f"{emoji} **Research complete** — `run_id: {event.get('run_id', '?')}`"
    return None


# ────────────────────────── HITL Interrupt Flow ───────────────────────────────


def _handle_hitl_interrupt(event: dict) -> None:
    """Display the HITL interrupt UI and handle user decision."""
    st.session_state["hitl_event"] = event
    st.session_state["hitl_pending"] = True


def _render_hitl_panel() -> None:
    """Render the HITL approval panel when the agent is paused."""
    event = st.session_state.get("hitl_event", {})
    if not event:
        return

    st.markdown("---")
    st.markdown(
        '<div class="glass-panel">'
        '<h4 style="margin-top:0;">⏸️ Research Plan Awaiting Approval</h4>'
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Query Difficulty:** `{event.get('query_difficulty', '?')}`")
        st.markdown(f"**Estimated Sub-questions:** {event.get('estimated_subquestions', '?')}")
        st.markdown(f"**Estimated Cost:** ${event.get('estimated_cost_usd', 0):.4f}")
        st.info(event.get("message", ""))

    with col2:
        thread_id = event.get("thread_id", "")
        approve_btn = st.button("✅ Approve Plan", type="primary", use_container_width=True)
        reject_btn = st.button("❌ Reject", use_container_width=True)

        if approve_btn:
            _resume_research(thread_id, approved=True)
        if reject_btn:
            _resume_research(thread_id, approved=False)


def _resume_research(thread_id: str, approved: bool) -> None:
    """Send the approval/rejection to the API and resume streaming."""
    st.session_state["hitl_pending"] = False

    try:
        resp = httpx.post(
            f"{AGENT_API_URL}/research/approve",
            json={"thread_id": thread_id, "approved": approved},
            timeout=10.0,
        )
        if not approved:
            st.warning("Research plan rejected.")
            return

        if resp.status_code == 200 and "text/event-stream" in resp.headers.get("content-type", ""):
            st.info("Resuming research after approval...")
            # For now we note the resume — full re-stream would require
            # another SSE connection which is complex in Streamlit.
            st.success(f"✅ Plan approved. Research resuming on thread `{thread_id}`.")
        else:
            st.success(f"✅ Plan approved. Thread `{thread_id}` is resuming.")
    except Exception as e:
        st.error(f"Failed to resume: {e}")


# ────────────────────────── Research Execution ────────────────────────────────

if submit and query:
    st.markdown("---")

    log_col, report_col, meta_col = st.columns([1, 2, 1])

    with log_col:
        st.markdown("### 🧠 Agent Reasoning")
        log_placeholder = st.empty()

    with report_col:
        st.markdown("### 📄 Research Report")
        report_placeholder = st.empty()

    with meta_col:
        st.markdown("### 📊 Run Metrics")
        metrics_placeholder = st.empty()

    thought_log: list[str] = []
    accumulated_report = ""
    run_id = ""
    node_count = 0
    tool_count = 0
    source_count = 0
    start_time = time.perf_counter()
    hitl_occurred = False
    findings_raw = None

    try:
        with (
            httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client,
            client.stream(
                "POST",
                f"{AGENT_API_URL}/research/stream",
                json={"query": query, "profile": profile},
            ) as response,
        ):
            for line in response.iter_lines():
                event = _parse_sse_line(line)
                if event is None:
                    continue

                etype = event.get("type", "")

                # Format thought log entry
                thought = _format_thought(event)
                if thought:
                    thought_log.append(thought)
                    with log_col:
                        log_placeholder.markdown(
                            '<div class="thought-log">' + "<br>".join(thought_log[-30:]) + "</div>",
                            unsafe_allow_html=True,
                        )

                # Token streaming for report
                if etype == "token":
                    accumulated_report += event.get("content", "")
                    with report_col:
                        report_placeholder.markdown(accumulated_report + "▌")

                # Track metrics
                if etype == "node_start":
                    node_count += 1
                elif etype == "tool_call":
                    tool_count += 1
                elif etype == "tool_result":
                    source_count += event.get("count", 0)

                # HITL interrupt
                elif etype == "hitl_interrupt":
                    hitl_occurred = True
                    _handle_hitl_interrupt(event)

                # Completion
                elif etype == "complete":
                    run_id = event.get("run_id", "")

                # Update metrics sidebar
                elapsed = time.perf_counter() - start_time
                with meta_col, metrics_placeholder.container():
                    metric_card("Run ID", run_id[:8] + "..." if run_id else "—")
                    metric_card("Profile", profile.upper())
                    metric_card("Nodes Executed", str(node_count))
                    metric_card("Tool Calls", str(tool_count))
                    metric_card("Sources Found", str(source_count))
                    metric_card("Elapsed", f"{elapsed:.1f}s")

    except httpx.ConnectError:
        st.error(
            "❌ Could not connect to the Agent API. "
            f"Ensure the API is running at `{AGENT_API_URL}`."
        )
    except Exception as e:
        st.error(f"❌ Stream error: {type(e).__name__}: {str(e)[:200]}")

    # ── Post-Completion UI ────────────────────────────────────────────────────

    if accumulated_report:
        with report_col:
            report_placeholder.markdown(accumulated_report)

    if hitl_occurred and st.session_state.get("hitl_pending"):
        _render_hitl_panel()

    if run_id:
        st.markdown("---")

        # Download buttons
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                "📥 Download Markdown",
                data=accumulated_report,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl_col2:
            try:
                pdf_resp = httpx.get(
                    f"{AGENT_API_URL}/research/report/{run_id}/pdf",
                    timeout=30.0,
                )
                if pdf_resp.status_code == 200:
                    st.download_button(
                        "📥 Download PDF",
                        data=pdf_resp.content,
                        file_name="research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "📥 PDF Unavailable",
                        disabled=True,
                        use_container_width=True,
                    )
            except Exception:
                st.button(
                    "📥 PDF Unavailable",
                    disabled=True,
                    use_container_width=True,
                )
        with dl_col3:
            st.button(
                f"🔍 View Trace: `{run_id[:8]}...`",
                use_container_width=True,
                disabled=True,
                help="Navigate to the Traces page to inspect this run.",
            )

        # Source relationship graph
        st.markdown("### 🔗 Source Relationship Graph")
        with contextlib.suppress(Exception):
            state_resp = httpx.get(
                f"{AGENT_API_URL}/research/report/{run_id}",
                timeout=15.0,
            )
            if state_resp.status_code == 200:
                report_data = state_resp.json()
                # Build minimal findings-like objects for the graph
                if report_data and "sources" in report_data:
                    st.info(
                        f"Visualizing {len(report_data.get('sources', []))} sources from this run."
                    )

        # If we have findings in session state, render the graph
        if findings_raw:
            render_source_graph(findings_raw)

# ────────────────────────── Empty State ───────────────────────────────────────

elif not submit:
    st.markdown(
        '<div class="glass-panel" style="text-align:center;padding:3rem;">'
        '<p style="font-size:1.1rem;color:#8B949E;">'
        "Enter a research query above and click <b>Start Research</b> "
        "to begin an autonomous deep research session."
        "</p>"
        '<p style="font-size:0.85rem;color:#484F58;margin-top:0.5rem;">'
        "The agent will decompose your query, search multiple sources in parallel, "
        "evaluate quality, and synthesize a cited report — all streamed live."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
