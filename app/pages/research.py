"""
app/pages/research.py

Full research interface for the DeepResearch Agent.

Architecture:
- LEFT SIDEBAR: History panel showing all past runs as clickable cards.
  Clicking a card loads the run report in the main area.
  A "New Research" button at the top opens the query input form.

- RIGHT MAIN PANEL:
  - New Query mode: query input + live SSE streaming (thought log, report, metrics)
  - View mode: renders the saved report for a selected past run
  - HITL Popup: uses st.dialog for a real modal overlay for plan approval

Key design principle:
  All live-streaming state (thought_log, accumulated_report, run_id) is stored
  in st.session_state keyed by run_id so it survives re-renders and refreshes.
"""

from __future__ import annotations

import contextlib
import html
import json
import os
import time
from typing import Any

import httpx
import streamlit as st

from app.components.auth import require_auth
from app.components.theme import format_ist, hero_header, inject_theme, metric_card

# ────────────────────────── Config ────────────────────────────────────────────

AGENT_API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8080")

try:
    from config.settings import settings
    from utils.cost_estimator import estimate_cost

    _OUTPUT_COST_PER_TOKEN = estimate_cost(settings.default_model, 0, 1_000_000) / 1_000_000
except Exception:
    _OUTPUT_COST_PER_TOKEN = 5.00 / 1_000_000
try:
    _AVG_TOKENS_PER_CHAR = settings.avg_tokens_per_char
except Exception:
    _AVG_TOKENS_PER_CHAR = 0.25

st.set_page_config(
    page_title="Research | DeepResearch",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()
require_auth()


# ────────────────────────── Session State Helpers ─────────────────────────────


def _init_session():
    """Initialise all required session_state keys if they don't exist."""
    if "active_run_id" not in st.session_state:
        st.session_state.active_run_id = None
    if "mode" not in st.session_state:
        st.session_state.mode = "new"  # "new" | "view" | "streaming"
    if "runs_cache" not in st.session_state:
        st.session_state.runs_cache = []
    if "run_states" not in st.session_state:
        st.session_state.run_states = {}  # keyed by run_id


def _get_run_state(run_id: str) -> dict:
    """Return (or create) the per-run state dict stored in session_state."""
    if run_id not in st.session_state.run_states:
        st.session_state.run_states[run_id] = {
            "thought_log": [],
            "accumulated_report": "",
            "node_count": 0,
            "tool_count": 0,
            "source_count": 0,
            "token_count": 0,
            "start_time": None,
            "hitl_event": None,
            "status": "running",
        }
    return st.session_state.run_states[run_id]


# ────────────────────────── API Helpers ───────────────────────────────────────


def _fetch_runs() -> list[dict]:
    try:
        resp = httpx.get(f"{AGENT_API_URL}/research/runs", params={"limit": 50}, timeout=8.0)
        if resp.status_code == 200:
            return resp.json().get("runs", [])
    except Exception:
        pass
    return []


def _fetch_report(run_id: str) -> dict | None:
    try:
        resp = httpx.get(f"{AGENT_API_URL}/research/report/{run_id}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _fetch_run_detail(run_id: str) -> dict | None:
    try:
        resp = httpx.get(f"{AGENT_API_URL}/research/runs/{run_id}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ────────────────────────── SSE Parsing ───────────────────────────────────────


def _parse_sse_line(line: str) -> dict | None:
    if line.startswith("data: "):
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            return None
    return None


_EVENT_EMOJIS = {
    "node_start": "🔄",
    "node_end": "✅",
    "llm_start": "🤖",
    "llm_end": "💡",
    "tool_call": "🔧",
    "tool_result": "📄",
    "hitl_interrupt": "⏸️",
    "complete": "🏁",
}

_NODE_LABELS = {
    "classifier": "Query Classifier",
    "planner": "Research Planner",
    "supervisor": "Smart Supervisor",
    "web_agent": "Web Search Agent",
    "arxiv_agent": "arXiv Academic Agent",
    "github_agent": "GitHub Code Agent",
    "critic": "Critic & Quality Judge",
    "synthesizer": "Multi-Model Synthesizer",
    "writer": "Report Writer & Grounding Verifier",
}

_NODE_DESCRIPTIONS = {
    "classifier": "Analyzing query difficulty & selecting optimal data sources...",
    "planner": "Decomposing query into targeted sub-questions...",
    "supervisor": "Orchestrating smart parallel fan-out to selected tools...",
    "web_agent": "Executing live web search queries via Tavily MCP...",
    "arxiv_agent": "Fetching & parsing relevant academic research papers...",
    "github_agent": "Searching open-source repositories & code implementations...",
    "critic": "Evaluating source diversity, recency & depth quality thresholds...",
    "synthesizer": "Synthesizing evidence & identifying potential contradictions...",
    "writer": "Cross-verifying citations & compiling final report...",
}


def _highlight_json_obj(obj: Any, indent: int = 0) -> str:
    """Format Python object into syntax-highlighted HTML with clean spacing."""
    pad = "&nbsp;&nbsp;" * indent
    if isinstance(obj, dict):
        if not obj:
            return '<span class="json-bracket">{}</span>'
        lines = ['<span class="json-bracket">{</span>']
        items = list(obj.items())
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            key_html = f'<span class="json-key">"{html.escape(str(k))}"</span>: '
            val_html = _highlight_json_obj(v, indent + 1)
            lines.append(f"{pad}&nbsp;&nbsp;{key_html}{val_html}{comma}")
        lines.append(f'{pad}<span class="json-bracket">}}</span>')
        return "<br>".join(lines)
    elif isinstance(obj, list):
        if not obj:
            return '<span class="json-bracket">[]</span>'
        lines = ['<span class="json-bracket">[</span>']
        for i, item in enumerate(obj):
            comma = "," if i < len(obj) - 1 else ""
            val_html = _highlight_json_obj(item, indent + 1)
            lines.append(f"{pad}&nbsp;&nbsp;{val_html}{comma}")
        lines.append(f'{pad}<span class="json-bracket">]</span>')
        return "<br>".join(lines)
    elif isinstance(obj, str):
        escaped = html.escape(obj)
        if "\n" in escaped:
            escaped = f"<br>{pad}&nbsp;&nbsp;".join(escaped.split("\n"))
        return f'<span class="json-string">"{escaped}"</span>'
    elif isinstance(obj, bool):
        return f'<span class="json-boolean">{str(obj).lower()}</span>'
    elif obj is None:
        return '<span class="json-null">null</span>'
    elif isinstance(obj, (int, float)):
        return f'<span class="json-number">{obj}</span>'
    else:
        return f'<span class="json-string">"{html.escape(str(obj))}"</span>'


def _format_payload_view(payload_raw: Any) -> str:
    """Format any raw payload (JSON string, dict, list, prompt) into visually rich HTML."""
    if not payload_raw:
        return ""

    parsed = None
    if isinstance(payload_raw, (dict, list)):
        parsed = payload_raw
    elif isinstance(payload_raw, str):
        s = payload_raw.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None

    chips_html = ""
    if isinstance(parsed, dict):
        pills = []
        for key in [
            "query",
            "query_difficulty",
            "relevant_sources",
            "subquestions",
            "node",
            "tool",
            "count",
            "coverage_score",
            "should_continue",
        ]:
            if key in parsed:
                val = parsed[key]
                val_str = ", ".join(str(x) for x in val) if isinstance(val, list) else str(val)
                if len(val_str) > 65:
                    val_str = val_str[:62] + "..."
                pills.append(
                    f'<span class="summary-pill">'
                    f'<span class="pill-key">{key}:</span> '
                    f'<span class="pill-val">{html.escape(val_str)}</span>'
                    f"</span>"
                )
        if pills:
            chips_html = f'<div class="summary-pills-row">{"".join(pills)}</div>'

    if parsed is not None:
        json_tree = _highlight_json_obj(parsed)
        return f'{chips_html}<div class="json-view-container">{json_tree}</div>'

    text = str(payload_raw)
    if "[SYSTEM]" in text or "[USER]" in text or "[HUMAN]" in text or "[ASSISTANT]" in text:
        import re

        parts = []
        role_blocks = re.split(r"(\[(?:SYSTEM|USER|HUMAN|ASSISTANT)\])", text)
        current_role = "user"
        for block in role_blocks:
            b = block.strip()
            if not b:
                continue
            if b in ("[SYSTEM]", "[USER]", "[HUMAN]", "[ASSISTANT]"):
                current_role = b.strip("[]").lower()
                if current_role in ("user", "human"):
                    current_role = "user"
                elif current_role in ("system", "assistant"):
                    current_role = "system"
            else:
                role_class = "system" if current_role == "system" else "user"
                role_title = "🤖 SYSTEM PROMPT" if current_role == "system" else "👤 USER PROMPT"
                escaped_b = html.escape(b).replace("\n", "<br>")
                parts.append(
                    f'<div class="prompt-role-card {role_class}">'
                    f'<span class="role-badge {role_class}">{role_title}</span>'
                    f'<div style="font-size:0.82rem;line-height:1.4;color:#1f2328;">{escaped_b}</div>'
                    f"</div>"
                )
        if parts:
            return "".join(parts)

    if "[1]" in text and ("URL:" in text or "Snippet:" in text):
        import re

        items_html = []
        items = re.split(r"\[\d+\]\s+", text)
        for item in items:
            item = item.strip()
            if not item:
                continue
            lines_list = item.split("\n")
            title_line = lines_list[0]
            url_match = re.search(r"URL:\s*(https?://[^\s]+)", item)
            snippet_match = re.search(r"Snippet:\s*(.+)", item, re.DOTALL)

            url = url_match.group(1) if url_match else "#"
            snippet = snippet_match.group(1).strip() if snippet_match else ""
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."

            items_html.append(
                f'<div class="tool-result-item">'
                f'<div><a href="{url}" target="_blank" class="tool-result-title">{html.escape(title_line)}</a></div>'
                f'<div class="tool-result-url">🔗 {html.escape(url)}</div>'
                f'<div class="tool-result-snippet">{html.escape(snippet)}</div>'
                f"</div>"
            )
        if items_html:
            return "".join(items_html)

    escaped_text = html.escape(text).replace("\n", "<br>")
    return f'<div class="json-view-container" style="white-space:pre-wrap;">{escaped_text}</div>'


def _render_status_bar(
    status_text: str,
    start_time: float | None,
    is_completed: bool = False,
    elapsed_frozen: float | None = None,
) -> str:
    """Render a single-line status bar with a ticking or frozen stopwatch."""
    if is_completed and elapsed_frozen is not None:
        elapsed_sec = int(elapsed_frozen)
        badge_class = "stopwatch-badge stopped"
        badge_icon = "🏁 PDF Generated"
    else:
        start_t = start_time or time.perf_counter()
        elapsed_sec = int(max(0, time.perf_counter() - start_t))
        badge_class = "stopwatch-badge running"
        badge_icon = "⏱️ Running"

    mins, secs = divmod(elapsed_sec, 60)
    time_str = f"{mins:02d}:{secs:02d}"

    return (
        f'<div class="status-bar-container">'
        f'<div class="status-bar-text">'
        f'<span style="font-size:1.1rem;">⚡</span>'
        f"<span>Current Status: <b>{html.escape(status_text)}</b></span>"
        f"</div>"
        f'<div class="{badge_class}">{badge_icon} — {time_str}</div>'
        f"</div>"
    )


def _format_thought_card(event: dict) -> str | None:
    """Convert an SSE event into a styled HTML thought-card div with full details."""
    etype = event.get("type", "")
    if etype not in _EVENT_EMOJIS:
        return None
    emoji = _EVENT_EMOJIS[etype]
    ts_raw = event.get("timestamp")
    ts = format_ist(ts_raw, fmt="%I:%M:%S %p IST")

    if etype == "node_start":
        node = event.get("node", "Agent")
        label = _NODE_LABELS.get(node, node)
        desc = _NODE_DESCRIPTIONS.get(node, "Processing step...")
        inp = event.get("input", "")
        inp_html = ""
        if inp:
            payload_html = _format_payload_view(inp)
            inp_html = (
                f"<details open>"
                f"<summary>📥 Phase Input State & Task Parameters</summary>"
                f"{payload_html}"
                f"</details>"
            )
        return (
            f'<div class="thought-card node_start">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">{emoji}</span>'
            f"<span>{label} — Phase Started</span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f'<div class="thought-card-body">{desc}{inp_html}</div>'
            f"</div>"
        )

    if etype == "node_end":
        node = event.get("node", "Agent")
        label = _NODE_LABELS.get(node, node)
        out = event.get("output", "")
        payload_html = _format_payload_view(out)
        return (
            f'<div class="thought-card node_end">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">✅</span>'
            f"<span>{label} — Phase Output Completed</span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f'<div class="thought-card-body">'
            f"<details open>"
            f"<summary>⚡ Phase Output, Findings & State Update</summary>"
            f"{payload_html}"
            f"</details>"
            f"</div>"
            f"</div>"
        )
    if etype == "llm_start":
        model = event.get("model", "LLM")
        prompt = event.get("prompt", "")
        payload_html = _format_payload_view(prompt)
        return (
            f'<div class="thought-card llm_start">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">{emoji}</span>'
            f"<span>LLM Call — <b>{model}</b></span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f'<div class="thought-card-body">'
            f"<details open>"
            f"<summary>💬 Input Question / System & User Messages</summary>"
            f"{payload_html}"
            f"</details>"
            f"</div>"
            f"</div>"
        )
    if etype == "llm_end":
        model = event.get("model", "LLM")
        resp = event.get("response", "")
        payload_html = _format_payload_view(resp)
        return (
            f'<div class="thought-card llm_end">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">{emoji}</span>'
            f"<span>LLM Response — <b>{model}</b></span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f'<div class="thought-card-body">'
            f"<details open>"
            f"<summary>⚡ LLM Generated Response & Reasoning</summary>"
            f"{payload_html}"
            f"</details>"
            f"</div>"
            f"</div>"
        )
    if etype == "tool_call":
        tool = event.get("tool", "tool")
        inp = event.get("input", "")
        payload_html = _format_payload_view(inp)
        return (
            f'<div class="thought-card tool_call">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">{emoji}</span>'
            f"<span>Invoking <b>{tool}</b></span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f'<div class="thought-card-body">'
            f"<details open>"
            f"<summary>🔍 Input Payload / Arguments</summary>"
            f"{payload_html}"
            f"</details>"
            f"</div>"
            f"</div>"
        )
    if etype == "tool_result":
        tool = event.get("tool", "tool")
        count = event.get("count", 0)
        full_output = event.get("full_output") or event.get("preview") or ""
        payload_html = _format_payload_view(full_output)
        return (
            f'<div class="thought-card tool_result">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">✅</span>'
            f"<span>{tool} — {count} results returned</span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f'<div class="thought-card-body">'
            f"<details open>"
            f"<summary>📄 Retrieved Data & Output Content ({count} items)</summary>"
            f"{payload_html}"
            f"</details>"
            f"</div>"
            f"</div>"
        )
    if etype == "hitl_interrupt":
        sources = event.get("relevant_sources", ["web"])
        sources_str = ", ".join(sources).upper()
        return (
            f'<div class="thought-card">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">{emoji}</span>'
            f"<span>Plan ready (Sources: {sources_str}) — awaiting approval</span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f"</div>"
        )
    if etype == "complete":
        return (
            f'<div class="thought-card complete">'
            f'<div class="thought-card-header">'
            f'<span class="thought-card-icon">{emoji}</span>'
            f"<span>Research Complete!</span>"
            f'<span style="margin-left:auto;font-size:0.72rem;opacity:0.6;">{ts}</span>'
            f"</div>"
            f"</div>"
        )
    return None


# ────────────────────────── HITL Dialog ───────────────────────────────────────


@st.dialog("⏸️ Research Plan — Awaiting Approval", width="large", dismissible=False)
def _hitl_dialog(hitl_event: dict, run_id: str):
    """Render the HITL approval as a real st.dialog modal popup."""
    difficulty = hitl_event.get("query_difficulty", "?")
    n_q = hitl_event.get("estimated_subquestions", "?")
    est_cost = hitl_event.get("estimated_cost_usd", 0)
    subquestions = hitl_event.get("subquestions", [])
    sources = hitl_event.get("relevant_sources", ["web"])

    source_labels = {
        "web": "🌐 Web Search (Tavily)",
        "arxiv": "📄 Academic Papers (arXiv)",
        "github": "⭐ Open-Source Code (GitHub)",
    }
    badges_html = " &nbsp; ".join([f"<code>{source_labels.get(s, s)}</code>" for s in sources])

    st.markdown(
        f"""
        The agent has analyzed your query and prepared a targeted research plan.

        | Detail | Value |
        |---|---|
        | **Query Difficulty** | `{difficulty}` |
        | **Targeted Data Sources** | {badges_html} |
        | **Sub-questions Planned** | {n_q} |
        | **Estimated Cost** | `${est_cost:.4f}` |
        """
    )

    if subquestions:
        st.markdown("**📋 Planned Sub-questions:**")
        for i, sq in enumerate(subquestions, 1):
            st.markdown(f"**{i}.** {sq}")
    st.info(hitl_event.get("message", ""))
    st.warning(
        "🔒 **Decision Required**: Select either **Approve & Continue** or **Reject & Cancel** to complete plan approval."
    )

    approve_col, reject_col = st.columns(2)
    with approve_col:
        if st.button(
            "✅ Approve & Continue Research",
            type="primary",
            use_container_width=True,
            key="hitl_approve",
        ):
            run_state = _get_run_state(run_id)
            run_state["hitl_event"] = None
            run_state["approved_to_resume"] = True
            st.session_state.active_run_id = run_id
            st.session_state.mode = "streaming"
            st.toast("✅ Plan approved — resuming research...", icon="🚀")
            st.rerun()
    with reject_col:
        if st.button("❌ Reject & Cancel", use_container_width=True, key="hitl_reject"):
            run_state = _get_run_state(run_id)
            run_state["hitl_event"] = None
            run_state["status"] = "rejected"
            run_state["approved_to_resume"] = False
            # Fire rejection to API
            with contextlib.suppress(Exception):
                httpx.post(
                    f"{AGENT_API_URL}/research/approve",
                    json={"thread_id": run_id, "approved": False},
                    timeout=5.0,
                )
            st.toast("Research plan rejected.", icon="⚠️")
            st.session_state.mode = "new"
            st.rerun()


# ────────────────────────── History Sidebar ────────────────────────────────────


def _status_pill(status: str) -> str:
    icons = {"running": "🔄", "completed": "✅", "failed": "❌", "rejected": "⚠️"}
    icon = icons.get(status, "•")
    return f'<span class="status-pill {status}">{icon} {status}</span>'


def _render_history_sidebar(runs: list[dict]):
    """Render the left-side history panel."""
    st.markdown(
        '<div style="padding: 0.5rem 0 1rem;">'
        '<h3 style="margin:0;font-size:1.1rem;font-weight:700;">🔬 DeepResearch</h3>'
        '<span style="font-size:0.75rem;color:#8B949E;">Autonomous Research Agent</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "➕ New Research", type="primary", use_container_width=True, key="new_research_btn"
    ):
        st.session_state.mode = "new"
        st.session_state.active_run_id = None
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.07em;'
        'color:#8B949E;margin-bottom:0.5rem;">Recent Runs</p>',
        unsafe_allow_html=True,
    )

    if not runs:
        st.caption("No runs yet. Start your first research query!")
        return

    for run in runs:
        run_id = run.get("run_id", "")
        query = run.get("query", "Unknown query")
        status = run.get("status", "unknown")
        cost = run.get("total_cost_usd")
        started = (
            format_ist(run.get("started_at"), fmt="%d %b %I:%M %p IST")
            if run.get("started_at")
            else ""
        )

        is_active = run_id == st.session_state.active_run_id
        card_class = "run-card active" if is_active else "run-card"
        query_short = query[:55] + "..." if len(query) > 55 else query

        cost_str = f"${cost:.4f}" if cost else "—"
        pill = _status_pill(status)

        st.markdown(
            f'<div class="{card_class}">'
            f'<div class="run-card-query">{query_short}</div>'
            f'<div class="run-card-meta">{pill} &nbsp;·&nbsp; {cost_str} &nbsp;·&nbsp; {started}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Open", key=f"open_{run_id}", use_container_width=True, help=f"Open run {run_id[:8]}"
        ):
            st.session_state.active_run_id = run_id
            st.session_state.mode = "view"
            st.rerun()

    st.markdown("---")
    st.caption("🔒 JWT Auth · 🧠 LangGraph · 🔧 3 MCP Servers")


# ────────────────────────── New Research Panel ────────────────────────────────


def _render_new_research():
    """Render the query input form and handle SSE streaming."""
    hero_header("🔬 Start Research", "Ask a complex question — the agent will do the deep work")

    with st.form("research_form", clear_on_submit=False):
        query = st.text_area(
            "Your Research Question",
            height=110,
            placeholder=(
                "e.g. Analyze the global demand and investment outlook for silver through 2026, "
                "including industrial usage in solar panels and EVs, macroeconomic drivers, "
                "and supply deficit projections..."
            ),
            key="query_input",
        )
        submitted = st.form_submit_button(
            "🚀 Start Research", type="primary", use_container_width=True
        )

    if submitted and query.strip():
        _run_research(query.strip(), profile="deep")


def _fetch_formatted_report(run_id: str) -> str:
    """Fetch the formatted markdown report from the API after completion."""
    try:
        resp = httpx.get(
            f"{AGENT_API_URL}/research/report/{run_id}/markdown",
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return ""


def _fetch_pdf_bytes(run_id: str) -> bytes | None:
    """Fetch the PDF report from the API."""
    try:
        resp = httpx.get(
            f"{AGENT_API_URL}/research/report/{run_id}/pdf",
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


def _render_final_report(run_id: str, report_ph, run_state: dict) -> None:
    """Fetch and render the clean formatted report from the API."""
    md_content = _fetch_formatted_report(run_id)
    if md_content:
        run_state["final_report_md"] = md_content
        report_ph.markdown(md_content)
    else:
        # Fallback to accumulated token stream if API fetch fails
        if run_state.get("accumulated_report"):
            report_ph.markdown(run_state["accumulated_report"])


def _render_download_buttons(run_id: str, run_state: dict) -> None:
    """Render the report download buttons after a completed run."""
    report_md = run_state.get("final_report_md") or run_state.get("accumulated_report", "")
    if not report_md:
        return

    st.markdown("---")
    st.markdown("### 📥 Download Report")
    dl1, dl2 = st.columns(2)

    # Primary: PDF download
    with dl1:
        pdf_bytes = _fetch_pdf_bytes(run_id)
        if pdf_bytes:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"research_report_{run_id[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.button(
                "📄 PDF Unavailable",
                disabled=True,
                use_container_width=True,
                help="PDF generation requires WeasyPrint to be installed.",
            )

    # Secondary: Markdown download
    with dl2:
        st.download_button(
            "📝 Download Markdown",
            data=report_md,
            file_name=f"research_report_{run_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _run_research(query: str, profile: str):
    """Open SSE stream, populate run state in session_state, and render live UI."""
    import uuid

    placeholder_id = str(uuid.uuid4())
    run_state = _get_run_state(placeholder_id)
    run_state["start_time"] = time.perf_counter()
    run_state["final_report_md"] = ""
    st.session_state.active_run_id = placeholder_id
    st.session_state.mode = "streaming"

    status_ph = st.empty()
    status_text = "Initializing research pipeline..."
    status_ph.markdown(
        _render_status_bar(status_text, run_state["start_time"]),
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Full Agent Chain & Thought Process Trace")
    log_ph = st.empty()

    st.markdown("### 📄 Research Report")
    report_ph = st.empty()
    report_ph.info("⏳ Research in progress — report will appear here once complete...")

    st.markdown("### 📊 Live Execution Metrics")
    metrics_ph = st.empty()

    real_run_id = placeholder_id
    hitl_event = None
    completed = False

    try:
        with (
            httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client,
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

                # Build thought card
                card = _format_thought_card(event)
                if card:
                    run_state["thought_log"].append(card)
                    log_ph.markdown(
                        '<div class="thought-log">' + "".join(run_state["thought_log"]) + "</div>",
                        unsafe_allow_html=True,
                    )

                # Token streaming — accumulate silently for fallback, do NOT show raw to user
                if etype == "token":
                    content = event.get("content", "")
                    run_state["accumulated_report"] += content
                    run_state["token_count"] += max(1, int(len(content) * _AVG_TOKENS_PER_CHAR))

                # Metric tracking & activity update
                if etype == "node_start":
                    run_state["node_count"] += 1
                    node = event.get("node", "Agent")
                    label = _NODE_LABELS.get(node, node)
                    desc = _NODE_DESCRIPTIONS.get(node, "Processing step...")
                    status_text = f"{label} — {desc}"
                    status_ph.markdown(
                        _render_status_bar(status_text, run_state["start_time"]),
                        unsafe_allow_html=True,
                    )
                    st.toast(f"🔄 {label} started", icon="🔄")
                elif etype == "tool_call":
                    run_state["tool_count"] += 1
                    tool = event.get("tool", "tool")
                    status_text = f"Invoking {tool}..."
                    status_ph.markdown(
                        _render_status_bar(status_text, run_state["start_time"]),
                        unsafe_allow_html=True,
                    )
                elif etype == "tool_result":
                    run_state["source_count"] += event.get("count", 0)
                    tool = event.get("tool", "tool")
                    count = event.get("count", 0)
                    status_text = f"Retrieved {count} results from {tool}"
                    status_ph.markdown(
                        _render_status_bar(status_text, run_state["start_time"]),
                        unsafe_allow_html=True,
                    )
                    st.toast(f"✅ {tool} returned results", icon="✅")
                elif etype == "hitl_interrupt":
                    hitl_event = event
                    run_state["hitl_event"] = event
                    status_text = "Research Paused — Awaiting Plan Approval"
                    status_ph.markdown(
                        _render_status_bar(status_text, run_state["start_time"]),
                        unsafe_allow_html=True,
                    )
                    server_thread_id = event.get("thread_id", "")
                    if server_thread_id and server_thread_id != placeholder_id:
                        real_run_id = server_thread_id
                        st.session_state.run_states[real_run_id] = st.session_state.run_states.pop(
                            placeholder_id, run_state
                        )
                        run_state = st.session_state.run_states[real_run_id]
                        st.session_state.active_run_id = real_run_id
                elif etype == "complete":
                    real_run_id = event.get("run_id", placeholder_id)
                    run_state["status"] = "completed"
                    completed = True
                    elapsed_frozen = time.perf_counter() - (
                        run_state["start_time"] or time.perf_counter()
                    )
                    run_state["elapsed_frozen"] = elapsed_frozen
                    status_text = "Research Complete! PDF Report Generated."
                    status_ph.markdown(
                        _render_status_bar(
                            status_text,
                            run_state["start_time"],
                            is_completed=True,
                            elapsed_frozen=elapsed_frozen,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.toast("🏁 Research completed!", icon="🎉")

                # Live horizontal metrics row
                est_cost = run_state["token_count"] * _OUTPUT_COST_PER_TOKEN
                with metrics_ph.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1:
                        metric_card("Nodes", str(run_state["node_count"]))
                    with m2:
                        metric_card("Tool Calls", str(run_state["tool_count"]))
                    with m3:
                        metric_card("Sources", str(run_state["source_count"]))
                    with m4:
                        metric_card("Tokens", f"{run_state['token_count']:,}")
                    with m5:
                        metric_card("Est. Cost", f"${est_cost:.4f}")

    except httpx.ConnectError:
        st.error(f"❌ Could not connect to the Agent API at `{AGENT_API_URL}`.")
        return
    except Exception as e:
        st.error(f"❌ Stream error: {type(e).__name__}: {str(e)[:300]}")
        return

    # Move state to the real run_id (if not already done via hitl_interrupt)
    if real_run_id != placeholder_id and placeholder_id in st.session_state.run_states:
        st.session_state.run_states[real_run_id] = st.session_state.run_states.pop(placeholder_id)
        st.session_state.active_run_id = real_run_id
        run_state = st.session_state.run_states[real_run_id]

    # Check if report is ready on server even if complete SSE event was missed
    if not completed and _fetch_formatted_report(real_run_id):
        run_state["status"] = "completed"
        completed = True

    # Refresh history sidebar
    st.session_state.runs_cache = _fetch_runs()

    # Show HITL dialog if interrupted
    if hitl_event:
        run_state["hitl_event"] = hitl_event
        _hitl_dialog(hitl_event, real_run_id)

    # After completion, switch to view mode and rerun automatically so user sees final report & PDF instantly
    if completed:
        st.session_state.mode = "view"
        st.session_state.active_run_id = real_run_id
        _render_final_report(real_run_id, report_ph, run_state)
        st.rerun()


# ────────────────────────── View Past Run ─────────────────────────────────────


def _render_run_viewer(run_id: str, runs: list[dict]):
    """Render a completed or in-progress run loaded from the API."""
    # Check if we have local streaming state for this run
    local_state = st.session_state.run_states.get(run_id)

    # If state is missing (e.g. after app restart), reconstruct from API graph state
    if not local_state or not local_state.get("thought_log"):
        try:
            resp = httpx.get(f"{AGENT_API_URL}/research/state/{run_id}", timeout=8.0)
            if resp.status_code == 200:
                gstate = resp.json()
                local_state = _get_run_state(run_id)
                if gstate.get("thought_log"):
                    local_state["thought_log"] = gstate["thought_log"]
                if gstate.get("findings_count"):
                    local_state["source_count"] = gstate["findings_count"]
                if gstate.get("has_final_report"):
                    local_state["status"] = "completed"
        except Exception:
            pass

    run_meta = next((r for r in runs if r.get("run_id") == run_id), {})
    query = run_meta.get("query", "Unknown query")
    status = run_meta.get("status", "unknown")

    # Check if report is available to override status if DB had not updated yet
    formatted_report_text = _fetch_formatted_report(run_id)
    if formatted_report_text or (local_state and local_state.get("status") == "completed"):
        status = "completed"
        if local_state:
            local_state["status"] = "completed"

    st.markdown(
        f'<div class="white-panel">'
        f'<h3 style="margin-top:0;color:#0969da;">📋 {html.escape(query)}</h3>'
        f"{_status_pill(status)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    status_str = (
        "Research Complete! PDF Report Generated."
        if status == "completed"
        else f"Run Status: {status.title()}"
    )
    elapsed_frozen = local_state.get("elapsed_frozen") if local_state else None
    st.markdown(
        _render_status_bar(
            status_str,
            local_state.get("start_time") if local_state else None,
            is_completed=(status == "completed"),
            elapsed_frozen=elapsed_frozen,
        ),
        unsafe_allow_html=True,
    )

    # Render thought log at top if present
    if local_state and local_state.get("thought_log"):
        st.markdown("### 🧠 Full Agent Chain & Thought Process Trace")
        st.markdown(
            '<div class="thought-log">' + "".join(local_state["thought_log"]) + "</div>",
            unsafe_allow_html=True,
        )

    # If we have accumulated report from live stream, show it
    if local_state and local_state.get("accumulated_report"):
        st.markdown("### 📄 Research Report")
        st.markdown(local_state["accumulated_report"])

        cost = local_state["token_count"] * _OUTPUT_COST_PER_TOKEN
        st.markdown("### 📊 Execution Metrics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Nodes", str(local_state["node_count"]))
        with m2:
            metric_card("Tool Calls", str(local_state["tool_count"]))
        with m3:
            metric_card("Sources", str(local_state["source_count"]))
        with m4:
            metric_card("Est. Cost", f"${cost:.4f}")

        # HITL re-trigger if still pending
        if local_state.get("hitl_event") and local_state.get("status") == "running":
            _hitl_dialog(local_state["hitl_event"], run_id)

    else:
        # Load from API — use the formatted markdown endpoint
        st.markdown("### 📄 Research Report")
        md_content = _fetch_formatted_report(run_id)

        if md_content:
            st.markdown(md_content)
        else:
            # No formatted report — check if the run is paused or still running
            graph_state = None
            try:
                resp = httpx.get(f"{AGENT_API_URL}/research/state/{run_id}", timeout=8.0)
                if resp.status_code == 200:
                    graph_state = resp.json()
            except Exception:
                pass

            detail = _fetch_run_detail(run_id)
            tracer_status = (
                (detail or {}).get("summary", {}).get("status", "unknown") if detail else "unknown"
            )

            if graph_state and "supervisor" in (graph_state.get("next_node") or []):
                # Run is paused at HITL planner — offer resume
                st.markdown(
                    '<div class="glass-panel">'
                    '<h4 style="margin-top:0;">⏸️ Research Paused — Awaiting Plan Approval</h4>'
                    "<p>This research run was interrupted before plan approval. "
                    "Click <b>Resume</b> to approve the plan and continue.</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                difficulty = graph_state.get("query_difficulty", "unknown")
                n_q = len(graph_state.get("subquestions", []))
                c1, c2 = st.columns([1, 2])
                with c1:
                    metric_card("Query Difficulty", difficulty)
                with c2:
                    metric_card("Sub-questions Ready", str(n_q))

                if st.button(
                    "▶️ Resume Research — Approve Plan",
                    type="primary",
                    use_container_width=True,
                    key="resume_btn",
                ):
                    fake_hitl = {
                        "thread_id": run_id,
                        "query_difficulty": difficulty,
                        "estimated_subquestions": n_q,
                        "estimated_cost_usd": 0.0,
                        "message": "Resuming previously paused research run.",
                    }
                    _hitl_dialog(fake_hitl, run_id)

            elif tracer_status == "running":
                st.info("🔄 Research execution in progress... Auto-refreshing live state.")
                # Automatically refresh every 2.5 seconds until complete
                time.sleep(2.5)
                st.session_state.runs_cache = _fetch_runs()
                st.rerun()
            else:
                st.warning(
                    f"No final report available for this run (status: `{tracer_status}`). "
                    "The run may have been interrupted or rejected."
                )

    # Download section
    st.markdown("---")
    st.markdown("### 📥 Download Report")
    dl1, dl2 = st.columns(2)
    with dl1:
        pdf_bytes = _fetch_pdf_bytes(run_id)
        if pdf_bytes:
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"research_report_{run_id[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        else:
            st.button(
                "📄 PDF Unavailable",
                disabled=True,
                use_container_width=True,
                help="PDF generation requires WeasyPrint.",
            )
    with dl2:
        md_text = _fetch_formatted_report(run_id)
        if md_text:
            st.download_button(
                "📝 Download Markdown",
                data=md_text,
                file_name=f"research_report_{run_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # Observability expander
    with st.expander("📊 Observability Details (node timings, tool stats)", expanded=False):
        detail = _fetch_run_detail(run_id)
        if detail:
            summary = detail.get("summary", {})
            node_timings = detail.get("node_timings", [])
            tool_stats = detail.get("tool_stats", [])
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Status", summary.get("status", "?").upper())
            with c2:
                cost = summary.get("total_cost_usd")
                metric_card("Cost", f"${cost:.4f}" if cost else "—")
            with c3:
                latency = summary.get("total_latency_ms")
                metric_card("Latency", f"{latency / 1000:.1f}s" if latency else "—")
            with c4:
                metric_card("Iterations", str(summary.get("iteration_count", "—")))

            if node_timings:
                import pandas as pd

                st.markdown("**Node Timings**")
                st.dataframe(pd.DataFrame(node_timings), hide_index=True, use_container_width=True)
            if tool_stats:
                import pandas as pd

                st.markdown("**Tool Statistics**")
                st.dataframe(pd.DataFrame(tool_stats), hide_index=True, use_container_width=True)
        else:
            st.info("Observability data not available.")


# ────────────────────────── Home / Empty State ────────────────────────────────


def _render_empty_state():
    hero_header(
        "🔬 DeepResearch Agent",
        "Autonomous deep research powered by LangGraph, MCP servers, and multi-model synthesis",
    )
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="white-panel animate-in">'
            '<h4 style="margin-top:0;">🧠 Intelligent Research</h4>'
            "<p>Planner-Executor-Critic loop decomposes queries, gathers multi-source evidence, "
            "and iterates until quality thresholds are met.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="white-panel animate-in">'
            '<h4 style="margin-top:0;">🔧 MCP Architecture</h4>'
            "<p>Three independent MCP servers (Web, arXiv, GitHub) over HTTP/SSE with "
            "JWT auth, SQLite caching, and per-tool circuit breakers.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="white-panel animate-in">'
            '<h4 style="margin-top:0;">📊 Full Observability</h4>'
            "<p>Every run is traced: node latency, token costs, tool success rates, "
            "and quality scores — all captured in SQLite.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="text-align:center;margin-top:2rem;color:#8B949E;">'
        "Click <b>➕ New Research</b> in the sidebar to start your first deep research query."
        "</div>",
        unsafe_allow_html=True,
    )


def _resume_research_stream(run_id: str):
    """Resume an approved research run on the main page, streaming SSE events in real-time."""
    run_state = _get_run_state(run_id)
    run_state["approved_to_resume"] = False
    run_state.setdefault("final_report_md", "")
    if not run_state.get("start_time"):
        run_state["start_time"] = time.perf_counter()

    status_ph = st.empty()
    status_text = "Resuming research pipeline..."
    status_ph.markdown(
        _render_status_bar(status_text, run_state["start_time"]),
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Full Agent Chain & Thought Process Trace")
    log_ph = st.empty()
    if run_state["thought_log"]:
        log_ph.markdown(
            '<div class="thought-log">' + "".join(run_state["thought_log"]) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 📄 Research Report")
    report_ph = st.empty()
    report_ph.info("⏳ Research in progress — report will appear here once complete...")

    st.markdown("### 📊 Live Execution Metrics")
    metrics_ph = st.empty()

    completed = False

    try:
        with (
            httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client,
            client.stream(
                "POST",
                f"{AGENT_API_URL}/research/approve",
                json={"thread_id": run_id, "approved": True},
            ) as response,
        ):
            for line in response.iter_lines():
                event = _parse_sse_line(line)
                if event is None:
                    continue

                etype = event.get("type", "")

                card = _format_thought_card(event)
                if card:
                    run_state["thought_log"].append(card)
                    log_ph.markdown(
                        '<div class="thought-log">' + "".join(run_state["thought_log"]) + "</div>",
                        unsafe_allow_html=True,
                    )

                # Accumulate tokens silently for fallback only
                if etype == "token":
                    content = event.get("content", "")
                    run_state["accumulated_report"] += content
                    run_state["token_count"] += max(1, int(len(content) * _AVG_TOKENS_PER_CHAR))

                if etype == "node_start":
                    run_state["node_count"] += 1
                    node = event.get("node", "Agent")
                    label = _NODE_LABELS.get(node, node)
                    desc = _NODE_DESCRIPTIONS.get(node, "Processing step...")
                    status_text = f"{label} — {desc}"
                    status_ph.markdown(
                        _render_status_bar(status_text, run_state["start_time"]),
                        unsafe_allow_html=True,
                    )
                    st.toast(f"🔄 {label} started", icon="🔄")
                elif etype == "tool_call":
                    run_state["tool_count"] += 1
                    tool = event.get("tool", "tool")
                    status_text = f"Invoking {tool}..."
                    status_ph.markdown(
                        _render_status_bar(status_text, run_state["start_time"]),
                        unsafe_allow_html=True,
                    )
                elif etype == "tool_result":
                    run_state["source_count"] += event.get("count", 0)
                    tool = event.get("tool", "tool")
                    count = event.get("count", 0)
                    status_text = f"Retrieved {count} results from {tool}"
                    status_ph.markdown(
                        _render_status_bar(status_text, run_state["start_time"]),
                        unsafe_allow_html=True,
                    )
                    st.toast(f"✅ {tool} returned results", icon="✅")
                elif etype == "complete":
                    run_state["status"] = "completed"
                    completed = True
                    elapsed_frozen = time.perf_counter() - (
                        run_state["start_time"] or time.perf_counter()
                    )
                    run_state["elapsed_frozen"] = elapsed_frozen
                    status_text = "Research Complete! PDF Report Generated."
                    status_ph.markdown(
                        _render_status_bar(
                            status_text,
                            run_state["start_time"],
                            is_completed=True,
                            elapsed_frozen=elapsed_frozen,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.toast("🏁 Research completed!", icon="🎉")

                est_cost = run_state["token_count"] * _OUTPUT_COST_PER_TOKEN
                with metrics_ph.container():
                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1:
                        metric_card("Status", run_state["status"].upper())
                    with m2:
                        metric_card("Nodes", str(run_state["node_count"]))
                    with m3:
                        metric_card("Tool Calls", str(run_state["tool_count"]))
                    with m4:
                        metric_card("Sources", str(run_state["source_count"]))
                    with m5:
                        metric_card("Est. Cost", f"${est_cost:.4f}")

    except httpx.ConnectError:
        st.error(f"❌ Could not connect to the Agent API at `{AGENT_API_URL}`.")
        return
    except Exception as e:
        st.error(f"❌ Stream error: {type(e).__name__}: {str(e)[:300]}")
        return

    # Check if report is ready on server even if complete SSE event was missed
    if not completed and _fetch_formatted_report(run_id):
        run_state["status"] = "completed"
        completed = True

    st.session_state.runs_cache = _fetch_runs()

    # After completion, switch to view mode and rerun automatically so user sees final report & PDF instantly
    if completed:
        st.session_state.mode = "view"
        st.session_state.active_run_id = run_id
        _render_final_report(run_id, report_ph, run_state)
        st.rerun()


# ────────────────────────── Main App ──────────────────────────────────────────

_init_session()

# Refresh runs list on every page load
if not st.session_state.runs_cache:
    st.session_state.runs_cache = _fetch_runs()
runs = st.session_state.runs_cache

# Sidebar
with st.sidebar:
    _render_history_sidebar(runs)

# Main content
mode = st.session_state.mode
active_run_id = st.session_state.active_run_id

if active_run_id and st.session_state.run_states.get(active_run_id, {}).get("approved_to_resume"):
    _resume_research_stream(active_run_id)
elif mode == "new":
    _render_new_research()
elif mode in ("view", "streaming") and active_run_id:
    _render_run_viewer(active_run_id, runs)
else:
    _render_empty_state()
