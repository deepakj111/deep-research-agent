"""
app/components/theme.py

Centralized CSS theme for the DeepResearch Agent Streamlit UI.

Design: Dark-mode base with glassmorphic panels, gradient accents,
and micro-animation keyframes. Injected once via st.markdown() at
app start-up so all pages inherit the same visual identity.
"""

from __future__ import annotations

import streamlit as st

# ────────────────────────── Color Palette ─────────────────────────────────────

COLORS = {
    "bg_primary": "#0e1117",
    "bg_secondary": "#161b22",
    "bg_glass": "rgba(22, 27, 34, 0.75)",
    "border_glass": "rgba(255, 255, 255, 0.08)",
    "accent_teal": "#00C49F",
    "accent_blue": "#0088FE",
    "accent_orange": "#FF8042",
    "accent_purple": "#8B5CF6",
    "accent_pink": "#EC4899",
    "text_primary": "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_muted": "#484F58",
    "success": "#3FB950",
    "warning": "#D29922",
    "error": "#F85149",
}

SOURCE_COLORS = {
    "web": COLORS["accent_teal"],
    "arxiv": COLORS["accent_blue"],
    "github": COLORS["accent_orange"],
}


# ────────────────────────── Global CSS ────────────────────────────────────────

_GLOBAL_CSS = """
<style>
/* ─── Import Google Fonts ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root Variables ─── */
:root {
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-glass: rgba(22, 27, 34, 0.75);
    --border-glass: rgba(255, 255, 255, 0.08);
    --accent-teal: #00C49F;
    --accent-blue: #0088FE;
    --accent-orange: #FF8042;
    --accent-purple: #8B5CF6;
    --text-primary: #E6EDF3;
    --text-secondary: #8B949E;
    --success: #3FB950;
    --warning: #D29922;
    --error: #F85149;
}

/* ─── Global Typography ─── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, [data-testid="stCode"] {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* ─── Glassmorphic Panel ─── */
.glass-panel {
    background: var(--bg-glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

/* ─── Thought Log Container ─── */
/* ─── Thought Log Container (Crisp White High-Contrast) ─── */
.thought-log {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    max-height: 650px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #0969da #f6f8fa;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}
.thought-log::-webkit-scrollbar { width: 8px; }
.thought-log::-webkit-scrollbar-track { background: #f6f8fa; border-radius: 4px; }
.thought-log::-webkit-scrollbar-thumb {
    background: #0969da;
    border-radius: 4px;
}

/* ─── High-Contrast Thought Card ─── */
.thought-card {
    background: #f6f8fa;
    border: 1px solid #d8dee4;
    border-left: 5px solid #0969da;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.9rem;
    animation: slideIn 0.3s ease-out;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    font-size: 0.88rem;
}
.thought-card.node_start {
    border-left-color: #8250df;
    background: linear-gradient(90deg, rgba(130, 80, 223, 0.08), #f6f8fa);
}
.thought-card.node_end {
    border-left-color: #1a7f37;
    background: linear-gradient(90deg, rgba(26, 127, 55, 0.08), #f6f8fa);
}
.thought-card.llm_start {
    border-left-color: #0969da;
    background: linear-gradient(90deg, rgba(9, 105, 218, 0.06), #f6f8fa);
}
.thought-card.llm_end {
    border-left-color: #1a7f37;
    background: linear-gradient(90deg, rgba(26, 127, 55, 0.06), #f6f8fa);
}
.thought-card.tool_call {
    border-left-color: #cf222e;
    background: linear-gradient(90deg, rgba(207, 34, 46, 0.06), #f6f8fa);
}
.thought-card.tool_result {
    border-left-color: #1a7f37;
    background: linear-gradient(90deg, rgba(26, 127, 55, 0.06), #f6f8fa);
}
.thought-card.complete {
    border-left-color: #0969da;
    background: linear-gradient(90deg, rgba(9, 105, 218, 0.08), #f6f8fa);
}
.thought-card-header {
    display: flex;
    align-items: center;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
    color: #1f2328;
}
.thought-card-icon {
    margin-right: 0.6rem;
    font-size: 1.15rem;
}
.thought-card-body {
    color: #24292f;
    font-size: 0.85rem;
    word-break: break-word;
}
.thought-card-body pre {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    max-height: 400px;
    overflow: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #1f2328;
    white-space: pre-wrap;
    scrollbar-width: thin;
    margin-top: 0.5rem;
}
.thought-card-body details {
    margin-top: 0.5rem;
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
}
.thought-card-body summary {
    font-weight: 700;
    cursor: pointer;
    color: #0969da;
    font-size: 0.85rem;
    user-select: none;
}

/* ─── JSON Pretty Viewer & Syntax Highlighting ─── */
.json-view-container {
    background: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.5;
    max-height: 420px;
    overflow: auto;
    scrollbar-width: thin;
    scrollbar-color: #30363d #0d1117;
    margin-top: 0.5rem;
    box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.4);
}
.json-key { color: #79c0ff; font-weight: 600; }
.json-string { color: #a5d6ff; word-break: break-word; }
.json-number { color: #d2a8ff; font-weight: 500; }
.json-boolean { color: #ff7b72; font-weight: 700; }
.json-null { color: #8b949e; font-style: italic; }
.json-bracket { color: #8b949e; font-weight: 600; }

/* ─── Structured Key Summary Badges ─── */
.summary-pills-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
}
.summary-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #24292f;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.summary-pill .pill-key {
    color: #57606a;
    font-weight: 500;
}
.summary-pill .pill-val {
    color: #0969da;
    font-weight: 700;
}

/* ─── Role Prompt Cards ─── */
.prompt-role-card {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.prompt-role-card.system { border-left: 4px solid #8250df; }
.prompt-role-card.user { border-left: 4px solid #0969da; }
.role-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.role-badge.system { background: rgba(130, 80, 223, 0.12); color: #8250df; }
.role-badge.user { background: rgba(9, 105, 218, 0.12); color: #0969da; }

/* ─── Tool Result Card Items ─── */
.tool-result-item {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 0.75rem 0.9rem;
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.tool-result-title {
    font-weight: 700;
    font-size: 0.88rem;
    color: #0969da;
    text-decoration: none;
}
.tool-result-title:hover { text-decoration: underline; }
.tool-result-url {
    font-size: 0.73rem;
    color: #57606a;
    word-break: break-all;
    margin-bottom: 0.3rem;
}
.tool-result-snippet {
    font-size: 0.8rem;
    color: #24292f;
    line-height: 1.4;
    background: #f6f8fa;
    padding: 0.4rem 0.6rem;
    border-radius: 4px;
    border: 1px solid #eaeef2;
}

/* ─── Live Activity Indicator ─── */
.live-activity {
    display: flex;
    align-items: center;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    background: rgba(0, 196, 159, 0.1);
    border: 1px solid rgba(0, 196, 159, 0.2);
    color: var(--accent-teal);
    font-weight: 500;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* ─── Metric Card ─── */
.metric-card {
    background: linear-gradient(135deg, rgba(0,196,159,0.08), rgba(0,136,254,0.08));
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-1px); }
.metric-card .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ─── Source Badge ─── */
.source-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0.15rem;
}
.source-badge.web { background: rgba(0,196,159,0.15); color: var(--accent-teal); }
.source-badge.arxiv { background: rgba(0,136,254,0.15); color: var(--accent-blue); }
.source-badge.github { background: rgba(255,128,66,0.15); color: var(--accent-orange); }

/* ─── Status Indicator ─── */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.running { background: var(--accent-teal); animation: pulse 1.5s ease-in-out infinite; }
.status-dot.completed { background: var(--success); }
.status-dot.failed { background: var(--error); }
.status-dot.rejected { background: var(--warning); }

/* ─── Hero Header ─── */
.hero-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
}
.hero-header h1 {
    background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
}
.hero-header .subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

/* ─── Animations ─── */
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.15); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeIn 0.35s ease-out;
}

/* ─── Table Improvements ─── */
.stDataFrame table {
    border-collapse: separate;
    border-spacing: 0;
}
.stDataFrame th {
    background: var(--bg-secondary) !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
}

/* ─── History Run Card ─── */
.run-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
}
.run-card:hover {
    border-color: rgba(0, 196, 159, 0.4);
    background: rgba(0, 196, 159, 0.05);
    transform: translateX(2px);
}
.run-card.active {
    border-color: var(--accent-teal);
    background: rgba(0, 196, 159, 0.08);
}
.run-card-query {
    font-size: 0.82rem;
    color: var(--text-primary);
    font-weight: 500;
    margin-bottom: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.run-card-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem;
    color: var(--text-secondary);
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.12rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.status-pill.running {
    background: rgba(0, 196, 159, 0.15);
    color: var(--accent-teal);
}
.status-pill.completed {
    background: rgba(63, 185, 80, 0.15);
    color: var(--success);
}
.status-pill.failed {
    background: rgba(248, 81, 73, 0.15);
    color: var(--error);
}
.status-pill.rejected {
    background: rgba(210, 153, 34, 0.15);
    color: var(--warning);
}

/* ─── Report Viewer ─── */
.report-viewer {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 2rem;
    line-height: 1.75;
    font-size: 0.95rem;
}
.report-viewer h1, .report-viewer h2, .report-viewer h3 {
    color: var(--accent-teal);
    margin-top: 1.5rem;
}

/* ─── Slide-in animation ─── */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


def inject_theme() -> None:
    """Inject the global CSS theme. Call once at the top of the main app."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def hero_header(title: str = "DeepResearch Agent", subtitle: str = "") -> None:
    """Render the gradient hero header."""
    sub_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="hero-header"><h1>{title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str) -> None:
    """Render a single KPI metric card."""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def source_badge(source_type: str) -> str:
    """Return an HTML badge for a source type."""
    return f'<span class="source-badge {source_type}">{source_type}</span>'


def status_dot(status: str) -> str:
    """Return an HTML status indicator dot."""
    return f'<span class="status-dot {status}"></span>'
