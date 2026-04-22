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
.thought-log {
    background: var(--bg-secondary);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 1rem;
    max-height: 520px;
    overflow-y: auto;
    font-size: 0.85rem;
    line-height: 1.7;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-teal) transparent;
}
.thought-log::-webkit-scrollbar { width: 6px; }
.thought-log::-webkit-scrollbar-thumb {
    background: var(--accent-teal);
    border-radius: 3px;
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
