"""
app/streamlit_app.py

Main entry point for the DeepResearch Agent Streamlit frontend.

This file configures the page, injects the premium dark-mode theme,
and renders the sidebar navigation. The actual page content lives in
app/pages/research.py and app/pages/traces.py, which Streamlit
discovers automatically via its multi-page app convention.
"""

from __future__ import annotations

import streamlit as st

from app.components.auth import require_auth
from app.components.theme import hero_header, inject_theme

# ────────────────────────── Page Config ───────────────────────────────────────

st.set_page_config(
    page_title="DeepResearch Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
require_auth()

# ────────────────────────── Sidebar ───────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:1rem 0;">'
        '<span style="font-size:2.2rem;">🔬</span>'
        '<h3 style="margin:0.3rem 0 0;font-weight:700;">DeepResearch</h3>'
        '<span style="font-size:0.78rem;color:#8B949E;">'
        "Autonomous Multi-Source Research Agent"
        "</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        """
        **Architecture**
        - 🧠 LangGraph agent graph
        - 🔧 3 MCP servers (SSE)
        - 🔄 Planner-Executor-Critic loop
        - 🤖 Multi-model synthesis
        - 🔒 JWT auth + circuit breakers

        **Source Types**
        - 🌐 Web (Tavily)
        - 📄 Academic (arXiv)
        - ⭐ Code (GitHub)
        """,
        unsafe_allow_html=True,
    )

# ────────────────────────── Home Page ─────────────────────────────────────────

hero_header(
    title="🔬 DeepResearch Agent",
    subtitle="Autonomous deep research powered by LangGraph, MCP servers, and multi-model synthesis",
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="glass-panel animate-in">'
        '<h4 style="margin-top:0;">🧠 Intelligent Research</h4>'
        "<p>Planner-Executor-Critic loop decomposes queries, "
        "gathers multi-source evidence, and iterates until "
        "quality thresholds are met.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        '<div class="glass-panel animate-in">'
        '<h4 style="margin-top:0;">🔧 MCP Architecture</h4>'
        "<p>Three independent MCP servers over HTTP/SSE with "
        "JWT authentication, Redis caching, and per-tool "
        "circuit breakers for graceful degradation.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        '<div class="glass-panel animate-in">'
        '<h4 style="margin-top:0;">📊 Full Observability</h4>'
        "<p>Every run is traced: node-level latency, token costs, "
        "tool call success rates, and LLM-as-judge quality scores "
        "— all captured in SQLite.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center;color:#8B949E;font-size:0.85rem;'
    'margin-top:2rem;">'
    "Navigate to <b>Research</b> to start a query, or "
    "<b>Traces</b> to inspect past runs."
    "</div>",
    unsafe_allow_html=True,
)
