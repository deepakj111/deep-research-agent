"""
app/components/source_graph.py

Interactive force-directed source graph using pyvis + networkx.

Renders a visual network of sources found during research, color-coded
by type (web=teal, arXiv=blue, GitHub=orange), with node size proportional
to trust score. Sources that share the same sub-question finding are
connected by edges to show topical relationships.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from app.components.theme import COLORS, SOURCE_COLORS

# ────────────────────────── Config ────────────────────────────────────────────

_NODE_MIN_SIZE = 12
_NODE_MAX_SIZE = 35


def _truncate(text: str, max_len: int = 45) -> str:
    return text[:max_len] + "…" if len(text) > max_len else text


# ────────────────────────── Public API ────────────────────────────────────────


def render_source_graph(
    findings: list,
    height: str = "420px",
) -> None:
    """
    Build and render an interactive source relationship graph.

    Args:
        findings: List of ResearchFindings from the agent state.
        height:   CSS height for the graph container.
    """
    if not findings:
        st.info("No sources to visualize — run a research query first.")
        return

    G = nx.Graph()

    # Collect all sources per sub-question for edge creation
    for finding in findings:
        subq_nodes: list[str] = []

        for w in getattr(finding, "web_results", []):
            node_id = w.url
            if not G.has_node(node_id):
                G.add_node(
                    node_id,
                    label=_truncate(w.title),
                    color=SOURCE_COLORS["web"],
                    size=_NODE_MIN_SIZE + w.relevance_score * (_NODE_MAX_SIZE - _NODE_MIN_SIZE),
                    title=f"🌐 {w.title}\n{w.url}\nRelevance: {w.relevance_score:.2f}",
                    source_type="web",
                )
            subq_nodes.append(node_id)

        for p in getattr(finding, "papers", []):
            node_id = p.url
            if not G.has_node(node_id):
                G.add_node(
                    node_id,
                    label=_truncate(p.title),
                    color=SOURCE_COLORS["arxiv"],
                    size=_NODE_MIN_SIZE + p.trust_score * (_NODE_MAX_SIZE - _NODE_MIN_SIZE),
                    title=f"📄 {p.title}\n{p.url}\nCitations: {p.citation_count}\nTrust: {p.trust_score:.2f}",
                    source_type="arxiv",
                )
            subq_nodes.append(node_id)

        for r in getattr(finding, "repos", []):
            node_id = r.url
            if not G.has_node(node_id):
                G.add_node(
                    node_id,
                    label=_truncate(r.name),
                    color=SOURCE_COLORS["github"],
                    size=_NODE_MIN_SIZE + r.trust_score * (_NODE_MAX_SIZE - _NODE_MIN_SIZE),
                    title=f"⭐ {r.name} ({r.stars}★)\n{r.url}\n{r.description[:100]}",
                    source_type="github",
                )
            subq_nodes.append(node_id)

        # Connect sources within the same sub-question finding
        for i, n1 in enumerate(subq_nodes):
            for n2 in subq_nodes[i + 1 :]:
                if not G.has_edge(n1, n2):
                    G.add_edge(n1, n2, color="rgba(255,255,255,0.06)", width=1)

    if G.number_of_nodes() == 0:
        st.info("No sources found in the research findings.")
        return

    # Build pyvis network
    net = Network(
        height=height,
        width="100%",
        bgcolor=COLORS["bg_primary"],
        font_color=COLORS["text_primary"],
        directed=False,
    )
    net.from_nx(G)

    # Physics settings for a clean, stabilized layout
    net.set_options(
        """{
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 120,
                "springConstant": 0.04
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 100}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 150,
            "zoomView": true,
            "dragView": true
        },
        "nodes": {
            "font": {"size": 11, "face": "Inter, sans-serif"},
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "shape": "dot"
        },
        "edges": {
            "smooth": {"type": "continuous"}
        }
    }"""
    )

    # Write to a temp file and embed in Streamlit
    # Use content hash for caching — avoids re-rendering unchanged graphs
    graph_hash = hashlib.md5(str(sorted(G.nodes)).encode()).hexdigest()[:8]
    tmp_path = Path(tempfile.gettempdir()) / f"source_graph_{graph_hash}.html"
    net.save_graph(str(tmp_path))

    html_content = tmp_path.read_text()
    components.html(html_content, height=int(height.replace("px", "")) + 20, scrolling=False)

    # Legend
    st.markdown(
        '<div style="text-align:center;font-size:0.78rem;color:#8B949E;margin-top:-8px;">'
        '<span style="color:#00C49F">●</span> Web &nbsp;&nbsp;'
        '<span style="color:#0088FE">●</span> arXiv &nbsp;&nbsp;'
        '<span style="color:#FF8042">●</span> GitHub &nbsp;&nbsp;'
        "| Node size = trust/relevance score"
        "</div>",
        unsafe_allow_html=True,
    )
