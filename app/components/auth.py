"""
app/components/auth.py

Provides password-based authentication for the Stremlit frontend.
Designed to prevent unauthorized access and API cost bleed when the portfolio
agent is deployed publicly.
"""

from __future__ import annotations

import streamlit as st

from config.settings import settings


def require_auth() -> None:
    """
    Enforce frontend password protection.
    If the environment variable `FRONTEND_PASSWORD` is not set, authentication is bypassed.
    If it is set, the user is prompted to enter it. Stops the Streamlit UI execution
    until authenticated.
    """
    # 1. Bypass authentication if no password is configured (e.g. local dev)
    if not settings.frontend_password:
        return

    # 2. Check session state
    if st.session_state.get("authenticated", False):
        return

    # 3. Render professional Glassmorphic Auth Gateway
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div class="glass-panel" style="text-align: center; padding: 3rem;">'
            '<span style="font-size:3rem;">🔒</span>'
            '<h2 style="margin-top:0.5rem; margin-bottom:1.5rem;">Access Restricted</h2>'
            '<p style="color:var(--text-secondary); margin-bottom: 2rem;">'
            "This deployment requires an authentication gateway password. "
            "Please verify your credentials to access the DeepResearch Agent."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Standard Streamlit input but styled nicely via themes
        password = st.text_input("Gateway Password", type="password", key="auth_password")

        if st.button("Unlock 🚀", type="primary", use_container_width=True):
            if password == settings.frontend_password:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect gateway password. Please try again.")

    # 4. Halt execution of the rest of the application
    st.stop()
