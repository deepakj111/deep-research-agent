"""
app/components/auth.py

Provides username and password authentication for the Streamlit frontend.
Designed to prevent unauthorized access and API cost bleed when the portfolio
agent is deployed publicly on Streamlit Community Cloud or custom hosting.
"""

from __future__ import annotations

import os

import streamlit as st

from config.settings import settings


def _get_expected_credentials() -> tuple[str, str]:
    """
    Retrieve expected username and password from:
    1. Streamlit secrets (st.secrets["DEMO_USERNAME"], st.secrets["DEMO_PASSWORD"])
    2. Environment variables (DEMO_USERNAME, DEMO_PASSWORD / FRONTEND_PASSWORD)
    3. Settings / default fallback credentials (admin / demo123)
    """
    username = ""
    password = ""

    # 1. Check Streamlit Community Cloud secrets
    try:
        if hasattr(st, "secrets") and st.secrets:
            username = st.secrets.get("DEMO_USERNAME") or st.secrets.get("username", "")
            password = st.secrets.get("DEMO_PASSWORD") or st.secrets.get("password", "")
    except Exception:
        pass

    # 2. Check environment variables & settings
    if not username:
        username = os.environ.get("DEMO_USERNAME") or getattr(settings, "demo_username", "admin")
    if not password:
        password = (
            os.environ.get("DEMO_PASSWORD")
            or os.environ.get("FRONTEND_PASSWORD")
            or getattr(settings, "demo_password", "")
            or getattr(settings, "frontend_password", "demo123")
        )

    return username.strip(), password.strip()


def require_auth() -> None:
    """
    Enforce frontend username and password protection.
    If already authenticated, renders a logout option in the sidebar.
    Stops the Streamlit UI execution until authenticated.
    """
    expected_username, expected_password = _get_expected_credentials()

    # Bypass authentication if no password is configured anywhere
    if not expected_password:
        return

    # If already authenticated, render logout control in sidebar and return
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.divider()
            user_label = st.session_state.get("auth_user", expected_username)
            st.caption(f"👤 Logged in as **{user_label}**")
            if st.button("🚪 Log Out", use_container_width=True, key="logout_btn"):
                st.session_state["authenticated"] = False
                st.session_state["auth_user"] = None
                st.rerun()
        return

    # Render Glassmorphic Auth Gateway
    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div class="glass-panel" style="text-align: center; padding: 2.5rem 2rem 1.5rem;">'
            '<span style="font-size:3rem;">🔒</span>'
            '<h2 style="margin-top:0.5rem; margin-bottom:0.5rem;">Portfolio Demo Access</h2>'
            '<p style="color:var(--text-secondary); font-size:0.9rem; margin-bottom: 1.5rem;">'
            "This research agent demo is protected. Please sign in with your credentials to access the application."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input(
                "Username", key="login_username", placeholder="Enter username"
            )
            password_input = st.text_input(
                "Password", type="password", key="login_password", placeholder="Enter password"
            )
            submit = st.form_submit_button(
                "🚀 Sign In to Demo", type="primary", use_container_width=True
            )

        if submit:
            user_valid = not expected_username or (username_input.strip() == expected_username)
            pass_valid = password_input.strip() == expected_password

            if user_valid and pass_valid:
                st.session_state["authenticated"] = True
                st.session_state["auth_user"] = username_input.strip() or "User"
                st.toast("✅ Signed in successfully!", icon="🎉")
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Please try again.")

    # Halt execution of the rest of the application until authenticated
    st.stop()
