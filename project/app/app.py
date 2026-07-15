"""Streamlit application entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "app"

from app.chat import ChatApplication
from app.streamlit.chat_ui import render_chat_history, render_response
from app.streamlit.sidebar import render_sidebar


def _get_app() -> ChatApplication:
    if "chat_app" not in st.session_state:
        st.session_state.chat_app = ChatApplication()
    return st.session_state.chat_app


def main() -> None:
    """Run the Streamlit chatbot UI."""
    st.set_page_config(
        page_title="AI Bank Loan Review Assistant",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background-color: #0f172a; color: #e2e8f0; }
        .stChatMessage, .stTextInput, .stButton button { border-radius: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    app_instance = _get_app()
    st.title("AI Powered Bank Loan Review Assistant")
    st.caption("Deterministic analytics and sentiment engines with Gemini as the presentation layer")

    example_query = render_sidebar(app_instance)
    render_chat_history(app_instance.history.list_entries())

    prompt = st.chat_input("Ask about loan analytics, bank reviews, or comparisons")
    active_query = prompt or example_query
    if active_query:
        with st.chat_message("user"):
            st.write(active_query)
        result = app_instance.process_query(active_query)
        render_response(result)


if __name__ == "__main__":
    main()
