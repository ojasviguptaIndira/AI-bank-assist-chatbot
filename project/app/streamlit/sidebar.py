"""Sidebar rendering."""

from __future__ import annotations

import streamlit as st

from app.streamlit.widgets import Widgets


def render_sidebar(app_instance) -> str | None:
    """Render sidebar information and return a clicked example query if any."""
    stats = app_instance.config.load_project_stats()
    widgets = Widgets()
    with st.sidebar:
        st.title("AI Bank Loan Review Assistant")
        st.caption("Explainable analytics + sentiment + Gemini presentation layer")
        st.markdown("### Project Information")
        st.write(f"Loan Records: {stats['loan_records']:,}")
        st.write(f"Reviews: {stats['reviews']:,}")
        st.write(f"Banks: {stats['banks']}")
        st.write(f"Model Accuracy: {stats['model_accuracy']}")
        st.write(f"Selected Model: {stats['selected_model']}")
        widgets.render_status("DuckDB Status", stats["duckdb_status"])
        widgets.render_status("Gemini Status", stats["gemini_status"])
        example = widgets.render_example_questions(app_instance.config.sample_questions)
        st.markdown("### Chat Controls")
        if st.button("Clear Chat", use_container_width=True):
            app_instance.clear_history()
            st.rerun()
        st.download_button(
            "Export Chat",
            data=app_instance.export_history(),
            file_name="chat_history.json",
            mime="application/json",
            use_container_width=True,
        )
        return example
