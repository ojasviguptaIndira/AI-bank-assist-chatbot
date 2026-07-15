"""Main chat UI rendering."""

from __future__ import annotations

import json

import streamlit as st


def render_chat_history(history: list[dict[str, object]]) -> None:
    """Render prior chat messages and explainability blocks."""
    for entry in history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            _render_explainability(entry)


def render_response(entry: dict[str, object]) -> None:
    """Render a newly generated response."""
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        _render_explainability(entry)
        with st.expander("Structured JSON"):
            st.code(json.dumps(entry["structured"], indent=2, default=str), language="json")


def _render_explainability(entry: dict[str, object]) -> None:
    explain = entry.get("explainability", {})
    st.markdown("**Answer Context**")
    st.write(f"Source: {explain.get('source', 'Unknown')}")
    st.write(f"Records Analysed: {explain.get('records_analyzed', 'N/A')}")
    st.write(f"Confidence: {explain.get('confidence', 'N/A')}")
    st.write(f"Generated Using: {explain.get('generated_using', 'N/A')}")
    st.write(f"Execution Time: {explain.get('total_time_sec', 'N/A')} sec")
    if explain.get("analytics_time_sec") is not None:
        st.write(f"Analytics Time: {explain['analytics_time_sec']} sec")
    if explain.get("sentiment_time_sec") is not None:
        st.write(f"Sentiment Time: {explain['sentiment_time_sec']} sec")
    if explain.get("sql_executed"):
        st.code(explain["sql_executed"], language="sql")
