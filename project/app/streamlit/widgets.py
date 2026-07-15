"""Reusable Streamlit widgets."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass
class Widgets:
    """Shared widget helpers."""

    def render_example_questions(self, examples: list[str]) -> str | None:
        """Render example-question buttons."""
        st.markdown("### Example Questions")
        for example in examples:
            if st.button(example, use_container_width=True):
                return example
        return None

    @staticmethod
    def render_status(label: str, status: bool) -> None:
        """Render a boolean status indicator."""
        st.write(f"{label}: {'Available' if status else 'Unavailable'}")
