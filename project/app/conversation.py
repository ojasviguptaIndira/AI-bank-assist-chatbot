"""Conversation context and follow-up resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ConversationManager:
    """Resolve follow-up questions using recent chat context."""

    def resolve_query(self, query: str, history: list[dict[str, object]]) -> str:
        """Expand follow-up queries using the last question when needed."""
        if not history:
            return query
        normalized = query.strip().lower()
        follow_up_patterns = (
            r"^how about\b",
            r"^what about\b",
            r"^and\b",
            r"^compare with\b",
            r"^how is\b",
        )
        if any(re.search(pattern, normalized) for pattern in follow_up_patterns):
            previous_query = history[-1].get("resolved_query") or history[-1].get("question", "")
            return f"{previous_query} {query}".strip()
        return query

    @staticmethod
    def build_context(history: list[dict[str, object]]) -> str:
        """Create a compact history context for prompting."""
        lines = []
        for entry in history[-10:]:
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            lines.append(f"User: {question}\nAssistant: {answer}")
        return "\n".join(lines)
