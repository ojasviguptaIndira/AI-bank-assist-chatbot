"""Prompt construction for Gemini response generation."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class PromptBuilder:
    """Build prompts that constrain Gemini to presentation-only behavior."""

    def build(
        self,
        user_query: str,
        structured_payload: dict[str, object],
        conversation_context: str = "",
    ) -> str:
        """Create a safe prompt using only user query and structured JSON."""
        payload = json.dumps(structured_payload, indent=2, default=str)
        return f"""
You are a Natural Language Response Generator for a bank loan review assistant.

Rules:
- Use only the supplied structured data.
- Do not generate SQL.
- Do not perform analytics.
- Do not perform sentiment analysis.
- Do not fabricate facts or missing values.
- If the data is insufficient, say so explicitly.
- Answer naturally and clearly.
- Mention confidence when it is available.
- Briefly reference the source and execution context when useful.

Conversation Context:
{conversation_context or "No prior context."}

User Query:
{user_query}

Structured JSON:
{payload}

Produce a concise, human-friendly response.
""".strip()
