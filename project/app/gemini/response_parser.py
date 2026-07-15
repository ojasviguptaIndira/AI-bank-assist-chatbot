"""Parse Gemini responses safely."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResponseParser:
    """Normalize raw generated text."""

    def parse(self, text: str) -> str:
        """Strip simple code fences and whitespace."""
        value = (text or "").strip()
        if value.startswith("```"):
            value = value.strip("`").strip()
        return value
