"""Chat history management."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ChatHistory:
    """Store recent chat exchanges and support export."""

    limit: int = 10
    _entries: deque[dict[str, object]] = field(default_factory=deque)

    def add(self, entry: dict[str, object]) -> None:
        """Add a history entry and enforce the size limit."""
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._entries.append(entry)
        while len(self._entries) > self.limit:
            self._entries.popleft()

    def list_entries(self) -> list[dict[str, object]]:
        """Return history entries as a list."""
        return list(self._entries)

    def last_entry(self) -> dict[str, object] | None:
        """Return the most recent entry if present."""
        return self._entries[-1] if self._entries else None

    def clear(self) -> None:
        """Clear history."""
        self._entries.clear()

    def export_json(self) -> str:
        """Export history as a JSON string."""
        return json.dumps(self.list_entries(), indent=2, default=str)
