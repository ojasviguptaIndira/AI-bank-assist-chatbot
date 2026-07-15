"""Format orchestrator output as structured JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ResponseFormatter:
    """Convert orchestrator payloads to JSON strings."""

    indent: int = 2

    def format(self, payload: dict[str, object]) -> str:
        """Serialize structured payload to JSON."""
        return json.dumps(payload, indent=self.indent, default=str)


if __name__ == "__main__":
    print(ResponseFormatter().format({"type": "analytics", "data": {"value": 1}}))
