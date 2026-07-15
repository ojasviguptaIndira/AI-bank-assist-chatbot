"""Report and Excel writing helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReportWriter:
    """Persist text and tabular outputs."""

    @staticmethod
    def write_text(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
        lines = [title, "=" * len(title), ""]
        for heading, body in sections:
            lines.extend([heading, "-" * len(heading), body.strip(), ""])
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                safe_name = sheet_name[:31]
                frame.to_excel(writer, sheet_name=safe_name, index=False)


def table_to_text(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    """Format a dataframe for text reports."""
    selection = frame.head(max_rows) if max_rows else frame
    return selection.to_string(index=False) if not selection.empty else "No data available."
