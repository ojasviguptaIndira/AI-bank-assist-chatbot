"""Entry point for the AI-powered Bank Loan Review Assistant EDA pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from bank_loan_eda.pipeline import EDAPipeline


def main() -> None:
    """Run the full exploratory data analysis workflow."""
    pipeline = EDAPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
