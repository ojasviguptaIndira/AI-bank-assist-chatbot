"""Project configuration and path constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project locations."""

    root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = root / "data"
    output_dir: Path = root / "output"
    excel_dir: Path = output_dir / "excel"
    reports_dir: Path = output_dir / "reports"
    plots_dir: Path = output_dir / "plots"
    loan_plot_dir: Path = plots_dir / "loan"
    review_plot_dir: Path = plots_dir / "review"
    loan_dataset: Path = data_dir / "loan_dataset.csv"
    review_dataset: Path = data_dir / "review_dataset.csv"


PLOTS_DPI = 140
TOP_WORD_COUNT = 50
TOP_NGRAM_COUNT = 30
