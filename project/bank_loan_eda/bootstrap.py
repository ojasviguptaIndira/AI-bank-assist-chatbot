"""Bootstrap source datasets into the required project layout."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from bank_loan_eda.config import ProjectPaths


LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetBootstrapper:
    """Ensure the expected folder structure and dataset paths exist."""

    paths: ProjectPaths

    def prepare(self) -> None:
        """Create directories and normalize input dataset filenames."""
        for directory in (
            self.paths.data_dir,
            self.paths.excel_dir,
            self.paths.reports_dir,
            self.paths.loan_plot_dir,
            self.paths.review_plot_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self._copy_if_missing(
            preferred=self.paths.loan_dataset,
            fallbacks=("loan_dataset.csv", "loan.csv"),
        )
        self._copy_if_missing(
            preferred=self.paths.review_dataset,
            fallbacks=("review_dataset.csv", "bank_reviews3 dataset.csv"),
        )

    def _copy_if_missing(self, preferred: Path, fallbacks: tuple[str, ...]) -> None:
        if preferred.exists():
            return

        for name in fallbacks:
            candidate = self.paths.root / name
            if candidate.exists():
                shutil.copy2(candidate, preferred)
                LOGGER.info("Copied %s to %s", candidate.name, preferred)
                return

        joined = ", ".join(fallbacks)
        raise FileNotFoundError(f"Could not locate dataset source. Tried: {joined}")
