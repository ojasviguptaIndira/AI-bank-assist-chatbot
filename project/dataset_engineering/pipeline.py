"""Dataset engineering pipeline orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bank_loan_eda.bootstrap import DatasetBootstrapper
from bank_loan_eda.config import ProjectPaths
from bank_loan_eda.logging_utils import configure_logging
from dataset_engineering.duckdb_builder import DuckDBBuilder
from dataset_engineering.loan_cleaner import LoanCleaner
from dataset_engineering.review_cleaner import ReviewCleaner
from dataset_engineering.statistics import StatisticsBuilder


LOGGER = logging.getLogger(__name__)


@dataclass
class DatasetEngineeringPipeline:
    """Build cleaned datasets, aggregated statistics, and DuckDB artifacts."""

    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self) -> None:
        configure_logging()
        self.paths = ProjectPaths(root=self.root)
        self.artifacts_dir = self.root / "artifacts"
        self.duckdb_dir = self.root / "duckdb"
        self.column_mapping_path = self.artifacts_dir / "column_mapping.json"
        self.loan_cleaned_path = self.artifacts_dir / "loan_cleaned.csv"
        self.review_cleaned_path = self.artifacts_dir / "review_cleaned.csv"
        self.loan_statistics_path = self.artifacts_dir / "loan_statistics.csv"
        self.bank_statistics_path = self.artifacts_dir / "bank_statistics.csv"
        self.sentiment_training_path = self.artifacts_dir / "sentiment_training.csv"
        self.database_path = self.duckdb_dir / "loan.duckdb"

    def run(self) -> None:
        DatasetBootstrapper(self.paths).prepare()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.duckdb_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Loading source datasets")
        loan_frame = pd.read_csv(self.paths.loan_dataset, low_memory=False)
        review_frame = pd.read_csv(self.paths.review_dataset, low_memory=False)

        LOGGER.info("Cleaning loan dataset")
        loan_cleaner = LoanCleaner(loan_frame)
        loan_cleaned = loan_cleaner.clean()

        LOGGER.info("Cleaning review dataset")
        review_cleaner = ReviewCleaner(review_frame)
        review_cleaned = review_cleaner.clean()

        LOGGER.info("Building statistics datasets")
        statistics = StatisticsBuilder(loan_cleaned, review_cleaned)
        loan_statistics = statistics.build_loan_statistics()
        bank_statistics = statistics.build_bank_statistics()
        sentiment_training = self._sentiment_training(review_cleaned)

        LOGGER.info("Writing CSV artifacts")
        loan_cleaned.to_csv(self.loan_cleaned_path, index=False)
        review_cleaned.to_csv(self.review_cleaned_path, index=False)
        loan_statistics.to_csv(self.loan_statistics_path, index=False)
        bank_statistics.to_csv(self.bank_statistics_path, index=False)
        sentiment_training.to_csv(self.sentiment_training_path, index=False)
        self._write_column_mapping(loan_cleaner.mapping(), review_cleaner.mapping())

        LOGGER.info("Building DuckDB database")
        DuckDBBuilder(self.database_path).build(
            loans=loan_cleaned,
            reviews=review_cleaned,
            bank_statistics=bank_statistics,
            loan_statistics=loan_statistics,
        )
        LOGGER.info("Dataset engineering pipeline completed successfully")

    def _sentiment_training(self, review_frame: pd.DataFrame) -> pd.DataFrame:
        required = ["review", "clean_review", "bank", "rating", "sentiment"]
        available = [column for column in required if column in review_frame.columns]
        return review_frame[available].copy()

    def _write_column_mapping(
        self,
        loan_mapping: dict[str, object],
        review_mapping: dict[str, object],
    ) -> None:
        payload = {
            "loan_dataset": loan_mapping,
            "review_dataset": review_mapping,
        }
        self.column_mapping_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
