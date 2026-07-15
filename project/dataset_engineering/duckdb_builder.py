"""DuckDB artifact builder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class DuckDBBuilder:
    """Create DuckDB database and load cleaned datasets."""

    database_path: Path

    def build(
        self,
        loans: pd.DataFrame,
        reviews: pd.DataFrame,
        bank_statistics: pd.DataFrame,
        loan_statistics: pd.DataFrame,
    ) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.database_path)) as connection:
            self._create_table(connection, "loans", loans)
            self._create_table(connection, "reviews", reviews)
            self._create_table(connection, "bank_statistics", bank_statistics)
            self._create_table(connection, "loan_statistics", loan_statistics)
            self._create_indexes(connection)
        LOGGER.info("Built DuckDB database at %s", self.database_path)

    @staticmethod
    def _create_table(connection: duckdb.DuckDBPyConnection, name: str, frame: pd.DataFrame) -> None:
        connection.register("frame_view", frame)
        connection.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame_view")
        connection.unregister("frame_view")

    @staticmethod
    def _create_indexes(connection: duckdb.DuckDBPyConnection) -> None:
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_loans_purpose ON loans(purpose)",
            "CREATE INDEX IF NOT EXISTS idx_loans_grade ON loans(grade)",
            "CREATE INDEX IF NOT EXISTS idx_loans_status ON loans(loan_status)",
            "CREATE INDEX IF NOT EXISTS idx_reviews_bank ON reviews(bank)",
            "CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment)",
            "CREATE INDEX IF NOT EXISTS idx_bank_stats_bank ON bank_statistics(bank)",
            "CREATE INDEX IF NOT EXISTS idx_loan_stats_purpose ON loan_statistics(purpose)",
        ]
        for statement in statements:
            try:
                connection.execute(statement)
            except duckdb.Error:
                LOGGER.warning("Skipped index statement: %s", statement)
