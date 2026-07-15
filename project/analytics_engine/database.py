"""DuckDB database builder and connection helpers."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class AnalyticsDatabase:
    """Manage the analytics DuckDB database lifecycle."""

    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self) -> None:
        self.artifacts_dir = self.root / "artifacts"
        self.database_dir = self.root / "database"
        self.database_path = self.database_dir / "loan.duckdb"
        self._runtime_database_path: Path | None = None

    def build(self) -> Path:
        """Create the DuckDB database from engineered CSV artifacts."""
        self.database_dir.mkdir(parents=True, exist_ok=True)
        loan_frame = pd.read_csv(self.artifacts_dir / "loan_cleaned.csv", low_memory=False)
        review_frame = pd.read_csv(self.artifacts_dir / "review_cleaned.csv", low_memory=False)
        loan_statistics = pd.read_csv(self.artifacts_dir / "loan_statistics.csv", low_memory=False)
        bank_statistics = pd.read_csv(self.artifacts_dir / "bank_statistics.csv", low_memory=False)

        with duckdb.connect(str(self.database_path)) as connection:
            self._load_table(connection, "loans", loan_frame)
            self._load_table(connection, "reviews", review_frame)
            self._load_table(connection, "loan_statistics", loan_statistics)
            self._load_table(connection, "bank_statistics", bank_statistics)
            self._create_indexes(connection)

        LOGGER.info("Analytics database ready at %s", self.database_path)
        return self.database_path

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Open a connection to the analytics database."""
        if not self.database_path.exists():
            self.build()
        if self._runtime_database_path is None or not self._runtime_database_path.exists():
            runtime_dir = Path(tempfile.gettempdir()) / "bank_loan_review_assistant"
            runtime_dir.mkdir(parents=True, exist_ok=True)
            self._runtime_database_path = runtime_dir / f"loan_{self.database_path.stat().st_mtime_ns}.duckdb"
            if not self._runtime_database_path.exists():
                shutil.copy2(self.database_path, self._runtime_database_path)
        return duckdb.connect(str(self._runtime_database_path), read_only=True)

    @staticmethod
    def _load_table(
        connection: duckdb.DuckDBPyConnection,
        table_name: str,
        frame: pd.DataFrame,
    ) -> None:
        connection.register("frame_view", frame)
        connection.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM frame_view")
        connection.unregister("frame_view")

    @staticmethod
    def _create_indexes(connection: duckdb.DuckDBPyConnection) -> None:
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_loans_state ON loans(addr_state)",
            "CREATE INDEX IF NOT EXISTS idx_loans_purpose ON loans(purpose)",
            "CREATE INDEX IF NOT EXISTS idx_loans_grade ON loans(grade)",
            "CREATE INDEX IF NOT EXISTS idx_loans_home_ownership ON loans(home_ownership)",
            "CREATE INDEX IF NOT EXISTS idx_loans_verification_status ON loans(verification_status)",
            "CREATE INDEX IF NOT EXISTS idx_loans_loan_status ON loans(loan_status)",
            "CREATE INDEX IF NOT EXISTS idx_reviews_bank ON reviews(bank)",
            "CREATE INDEX IF NOT EXISTS idx_bank_statistics_bank ON bank_statistics(bank)",
        ]
        for statement in statements:
            try:
                connection.execute(statement)
            except duckdb.Error:
                LOGGER.warning("Unable to create index with statement: %s", statement)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    AnalyticsDatabase().build()
