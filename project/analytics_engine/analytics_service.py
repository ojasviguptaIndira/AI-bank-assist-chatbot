"""Service layer for analytics queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics_engine.database import AnalyticsDatabase
from analytics_engine.query_templates import QueryTemplates


@dataclass
class AnalyticsService:
    """Provide analytics query methods backed by DuckDB."""

    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self) -> None:
        self.database = AnalyticsDatabase(root=self.root)

    def get_highest_interest_rate(self) -> pd.DataFrame:
        return self._query(QueryTemplates.HIGHEST_INTEREST_RATE)

    def get_lowest_interest_rate(self) -> pd.DataFrame:
        return self._query(QueryTemplates.LOWEST_INTEREST_RATE)

    def get_average_interest_rate(self) -> pd.DataFrame:
        return self._query(QueryTemplates.AVERAGE_INTEREST_RATE)

    def get_average_income(self) -> pd.DataFrame:
        return self._query(QueryTemplates.AVERAGE_INCOME)

    def get_top_states(self, limit: int = 10) -> pd.DataFrame:
        return self._query(QueryTemplates.TOP_STATES, [limit])

    def get_top_purposes(self, limit: int = 10) -> pd.DataFrame:
        return self._query(QueryTemplates.TOP_PURPOSES, [limit])

    def get_grade_statistics(self) -> pd.DataFrame:
        return self._query(QueryTemplates.GRADE_STATISTICS)

    def get_home_ownership_distribution(self) -> pd.DataFrame:
        return self._query(QueryTemplates.HOME_OWNERSHIP_DISTRIBUTION)

    def get_verification_distribution(self) -> pd.DataFrame:
        return self._query(QueryTemplates.VERIFICATION_DISTRIBUTION)

    def get_loan_status_distribution(self) -> pd.DataFrame:
        return self._query(QueryTemplates.LOAN_STATUS_DISTRIBUTION)

    def get_bank_statistics(self, bank: str) -> pd.DataFrame:
        return self._query(QueryTemplates.BANK_STATISTICS, [bank])

    def _query(self, sql: str, params: list[object] | None = None) -> pd.DataFrame:
        with self.database.connect() as connection:
            if params:
                return connection.execute(sql, params).fetchdf()
            return connection.execute(sql).fetchdf()
