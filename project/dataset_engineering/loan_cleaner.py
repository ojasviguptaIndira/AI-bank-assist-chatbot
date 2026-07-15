"""Loan dataset cleaning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset_engineering.cleaner import BaseCleaner


IDENTIFIER_COLUMNS = {"id", "member_id"}
TEXT_DROP_COLUMNS = {"url", "desc"}
DATE_COLUMNS = {
    "issue_d",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d",
    "earliest_cr_line",
}


@dataclass
class LoanCleaner(BaseCleaner):
    """Clean loan dataset for analytics and ML."""

    def clean(self) -> pd.DataFrame:
        self.frame = self.frame.copy().replace({"NA": np.nan, "": np.nan})
        self.drop_columns([column for column in self.frame.columns if column in IDENTIFIER_COLUMNS | TEXT_DROP_COLUMNS])
        self.drop_high_missing(0.70)
        self._convert_percent_column("int_rate")
        self._convert_percent_column("revol_util")
        self._convert_term_column("term")
        self._convert_dates()
        self._convert_numeric_columns()
        self._fill_missing_values()
        return self.frame

    def mapping(self) -> dict[str, object]:
        return {
            "removed_columns": sorted(set(self.removed_columns)),
            "renamed_columns": self.renamed_columns,
            "derived_columns": self.derived_columns,
            "date_columns": sorted(DATE_COLUMNS.intersection(self.frame.columns)),
        }

    def _convert_percent_column(self, column: str) -> None:
        if column not in self.frame.columns:
            return
        cleaned = self.frame[column].astype("string").str.replace("%", "", regex=False).str.strip()
        self.frame[column] = pd.to_numeric(cleaned, errors="coerce")

    def _convert_term_column(self, column: str) -> None:
        if column not in self.frame.columns:
            return
        cleaned = self.frame[column].astype("string").str.extract(r"(\d+)")[0]
        self.frame[column] = pd.to_numeric(cleaned, errors="coerce")

    def _convert_dates(self) -> None:
        for column in DATE_COLUMNS:
            if column in self.frame.columns:
                self.frame[column] = self.parse_dates(self.frame[column])

    def _convert_numeric_columns(self) -> None:
        for column in self.frame.columns:
            if pd.api.types.is_datetime64_any_dtype(self.frame[column]):
                continue
            try:
                converted = pd.to_numeric(self.frame[column])
            except (TypeError, ValueError):
                continue
            if converted.notna().sum() > 0:
                self.frame[column] = converted

    def _fill_missing_values(self) -> None:
        for column in self.frame.columns:
            series = self.frame[column]
            if pd.api.types.is_numeric_dtype(series):
                self.frame[column] = series.fillna(series.median())
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            mode = series.mode(dropna=True)
            if not mode.empty:
                self.frame[column] = series.fillna(mode.iloc[0])
