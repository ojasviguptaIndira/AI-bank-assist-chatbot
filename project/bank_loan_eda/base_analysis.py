"""Shared dataframe analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DatasetAnalyzer:
    """Reusable dataset analysis operations."""

    name: str
    frame: pd.DataFrame

    def dataset_summary(self) -> pd.DataFrame:
        numeric_cols = self.frame.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.frame.select_dtypes(exclude=[np.number]).columns.tolist()
        summary = {
            "dataset": self.name,
            "rows": len(self.frame),
            "columns": len(self.frame.columns),
            "duplicate_rows": int(self.frame.duplicated().sum()),
            "missing_percent": round(
                float(self.frame.isna().sum().sum()) / max(self.frame.size, 1) * 100, 2
            ),
            "memory_usage_mb": round(
                float(self.frame.memory_usage(deep=True).sum()) / (1024 * 1024), 2
            ),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
        }
        return pd.DataFrame([summary])

    def missing_report(self) -> pd.DataFrame:
        total_rows = max(len(self.frame), 1)
        report = pd.DataFrame(
            {
                "column": self.frame.columns,
                "missing_count": self.frame.isna().sum().values,
            }
        )
        report["missing_percent"] = (report["missing_count"] / total_rows * 100).round(2)
        report["recommendation"] = report["missing_percent"].apply(self._missing_recommendation)
        return report.sort_values(["missing_percent", "column"], ascending=[False, True])

    def column_summary(self) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        total_rows = max(len(self.frame), 1)
        for column in self.frame.columns:
            series = self.frame[column]
            records.append(
                {
                    "column": column,
                    "datatype": str(series.dtype),
                    "unique_values": int(series.nunique(dropna=True)),
                    "sample_values": ", ".join(series.dropna().astype(str).head(3).tolist()),
                    "missing_percent": round(float(series.isna().sum()) / total_rows * 100, 2),
                }
            )
        return pd.DataFrame(records)

    def duplicate_report(self) -> pd.DataFrame:
        duplicates = self.frame[self.frame.duplicated(keep=False)].copy()
        if duplicates.empty:
            return pd.DataFrame(columns=self.frame.columns)
        return duplicates

    def column_recommendations(self) -> pd.DataFrame:
        rows = []
        for column in self.frame.columns:
            series = self.frame[column]
            rows.append(
                {
                    "column": column,
                    "missing_percent": round(series.isna().mean() * 100, 2),
                    "role": self._classify_column(column, series),
                    "action": "DROP" if series.isna().mean() > 0.8 else "KEEP",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _missing_recommendation(value: float) -> str:
        if value >= 60:
            return "DROP"
        if value >= 10:
            return "IMPUTE"
        return "KEEP"

    @staticmethod
    def _classify_column(column: str, series: pd.Series) -> str:
        name = column.lower()
        if "id" in name and series.nunique(dropna=True) >= len(series) * 0.8:
            return "IDENTIFIER"
        if any(token in name for token in ("status", "target", "label", "sentiment")):
            return "TARGET"
        if series.dtype == object:
            avg_length = series.dropna().astype(str).str.len().mean()
            if avg_length and avg_length > 40:
                return "TEXT"
            if series.nunique(dropna=True) <= max(len(series) * 0.2, 50):
                return "CATEGORY"
            return "TEXT"
        if pd.api.types.is_numeric_dtype(series):
            return "NUMERIC"
        return "CATEGORY"
