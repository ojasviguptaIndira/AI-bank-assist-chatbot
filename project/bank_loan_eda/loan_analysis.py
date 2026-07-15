"""Loan dataset analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bank_loan_eda.base_analysis import DatasetAnalyzer
from bank_loan_eda.plotting import PlotFactory


@dataclass
class LoanEDA:
    """Perform loan dataset exploratory analysis."""

    frame: pd.DataFrame
    plot_dir: Path

    def __post_init__(self) -> None:
        self.frame = self._prepare_frame(self.frame.copy())
        self.base = DatasetAnalyzer("Loan Dataset", self.frame)

    def analyze(self) -> dict[str, pd.DataFrame]:
        numeric = self.frame.select_dtypes(include=[np.number])
        categorical = self.frame.select_dtypes(exclude=[np.number])
        outputs = {
            "Dataset Summary": self.base.dataset_summary(),
            "Missing Report": self.base.missing_report(),
            "Column Summary": self.base.column_summary(),
            "Numerical Statistics": self._numerical_statistics(numeric),
            "Categorical Statistics": self._categorical_statistics(categorical),
            "Duplicate Report": self.base.duplicate_report(),
            "Correlation Matrix": numeric.corr(numeric_only=True).round(3).reset_index(),
            "Outlier Report": self._outlier_report(numeric),
            "Skewness Report": self._skewness_report(numeric),
            "Business Insights": self._business_insights(),
            "Grade Analysis": self._grade_analysis(),
            "Column Recommendations": self.base.column_recommendations(),
        }
        self._generate_plots(numeric)
        return outputs

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("int_rate", "revol_util"):
            if column in frame.columns:
                frame[column] = (
                    frame[column].astype(str).str.replace("%", "", regex=False).replace("nan", np.nan)
                )
        for column in frame.columns:
            try:
                frame[column] = pd.to_numeric(frame[column])
            except (TypeError, ValueError):
                continue
        return frame.replace({"NA": np.nan, "": np.nan})

    def _numerical_statistics(self, numeric: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for column in numeric.columns:
            series = numeric[column].dropna()
            if series.empty:
                continue
            mode_value = series.mode().iloc[0] if not series.mode().empty else np.nan
            rows.append(
                {
                    "column": column,
                    "mean": round(series.mean(), 3),
                    "median": round(series.median(), 3),
                    "mode": round(float(mode_value), 3) if pd.notna(mode_value) else np.nan,
                    "min": round(series.min(), 3),
                    "max": round(series.max(), 3),
                    "std": round(series.std(), 3),
                    "variance": round(series.var(), 3),
                    "skewness": round(series.skew(), 3),
                    "kurtosis": round(series.kurt(), 3),
                }
            )
        return pd.DataFrame(rows)

    def _categorical_statistics(self, categorical: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for column in categorical.columns:
            series = categorical[column].dropna().astype(str)
            if series.empty:
                continue
            rows.append(
                {
                    "column": column,
                    "unique_values": int(series.nunique()),
                    "top_value": series.mode().iloc[0],
                    "top_frequency": int(series.value_counts().iloc[0]),
                }
            )
        return pd.DataFrame(rows)

    def _outlier_report(self, numeric: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for column in numeric.columns:
            series = numeric[column].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            rows.append(
                {
                    "column": column,
                    "outlier_count": int(outliers.count()),
                    "outlier_percent": round(outliers.count() / max(len(series), 1) * 100, 2),
                    "lower_bound": round(lower, 3),
                    "upper_bound": round(upper, 3),
                }
            )
        return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)

    def _skewness_report(self, numeric: pd.DataFrame) -> pd.DataFrame:
        skew = numeric.skew().dropna().sort_values(ascending=False)
        return skew.rename("skewness").reset_index().rename(columns={"index": "column"})

    def _business_insights(self) -> pd.DataFrame:
        rows = []
        metrics = {
            "Average Loan Amount": self._safe_stat("loan_amnt", "mean"),
            "Highest Loan Amount": self._safe_stat("loan_amnt", "max"),
            "Average Interest Rate": self._safe_stat("int_rate", "mean"),
            "Highest Interest Rate": self._safe_stat("int_rate", "max"),
            "Lowest Interest Rate": self._safe_stat("int_rate", "min"),
            "Average Income": self._safe_stat("annual_inc", "mean"),
            "Most Common Purpose": self._safe_mode("purpose"),
            "Most Common Grade": self._safe_mode("grade"),
            "Most Common Loan Status": self._safe_mode("loan_status"),
            "Top States": ", ".join(self._top_values("addr_state", 5)),
            "Top Verification Status": ", ".join(self._top_values("verification_status", 3)),
        }
        for metric, value in metrics.items():
            rows.append({"metric": metric, "value": value})
        return pd.DataFrame(rows)

    def _grade_analysis(self) -> pd.DataFrame:
        required = {"grade", "loan_amnt", "int_rate", "annual_inc"}
        if not required.issubset(self.frame.columns):
            return pd.DataFrame(columns=["grade"])
        grouped = self.frame.groupby("grade", dropna=False).agg(
            average_loan_amount=("loan_amnt", "mean"),
            average_interest_rate=("int_rate", "mean"),
            average_income=("annual_inc", "mean"),
            loan_count=("grade", "size"),
        )
        return grouped.reset_index().round(2).sort_values("grade")

    def _generate_plots(self, numeric: pd.DataFrame) -> None:
        mapping = {
            "loan_amnt": ("Loan Amount Distribution", "loan_amount_distribution.png", "hist"),
            "int_rate": ("Interest Rate Distribution", "interest_rate_distribution.png", "hist"),
            "loan_status": ("Loan Status", "loan_status.png", "bar"),
            "purpose": ("Purpose", "purpose.png", "bar"),
            "grade": ("Grade", "grade.png", "bar"),
            "verification_status": ("Verification Status", "verification_status.png", "bar"),
            "home_ownership": ("Home Ownership", "home_ownership.png", "bar"),
            "addr_state": ("State Distribution", "state_distribution.png", "bar"),
        }
        for column, (title, filename, chart) in mapping.items():
            if column not in self.frame.columns:
                continue
            path = self.plot_dir / filename
            if chart == "hist":
                PlotFactory.histogram(self.frame[column], title, path)
            else:
                PlotFactory.bar(self.frame[column], title, path)

        focus_columns = [c for c in ("loan_amnt", "int_rate", "annual_inc", "installment", "dti") if c in numeric]
        if focus_columns:
            corr = numeric[focus_columns].corr(numeric_only=True).round(2)
            PlotFactory.heatmap(corr, "Correlation Heatmap", self.plot_dir / "correlation_heatmap.png")

    def _safe_stat(self, column: str, metric: str) -> float | str:
        if column not in self.frame.columns:
            return "Column not available"
        series = self.frame[column].dropna()
        if series.empty:
            return "No data"
        return round(float(getattr(series, metric)()), 2)

    def _safe_mode(self, column: str) -> str:
        if column not in self.frame.columns or self.frame[column].dropna().empty:
            return "Column not available"
        return str(self.frame[column].mode(dropna=True).iloc[0])

    def _top_values(self, column: str, count: int) -> list[str]:
        if column not in self.frame.columns:
            return []
        return self.frame[column].dropna().astype(str).value_counts().head(count).index.tolist()
