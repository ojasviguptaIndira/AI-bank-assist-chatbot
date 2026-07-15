"""Statistics artifact builders."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StatisticsBuilder:
    """Create aggregated analytics datasets."""

    loan_frame: pd.DataFrame
    review_frame: pd.DataFrame

    def build_loan_statistics(self) -> pd.DataFrame:
        required = {"purpose", "loan_amnt", "int_rate", "annual_inc"}
        if not required.issubset(self.loan_frame.columns):
            return pd.DataFrame(columns=["purpose", "average_loan", "average_interest", "average_income", "loan_count"])
        frame = self.loan_frame.copy()
        grouped = frame.groupby("purpose", dropna=False).agg(
            average_loan=("loan_amnt", "mean"),
            average_interest=("int_rate", "mean"),
            average_income=("annual_inc", "mean"),
            loan_count=("purpose", "size"),
        )
        return grouped.reset_index().round(2).sort_values("loan_count", ascending=False)

    def build_bank_statistics(self) -> pd.DataFrame:
        required = {"bank", "rating", "sentiment", "review_length"}
        if not required.issubset(self.review_frame.columns):
            return pd.DataFrame()
        base_frame = self.review_frame[self.review_frame["bank"].fillna("Unknown") != "Unknown"].copy()
        if base_frame.empty:
            return pd.DataFrame()
        sentiment_counts = (
            base_frame.pivot_table(
                index="bank",
                columns="sentiment",
                values="review",
                aggfunc="count",
                fill_value=0,
            )
            .rename_axis(None, axis=1)
            .reset_index()
        )
        base = base_frame.groupby("bank", dropna=False).agg(
            average_rating=("rating", "mean"),
            total_reviews=("bank", "size"),
            average_review_length=("review_length", "mean"),
        )
        merged = base.reset_index().merge(sentiment_counts, on="bank", how="left")
        for column in ("Satisfied", "Average", "Dissatisfied"):
            if column not in merged.columns:
                merged[column] = 0
        columns = [
            "bank",
            "average_rating",
            "total_reviews",
            "Satisfied",
            "Average",
            "Dissatisfied",
            "average_review_length",
        ]
        return merged[columns].round(2).sort_values("total_reviews", ascending=False)
