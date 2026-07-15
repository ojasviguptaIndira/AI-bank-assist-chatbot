"""Business, schema, and ML recommendation builders."""

from __future__ import annotations

import pandas as pd


class RecommendationBuilder:
    """Create human-readable recommendation reports."""

    @staticmethod
    def mysql_schema(loan_frame: pd.DataFrame, review_frame: pd.DataFrame) -> str:
        loan_indexes = RecommendationBuilder._top_numeric_columns(loan_frame, 5)
        review_indexes = ["bank", "rating", "date", "useful_count"]
        return "\n".join(
            [
                "Primary Key",
                "Loan.id",
                "Reviews.review_id (synthetic auto-increment recommended)",
                "Bank.bank_id (dimension table)",
                "",
                "Indexes",
                f"Loan: {', '.join(loan_indexes)}",
                f"Reviews: {', '.join(review_indexes)}",
                "Bank: bank_name unique index",
                "",
                "Analytics Columns",
                "Loan: loan_amnt, int_rate, annual_inc, grade, purpose, loan_status, addr_state",
                "Reviews: bank, rating, sentiment_label, useful_count, date",
                "",
                "Recommended Tables",
                "Loan fact table",
                "Reviews fact table",
                "Bank dimension table",
            ]
        )

    @staticmethod
    def ml_dataset(loan_frame: pd.DataFrame, review_frame: pd.DataFrame) -> str:
        loan_features = RecommendationBuilder._top_numeric_columns(loan_frame, 8)
        review_features = ["clean_review", "rating", "bank", "useful_count", "review_length"]
        dropped = [column for column in loan_frame.columns if loan_frame[column].isna().mean() > 0.8]
        return "\n".join(
            [
                "Loan Modeling Recommendation",
                f"Features: {', '.join(loan_features)}",
                "Target: loan_status",
                f"Dropped Columns: {', '.join(dropped[:15]) or 'None'}",
                "Synthetic Data Requirement: Not recommended until class imbalance is measured after target cleanup.",
                f"Class Balance: {RecommendationBuilder._class_balance(loan_frame, 'loan_status')}",
                "",
                "Review Sentiment Recommendation",
                f"Features: {', '.join(review_features)}",
                "Target: sentiment_label",
                "Dropped Columns: bank_image",
                "Synthetic Data Requirement: Consider augmentation only if Dissatisfied reviews are underrepresented.",
                f"Class Balance: {RecommendationBuilder._class_balance(review_frame, 'sentiment_label')}",
            ]
        )

    @staticmethod
    def business_insight_text(loan_insights: pd.DataFrame, review_insights: pd.DataFrame) -> str:
        parts = ["Loan Summary", loan_insights.to_string(index=False), "", "Bank Ratings", review_insights.to_string(index=False)]
        return "\n".join(parts)

    @staticmethod
    def _top_numeric_columns(frame: pd.DataFrame, count: int) -> list[str]:
        numeric_cols = frame.select_dtypes(include="number").columns.tolist()
        priority = [col for col in ["loan_amnt", "int_rate", "annual_inc", "rating", "useful_count"] if col in numeric_cols]
        remainder = [col for col in numeric_cols if col not in priority]
        return (priority + remainder)[:count]

    @staticmethod
    def _class_balance(frame: pd.DataFrame, column: str) -> str:
        if column not in frame.columns:
            return "Column not available"
        counts = frame[column].fillna("Unknown").value_counts(normalize=True).mul(100).round(2)
        return ", ".join(f"{label}: {value}%" for label, value in counts.items())
