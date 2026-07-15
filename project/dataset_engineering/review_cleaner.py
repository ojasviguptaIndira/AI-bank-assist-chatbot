"""Review dataset cleaning."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dataset_engineering.cleaner import BaseCleaner


BANK_STANDARDIZATION = {
    "sbi": "SBI",
    "hdfc bank": "HDFC Bank",
    "axis bank": "Axis Bank",
    "kotak": "Kotak",
    "canara bank": "Canara Bank",
    "indusind bank": "IndusInd Bank",
    "idbi": "IDBI",
    "citibank": "Citibank",
    "punjab national bank": "Punjab National Bank",
    "review": "Unknown",
    "bank": "Unknown",
}


@dataclass
class ReviewCleaner(BaseCleaner):
    """Clean review dataset and create sentiment-ready fields."""

    def clean(self) -> pd.DataFrame:
        self.frame = self.frame.copy()
        self.rename_columns_snake_case()
        self.frame = self.frame.drop_duplicates()
        self._standardize_bank_names()
        self._clean_reviews()
        self._convert_types()
        self._drop_empty_reviews()
        self._derive_sentiment()
        return self.frame

    def mapping(self) -> dict[str, object]:
        return {
            "removed_columns": sorted(set(self.removed_columns)),
            "renamed_columns": self.renamed_columns,
            "derived_columns": self.derived_columns,
        }

    def _standardize_bank_names(self) -> None:
        if "bank" not in self.frame.columns:
            return
        bank = self.frame["bank"].astype("string").fillna("Unknown").str.strip().str.lower()
        standardized = bank.map(BANK_STANDARDIZATION).fillna(bank.str.title())
        self.frame["bank"] = standardized

    def _clean_reviews(self) -> None:
        if "review" not in self.frame.columns:
            return
        self.frame["review"] = self.frame["review"].astype("string").fillna("").str.strip()
        self.frame["clean_review"] = self.frame["review"].apply(self.clean_text)
        self.frame["review_length"] = self.frame["clean_review"].str.split().str.len().fillna(0).astype(int)
        self.derived_columns.extend(["clean_review", "review_length"])

    def _convert_types(self) -> None:
        if "rating" in self.frame.columns:
            self.frame["rating"] = pd.to_numeric(self.frame["rating"], errors="coerce")
        if "useful_count" in self.frame.columns:
            self.frame["useful_count"] = pd.to_numeric(self.frame["useful_count"], errors="coerce").fillna(0)
        if "date" in self.frame.columns:
            self.frame["date"] = pd.to_datetime(self.frame["date"], errors="coerce")

    def _drop_empty_reviews(self) -> None:
        if "clean_review" not in self.frame.columns:
            return
        self.frame = self.frame[self.frame["clean_review"].str.len() > 0].copy()

    def _derive_sentiment(self) -> None:
        if "rating" not in self.frame.columns:
            return
        self.frame["sentiment"] = self.frame["rating"].apply(self._label_sentiment)
        self.derived_columns.append("sentiment")

    @staticmethod
    def _label_sentiment(rating: object) -> str:
        if pd.isna(rating):
            return "Unknown"
        if rating >= 4:
            return "Satisfied"
        if rating == 3:
            return "Average"
        return "Dissatisfied"
