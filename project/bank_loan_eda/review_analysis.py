"""Review dataset analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bank_loan_eda.base_analysis import DatasetAnalyzer
from bank_loan_eda.config import TOP_NGRAM_COUNT, TOP_WORD_COUNT
from bank_loan_eda.plotting import PlotFactory
from bank_loan_eda.text_processing import TextProcessor


@dataclass
class ReviewEDA:
    """Perform review and text analysis."""

    frame: pd.DataFrame
    plot_dir: Path

    def __post_init__(self) -> None:
        self.frame = self.frame.copy().replace({"": pd.NA})
        if "bank" in self.frame.columns:
            self.frame["bank"] = self.frame["bank"].astype("string").str.strip()
            self.frame["bank"] = self.frame["bank"].replace({"review": pd.NA, "bank": pd.NA})
        self.frame["rating"] = pd.to_numeric(self.frame.get("rating"), errors="coerce")
        self.frame["useful_count"] = pd.to_numeric(self.frame.get("useful_count"), errors="coerce")
        self.text_processor = TextProcessor()
        self.frame["clean_review"] = self.frame.get("review", pd.Series(dtype=str)).fillna("").apply(
            self.text_processor.clean_text
        )
        self.frame["review_length"] = self.frame["clean_review"].str.split().str.len()
        self.frame["sentiment_label"] = self.frame["rating"].apply(self._sentiment_label)
        self.base = DatasetAnalyzer("Review Dataset", self.frame)

    def analyze(self) -> dict[str, pd.DataFrame]:
        outputs = {
            "Dataset Summary": self.base.dataset_summary(),
            "Missing Report": self.base.missing_report(),
            "Column Summary": self.base.column_summary(),
            "Duplicate Report": self.base.duplicate_report(),
            "Rating Distribution": self._value_count_frame("rating"),
            "Bank Distribution": self._value_count_frame("bank"),
            "Review Length Statistics": self._review_length_stats(),
            "Top 50 Words": self._pairs_to_frame(self._words()),
            "Top 30 Bigrams": self._pairs_to_frame(self._ngrams((2, 2))),
            "Top 30 Trigrams": self._pairs_to_frame(self._ngrams((3, 3))),
            "Review Count Per Bank": self._bank_counts(),
            "Average Rating Per Bank": self._average_rating_per_bank(),
            "Sentiment Distribution": self._value_count_frame("sentiment_label"),
            "Bank Wise Sentiment Distribution": self._bank_sentiment_distribution(),
            "Complaint Keywords": self._keyword_frame({"dissatisfied"}),
            "Praise Keywords": self._keyword_frame({"satisfied"}),
            "Most Useful Reviews": self._top_reviews("useful_count", ascending=False),
            "Longest Reviews": self._top_reviews("review_length", ascending=False),
            "Business Insights": self._business_insights(),
            "Column Recommendations": self.base.column_recommendations(),
        }
        self._generate_plots()
        return outputs

    def _review_length_stats(self) -> pd.DataFrame:
        series = self.frame["review_length"].dropna()
        if series.empty:
            return pd.DataFrame(columns=["metric", "value"])
        return pd.DataFrame(
            [
                {"metric": "Average Rating", "value": round(float(self.frame["rating"].mean()), 2)},
                {"metric": "Average Review Length", "value": round(float(series.mean()), 2)},
                {"metric": "Median Review Length", "value": round(float(series.median()), 2)},
                {"metric": "Max Review Length", "value": int(series.max())},
            ]
        )

    def _words(self) -> list[tuple[str, int]]:
        return self.text_processor.word_frequency(self.frame["clean_review"], TOP_WORD_COUNT)

    def _ngrams(self, gram_range: tuple[int, int]) -> list[tuple[str, int]]:
        texts = [text for text in self.frame["clean_review"] if text.strip()]
        if not texts:
            return []
        return self.text_processor.ngrams(texts, gram_range, TOP_NGRAM_COUNT)

    def _bank_counts(self) -> pd.DataFrame:
        if "bank" not in self.frame.columns:
            return pd.DataFrame(columns=["bank", "review_count"])
        return self.frame["bank"].value_counts().rename_axis("bank").reset_index(name="review_count")

    def _average_rating_per_bank(self) -> pd.DataFrame:
        required = {"bank", "rating"}
        if not required.issubset(self.frame.columns):
            return pd.DataFrame(columns=["bank", "average_rating"])
        grouped = self.frame.groupby("bank", dropna=False)["rating"].mean().round(2)
        return grouped.reset_index(name="average_rating").sort_values("average_rating", ascending=False)

    def _bank_sentiment_distribution(self) -> pd.DataFrame:
        required = {"bank", "sentiment_label"}
        if not required.issubset(self.frame.columns):
            return pd.DataFrame()
        grouped = self.frame.groupby(["bank", "sentiment_label"]).size().unstack(fill_value=0)
        return grouped.reset_index()

    def _keyword_frame(self, sentiments: set[str]) -> pd.DataFrame:
        subset = self.frame[self.frame["sentiment_label"].str.lower().isin(sentiments)]
        return self._pairs_to_frame(self.text_processor.word_frequency(subset["clean_review"], 20))

    def _top_reviews(self, column: str, ascending: bool) -> pd.DataFrame:
        required = {"bank", "rating", "review"}
        if column not in self.frame.columns or not required.issubset(self.frame.columns):
            return pd.DataFrame()
        base_columns = ["bank", "rating", column, "review"]
        return self.frame[base_columns].sort_values(column, ascending=ascending).head(10)

    def _business_insights(self) -> pd.DataFrame:
        rows = [
            {"metric": "Average Rating", "value": round(float(self.frame["rating"].mean()), 2)},
            {"metric": "Most Reviewed Bank", "value": self._top_label("bank")},
            {"metric": "Most Common Sentiment", "value": self._top_label("sentiment_label")},
            {"metric": "Longest Review Bank", "value": self._bank_for_max("review_length")},
            {"metric": "Most Useful Review Bank", "value": self._bank_for_max("useful_count")},
        ]
        return pd.DataFrame(rows)

    def _generate_plots(self) -> None:
        if "rating" in self.frame.columns:
            PlotFactory.bar(self.frame["rating"], "Rating Distribution", self.plot_dir / "rating_distribution.png")
        if "bank" in self.frame.columns:
            PlotFactory.bar(self.frame["bank"], "Bank Distribution", self.plot_dir / "bank_distribution.png")
        PlotFactory.histogram(
            self.frame["review_length"], "Review Length Distribution", self.plot_dir / "review_length_distribution.png"
        )
        if "bank" in self.frame.columns and "rating" in self.frame.columns:
            avg = self._average_rating_per_bank().set_index("bank")["average_rating"]
            PlotFactory.bar_from_series(
                avg, "Average Rating Per Bank", self.plot_dir / "average_rating_per_bank.png"
            )
        PlotFactory.wordcloud(
            " ".join(self.frame["clean_review"].tolist()),
            "Review Word Cloud",
            self.plot_dir / "review_wordcloud.png",
        )

    def _value_count_frame(self, column: str) -> pd.DataFrame:
        if column not in self.frame.columns:
            return pd.DataFrame(columns=[column, "count"])
        return self.frame[column].value_counts(dropna=False).rename_axis(column).reset_index(name="count")

    @staticmethod
    def _pairs_to_frame(pairs: list[tuple[str, int]]) -> pd.DataFrame:
        return pd.DataFrame(pairs, columns=["token", "count"])

    @staticmethod
    def _sentiment_label(rating: object) -> str:
        if pd.isna(rating):
            return "Unknown"
        if rating >= 4:
            return "Satisfied"
        if rating == 3:
            return "Average"
        return "Dissatisfied"

    def _top_label(self, column: str) -> str:
        if column not in self.frame.columns or self.frame[column].dropna().empty:
            return "Column not available"
        return str(self.frame[column].mode(dropna=True).iloc[0])

    def _bank_for_max(self, column: str) -> str:
        if column not in self.frame.columns or "bank" not in self.frame.columns:
            return "Column not available"
        subset = self.frame[["bank", column]].dropna()
        if subset.empty:
            return "No data"
        return str(subset.sort_values(column, ascending=False).iloc[0]["bank"])
