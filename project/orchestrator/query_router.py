"""Route queries to analytics and sentiment engines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from analytics_engine.analytics_service import AnalyticsService
from analytics_engine.query_templates import QueryTemplates
from sentiment.predictor import SentimentPredictor


@dataclass
class QueryRouter:
    """Execute orchestrated requests against deterministic engines."""

    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self) -> None:
        self.analytics_service = AnalyticsService(root=self.root)
        self.sentiment_predictor = SentimentPredictor(root=self.root)
        self.review_frame = pd.read_csv(self.root / "artifacts" / "review_cleaned.csv", low_memory=False)
        self.bank_statistics = pd.read_csv(self.root / "artifacts" / "bank_statistics.csv", low_memory=False)
        self.loan_frame = pd.read_csv(self.root / "artifacts" / "loan_cleaned.csv", low_memory=False)

    def route(self, classification: dict[str, object], extraction: dict[str, object]) -> dict[str, object]:
        query_type = classification["type"]
        if query_type == "ANALYTICAL":
            return self._run_analytics(extraction)
        if query_type == "SENTIMENT":
            return self._run_sentiment(extraction)
        if query_type == "HYBRID":
            return self._run_hybrid(extraction)
        return {
            "type": "unknown",
            "intent": "UNSUPPORTED",
            "message": "Unsupported Query",
        }

    def _run_analytics(self, extraction: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        intent = extraction["intent"]
        entities = extraction["entities"]
        dispatch = {
            "MAX_INTEREST_RATE": (self.analytics_service.get_highest_interest_rate, QueryTemplates.HIGHEST_INTEREST_RATE),
            "MIN_INTEREST_RATE": (self.analytics_service.get_lowest_interest_rate, QueryTemplates.LOWEST_INTEREST_RATE),
            "AVERAGE_INTEREST_RATE": (self.analytics_service.get_average_interest_rate, QueryTemplates.AVERAGE_INTEREST_RATE),
            "AVERAGE_INCOME": (self.analytics_service.get_average_income, QueryTemplates.AVERAGE_INCOME),
            "TOP_STATE": (self.analytics_service.get_top_states, QueryTemplates.TOP_STATES),
            "TOP_PURPOSE": (self.analytics_service.get_top_purposes, QueryTemplates.TOP_PURPOSES),
            "GRADE_STATISTICS": (self.analytics_service.get_grade_statistics, QueryTemplates.GRADE_STATISTICS),
            "HOME_OWNERSHIP": (self.analytics_service.get_home_ownership_distribution, QueryTemplates.HOME_OWNERSHIP_DISTRIBUTION),
            "VERIFICATION": (self.analytics_service.get_verification_distribution, QueryTemplates.VERIFICATION_DISTRIBUTION),
            "LOAN_STATUS": (self.analytics_service.get_loan_status_distribution, QueryTemplates.LOAN_STATUS_DISTRIBUTION),
            "LOAN_STATISTICS": (self.analytics_service.get_grade_statistics, QueryTemplates.GRADE_STATISTICS),
        }
        func, sql = dispatch.get(intent)
        frame = func(5) if intent in {"TOP_STATE", "TOP_PURPOSE"} else func()
        return {
            "type": "analytics",
            "intent": intent,
            "entities": entities,
            "data": frame.to_dict(orient="records"),
            "metadata": {
                "source": "DuckDB",
                "records_analyzed": int(len(self.loan_frame)),
                "sql_executed": " ".join(sql.split()),
                "execution_time_sec": round(time.perf_counter() - started, 4),
            },
        }

    def _run_sentiment(self, extraction: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        entities = extraction["entities"]
        bank = entities.get("bank")
        intent = extraction["intent"]
        if intent == "COMMON_COMPLAINT":
            return self._common_complaints(bank, entities, started)
        if bank:
            bank_frame = self.review_frame[self.review_frame["bank"].str.lower() == bank.lower()].copy()
            bank_reviews = self._bank_reviews(bank)
            prediction = self.sentiment_predictor.predict(bank_reviews)
            stats = self.analytics_service.get_bank_statistics(bank).to_dict(orient="records")
            return {
                "type": "sentiment",
                "intent": intent,
                "bank": bank,
                "prediction": prediction["sentiment"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "bank_statistics": stats,
                "metadata": {
                    "source": "Customer Reviews",
                    "records_analyzed": int(len(bank_frame)),
                    "generated_using": "Sentiment Model",
                    "execution_time_sec": round(time.perf_counter() - started, 4),
                },
            }
        prediction = self.sentiment_predictor.predict(entities["raw_query"])
        return {
            "type": "sentiment",
            "intent": intent,
            "prediction": prediction["sentiment"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "metadata": {
                "source": "Sentiment Model",
                "records_analyzed": 1,
                "generated_using": "Sentiment Model",
                "execution_time_sec": round(time.perf_counter() - started, 4),
            },
        }

    def _run_hybrid(self, extraction: dict[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        entities = extraction["entities"]
        analytics_payload = {
            "available": False,
            "message": "Bank-level loan pricing is not available in the current loan dataset.",
            "metadata": {
                "source": "DuckDB",
                "records_analyzed": int(len(self.loan_frame)),
                "execution_time_sec": 0.0,
            },
        }
        sentiment_payload: dict[str, object]
        banks = entities.get("banks") or []
        if banks:
            comparisons = []
            for bank in banks:
                stats_frame = self.analytics_service.get_bank_statistics(bank)
                bank_text = self._bank_reviews(bank)
                prediction = self.sentiment_predictor.predict(bank_text)
                comparisons.append(
                    {
                        "bank": bank,
                        "statistics": stats_frame.to_dict(orient="records"),
                        "prediction": prediction,
                        "records_analyzed": int(
                            len(self.review_frame[self.review_frame["bank"].str.lower() == bank.lower()])
                        ),
                    }
                )
            sentiment_payload = {
                "available": True,
                "banks": comparisons,
                "metadata": {
                    "source": "Customer Reviews",
                    "generated_using": "Sentiment Model",
                },
            }
        else:
            sentiment_payload = {
                "available": False,
                "message": "No bank entities were identified for hybrid comparison.",
            }
        return {
            "type": "hybrid",
            "intent": extraction["intent"],
            "entities": entities,
            "analytics": analytics_payload,
            "sentiment": sentiment_payload,
            "metadata": {
                "execution_time_sec": round(time.perf_counter() - started, 4),
            },
        }

    def _common_complaints(self, bank: str | None, entities: dict[str, object], started: float) -> dict[str, object]:
        frame = self.review_frame.copy()
        if bank:
            frame = frame[frame["bank"].str.lower() == bank.lower()]
        complaints = frame[frame["sentiment"] == "Dissatisfied"]["clean_review"].astype(str)
        words = (
            complaints.str.split().explode().value_counts().head(10).index.tolist()
            if not complaints.empty else []
        )
        return {
            "type": "sentiment",
            "intent": "COMMON_COMPLAINT",
            "bank": bank,
            "entities": entities,
            "complaint_keywords": words,
            "metadata": {
                "source": "Customer Reviews",
                "records_analyzed": int(len(frame)),
                "generated_using": "Sentiment Model",
                "execution_time_sec": round(time.perf_counter() - started, 4),
            },
        }

    def _bank_reviews(self, bank: str) -> str:
        subset = self.review_frame[self.review_frame["bank"].str.lower() == bank.lower()].copy()
        if subset.empty:
            return bank
        top_reviews = subset.sort_values("useful_count", ascending=False)["review"].astype(str).head(20)
        return " ".join(top_reviews.tolist())


if __name__ == "__main__":
    router = QueryRouter()
    sample = router.route(
        {"type": "SENTIMENT"},
        {"intent": "BANK_REVIEW", "entities": {"bank": "SBI", "raw_query": "Is SBI good?"}},
    )
    print(sample)
