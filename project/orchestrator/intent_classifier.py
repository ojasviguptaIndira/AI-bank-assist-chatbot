"""Rule-based intent classification."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class IntentClassifier:
    """Classify user queries into deterministic intent buckets."""

    analytical_keywords: tuple[str, ...] = (
        "highest", "lowest", "average", "avg", "top", "count", "distribution",
        "statistics", "loan", "interest", "income", "purpose", "grade", "state",
        "home ownership", "verification", "loan status", "most common", "max", "min",
        "approval",
    )
    sentiment_keywords: tuple[str, ...] = (
        "good", "better", "best", "reviews", "review", "complaints", "complaint",
        "satisfaction", "satisfied", "dissatisfied", "customer service", "sentiment",
        "feedback", "experience", "opinion", "summary", "customer", "customers",
        "service", "say",
    )
    comparison_keywords: tuple[str, ...] = (
        "compare", "versus", "vs", "better", "between", "lower", "higher",
        "comparison",
    )

    def classify(self, query: str) -> dict[str, object]:
        """Return high-level type, normalized intent, and confidence."""
        normalized = query.lower().strip()
        analytical_score = self._score(normalized, self.analytical_keywords)
        sentiment_score = self._score(normalized, self.sentiment_keywords)
        comparison_score = self._score(normalized, self.comparison_keywords)
        bank_mentions = len(
            re.findall(
                r"\b(sbi|hdfc|hdfc bank|icici|axis|axis bank|pnb|punjab national bank|kotak|canara|canara bank|idbi|citibank|indusind|indusind bank)\b",
                normalized,
            )
        )

        hybrid_cross_signal = (
            analytical_score
            and sentiment_score
            and any(term in normalized for term in ("review", "customer", "satisfaction", "complaint"))
            and any(term in normalized for term in ("interest", "approval", "loan"))
        )
        if (comparison_score and bank_mentions >= 2) or hybrid_cross_signal:
            return self._payload("HYBRID", self._hybrid_intent(normalized), 0.95)
        if sentiment_score and (bank_mentions or "review" in normalized or "complaint" in normalized or "bank" in normalized):
            return self._payload("SENTIMENT", self._sentiment_intent(normalized), 0.9)
        if analytical_score:
            return self._payload("ANALYTICAL", self._analytical_intent(normalized), 0.88)
        return self._payload("UNKNOWN", "UNSUPPORTED", 0.2)

    def _analytical_intent(self, query: str) -> str:
        if "highest" in query and "interest" in query:
            return "MAX_INTEREST_RATE"
        if "lowest" in query and "interest" in query:
            return "MIN_INTEREST_RATE"
        if ("average" in query or "avg" in query) and "interest" in query:
            return "AVERAGE_INTEREST_RATE"
        if ("average" in query or "avg" in query) and "income" in query:
            return "AVERAGE_INCOME"
        if "top" in query and "state" in query:
            return "TOP_STATE"
        if "top" in query and "purpose" in query:
            return "TOP_PURPOSE"
        if "purpose" in query and "most common" in query:
            return "TOP_PURPOSE"
        if "grade" in query:
            return "GRADE_STATISTICS"
        if "home ownership" in query:
            return "HOME_OWNERSHIP"
        if "verification" in query:
            return "VERIFICATION"
        if "loan status" in query or "status distribution" in query:
            return "LOAN_STATUS"
        return "LOAN_STATISTICS"

    def _sentiment_intent(self, query: str) -> str:
        if "complaint" in query:
            return "COMMON_COMPLAINT"
        if "satisfaction" in query or "satisfied" in query:
            return "CUSTOMER_SATISFACTION"
        if "customer service" in query or "good" in query or "review" in query:
            return "BANK_REVIEW"
        return "REVIEW_SUMMARY"

    def _hybrid_intent(self, query: str) -> str:
        if "compare" in query or "vs" in query or "versus" in query:
            return "BANK_COMPARISON"
        return "COMPARE"

    @staticmethod
    def _score(query: str, keywords: tuple[str, ...]) -> int:
        return sum(1 for keyword in keywords if keyword in query)

    @staticmethod
    def _payload(query_type: str, intent: str, confidence: float) -> dict[str, object]:
        return {
            "type": query_type,
            "intent": intent,
            "confidence": confidence,
        }


if __name__ == "__main__":
    classifier = IntentClassifier()
    print(classifier.classify("Which bank has better customer service?"))
