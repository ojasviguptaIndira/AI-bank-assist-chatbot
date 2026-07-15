"""Entity extraction using regex, dictionaries, and RapidFuzz."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process


@dataclass
class EntityExtractor:
    """Extract query entities deterministically."""

    banks: tuple[str, ...] = (
        "SBI", "HDFC Bank", "ICICI", "Axis Bank", "PNB", "Kotak",
        "Canara Bank", "IDBI", "Citibank", "IndusInd Bank",
    )
    loan_types: tuple[str, ...] = (
        "home loan", "personal loan", "car loan", "credit card",
        "debt consolidation", "small business", "major purchase",
    )

    def extract(self, query: str, classification: dict[str, object]) -> dict[str, object]:
        """Return normalized intent, extracted entities, and confidence."""
        normalized = query.lower().strip()
        banks = self._extract_banks(query)
        loan_type = self._extract_loan_type(query)
        aggregation = self._extract_aggregation(normalized)
        metric = self._extract_metric(normalized)

        entities = {
            "banks": banks,
            "bank": banks[0] if len(banks) == 1 else None,
            "loan_type": loan_type,
            "aggregation": aggregation,
            "metric": metric,
            "raw_query": query,
        }
        return {
            "intent": classification["intent"],
            "entities": entities,
            "confidence": round(min(0.99, classification["confidence"] + (0.05 if banks or metric else 0.0)), 2),
        }

    def _extract_banks(self, query: str) -> list[str]:
        matches: list[str] = []
        lowered = query.lower()
        for bank in self.banks:
            alias = bank.lower().replace(" bank", "")
            if re.search(rf"\b{re.escape(alias)}\b", lowered) or re.search(rf"\b{re.escape(bank.lower())}\b", lowered):
                matches.append(bank)
        if matches:
            return self._unique(matches)

        words = re.findall(r"[a-zA-Z]+(?:\s+[a-zA-Z]+)?", query)
        fuzzy = []
        for word in words:
            result = process.extractOne(word, self.banks, scorer=fuzz.WRatio, score_cutoff=85)
            if result:
                fuzzy.append(result[0])
        return self._unique(fuzzy)

    def _extract_loan_type(self, query: str) -> str | None:
        result = process.extractOne(query.lower(), self.loan_types, scorer=fuzz.partial_ratio, score_cutoff=80)
        return result[0] if result else None

    @staticmethod
    def _extract_aggregation(query: str) -> str | None:
        if "highest" in query or "max" in query:
            return "MAX"
        if "lowest" in query or "min" in query:
            return "MIN"
        if "average" in query or "avg" in query:
            return "AVERAGE"
        if "count" in query:
            return "COUNT"
        if "top" in query or "most common" in query:
            return "TOP"
        if "compare" in query or "vs" in query or "versus" in query:
            return "COMPARE"
        return None

    @staticmethod
    def _extract_metric(query: str) -> str | None:
        mapping = {
            "interest": "INTEREST_RATE",
            "loan amount": "LOAN_AMOUNT",
            "income": "AVERAGE_INCOME",
            "purpose": "PURPOSE",
            "grade": "GRADE",
            "state": "STATE",
            "home ownership": "HOME_OWNERSHIP",
            "verification": "VERIFICATION",
            "loan status": "LOAN_STATUS",
            "document": "DOCUMENT_REQUIREMENT",
            "complaint": "COMMON_COMPLAINT",
            "satisfaction": "CUSTOMER_SATISFACTION",
            "review": "BANK_REVIEW",
        }
        for key, value in mapping.items():
            if key in query:
                return value
        return None

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value not in seen:
                result.append(value)
                seen.add(value)
        return result


if __name__ == "__main__":
    extractor = EntityExtractor()
    sample = extractor.extract("Compare SBI and HDFC on reviews", {"intent": "BANK_COMPARISON", "confidence": 0.9})
    print(sample)
