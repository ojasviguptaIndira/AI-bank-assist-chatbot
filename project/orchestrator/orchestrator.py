"""End-to-end query orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from orchestrator.entity_extractor import EntityExtractor
from orchestrator.intent_classifier import IntentClassifier
from orchestrator.query_router import QueryRouter
from orchestrator.response_formatter import ResponseFormatter


LOGGER = logging.getLogger(__name__)


@dataclass
class QueryOrchestrator:
    """Classify, extract, route, and format user queries."""

    def __post_init__(self) -> None:
        self.classifier = IntentClassifier()
        self.extractor = EntityExtractor()
        self.router = QueryRouter()
        self.formatter = ResponseFormatter()

    def handle_query(self, query: str) -> dict[str, object]:
        """Return a structured response payload."""
        started = time.perf_counter()
        classification = self.classifier.classify(query)
        extraction = self.extractor.extract(query, classification)
        routed = self.router.route(classification, extraction)
        response = {
            "query": query,
            "classification": classification,
            "extraction": extraction,
            "response": routed,
            "timing": {
                "total_time_sec": round(time.perf_counter() - started, 4),
            },
        }
        LOGGER.info("Handled query with type=%s intent=%s", classification["type"], classification["intent"])
        return response

    def handle_and_format(self, query: str) -> str:
        """Return a JSON string response."""
        return self.formatter.format(self.handle_query(query))


def main() -> None:
    """Interactive CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    orchestrator = QueryOrchestrator()
    print("Query Orchestrator is ready. Type 'exit' to quit.")
    while True:
        query = input("Enter query: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        print(orchestrator.handle_and_format(query))


if __name__ == "__main__":
    main()
