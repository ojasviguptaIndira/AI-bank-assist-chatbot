"""Query orchestrator tests with 100+ sample queries."""

from __future__ import annotations

import unittest

from orchestrator.orchestrator import QueryOrchestrator


ANALYTICAL_QUERIES = [
    "Highest interest rate",
    "Lowest interest rate",
    "Average interest rate",
    "Average income",
    "Top states",
    "Top purposes",
    "Most common loan purpose",
    "Loan statistics by grade",
    "Home ownership distribution",
    "Verification distribution",
    "Loan status distribution",
    "What is the max interest rate?",
    "Show the minimum interest rate",
    "Give me the average borrower income",
    "Which states have the most loans?",
    "Which purposes are top?",
    "Show grade statistics",
    "Distribution of home ownership",
    "Distribution of verification status",
    "Count loan status groups",
    "Top 5 states",
    "Top 5 purposes",
    "Average annual income",
    "Highest loan interest",
    "Lowest loan interest",
    "Purpose statistics",
    "Grade wise loan statistics",
    "Loan status stats",
    "Most common purpose",
    "Verification status summary",
]

SENTIMENT_QUERIES = [
    "Is SBI good?",
    "Is HDFC good for loans?",
    "Which bank has better customer service?",
    "Show SBI review summary",
    "Customer satisfaction for Axis",
    "Common complaints about SBI",
    "What do customers say about HDFC?",
    "Is Kotak a good bank?",
    "Review summary for PNB",
    "Bank review for Citibank",
    "Tell me customer satisfaction for SBI",
    "Any complaints about Axis Bank?",
    "How are HDFC reviews?",
    "Is Canara Bank good?",
    "Review opinion on IDBI",
    "Customer feedback for Kotak",
    "Dissatisfied users at SBI",
    "Complaint keywords for HDFC Bank",
    "What is the sentiment for Axis Bank?",
    "Summarize reviews for Citibank",
    "SBI customer service review",
    "HDFC customer satisfaction",
    "PNB review summary",
    "Citibank complaints",
    "Kotak feedback",
    "Axis Bank opinion",
    "Canara customer satisfaction",
    "IDBI review summary",
    "IndusInd Bank review summary",
    "Bank review for SBI",
]

HYBRID_QUERIES = [
    "Compare SBI and HDFC",
    "Compare Axis and Kotak reviews and analytics",
    "Which bank has lower interest rate and better reviews?",
    "Which bank has fastest approval and highest customer satisfaction?",
    "Compare SBI vs HDFC",
    "Compare PNB and SBI customer service",
    "SBI versus HDFC satisfaction and interest",
    "Compare Axis Bank and Citibank",
    "Which is better SBI or Axis?",
    "Compare HDFC and Kotak",
    "Which bank is better between SBI and HDFC?",
    "Compare SBI and PNB for customer satisfaction",
    "HDFC vs Axis reviews",
    "Citibank versus SBI",
    "Compare Canara Bank and IDBI",
    "SBI and Kotak comparison",
    "Axis vs HDFC which is better",
    "Compare IndusInd Bank and Citibank",
    "SBI vs PNB loan and review comparison",
    "Compare Kotak versus HDFC",
    "Which bank is better HDFC or SBI",
    "Compare SBI HDFC Axis",
    "SBI vs HDFC interest and reviews",
    "Compare Canara and PNB",
    "Axis and Kotak comparison",
]

UNKNOWN_QUERIES = [
    "Weather today",
    "Who won IPL",
    "Tell me a joke",
    "Capital of France",
    "Open the camera",
    "What is the time now?",
    "How to cook pasta?",
    "Who is the president?",
    "Play music",
    "Random unrelated question",
    "Best movie this year",
    "Show me cricket scores",
    "What is quantum physics?",
    "Translate hello to French",
    "Who won yesterday?",
    "Latest stock price of Tesla",
    "Temperature in Delhi",
    "How are you?",
    "Set an alarm",
    "Search the web for me",
    "Who won the match",
    "What is today's date",
    "news headlines",
    "best restaurant nearby",
    "write a poem",
]


class QueryOrchestratorTests(unittest.TestCase):
    """Validate the query orchestrator with broad deterministic coverage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.orchestrator = QueryOrchestrator()

    def test_query_count(self) -> None:
        total = len(ANALYTICAL_QUERIES) + len(SENTIMENT_QUERIES) + len(HYBRID_QUERIES) + len(UNKNOWN_QUERIES)
        self.assertGreaterEqual(total, 100)

    def test_analytical_queries(self) -> None:
        for query in ANALYTICAL_QUERIES:
            with self.subTest(query=query):
                payload = self.orchestrator.handle_query(query)
                self.assertEqual(payload["classification"]["type"], "ANALYTICAL")
                self.assertEqual(payload["response"]["type"], "analytics")

    def test_sentiment_queries(self) -> None:
        for query in SENTIMENT_QUERIES:
            with self.subTest(query=query):
                payload = self.orchestrator.handle_query(query)
                self.assertEqual(payload["classification"]["type"], "SENTIMENT")
                self.assertEqual(payload["response"]["type"], "sentiment")

    def test_hybrid_queries(self) -> None:
        for query in HYBRID_QUERIES:
            with self.subTest(query=query):
                payload = self.orchestrator.handle_query(query)
                self.assertEqual(payload["classification"]["type"], "HYBRID")
                self.assertEqual(payload["response"]["type"], "hybrid")

    def test_unknown_queries(self) -> None:
        for query in UNKNOWN_QUERIES:
            with self.subTest(query=query):
                payload = self.orchestrator.handle_query(query)
                self.assertEqual(payload["classification"]["type"], "UNKNOWN")
                self.assertEqual(payload["response"]["type"], "unknown")


if __name__ == "__main__":
    unittest.main()
