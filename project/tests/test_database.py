"""Tests for the DuckDB analytics engine."""

from __future__ import annotations

import unittest
from pathlib import Path

from analytics_engine.analytics_service import AnalyticsService
from analytics_engine.database import AnalyticsDatabase


class AnalyticsDatabaseTests(unittest.TestCase):
    """Validate analytics database creation and query access."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parent.parent
        cls.database = AnalyticsDatabase(root=cls.root)
        cls.database.build()
        cls.service = AnalyticsService(root=cls.root)

    def test_database_file_exists(self) -> None:
        self.assertTrue((self.root / "database" / "loan.duckdb").exists())

    def test_top_states_returns_dataframe(self) -> None:
        frame = self.service.get_top_states(5)
        self.assertFalse(frame.empty)
        self.assertIn("addr_state", frame.columns)
        self.assertIn("loan_count", frame.columns)

    def test_bank_statistics_lookup(self) -> None:
        frame = self.service.get_bank_statistics("SBI")
        self.assertFalse(frame.empty)
        self.assertEqual(frame.iloc[0]["bank"], "SBI")


if __name__ == "__main__":
    unittest.main()
