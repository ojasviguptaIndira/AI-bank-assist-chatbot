"""Executable sample analytics queries."""

from __future__ import annotations

import logging

from analytics_engine.analytics_service import AnalyticsService


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run sample analytics queries."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    service = AnalyticsService()
    samples = {
        "Highest Interest Rate": service.get_highest_interest_rate(),
        "Lowest Interest Rate": service.get_lowest_interest_rate(),
        "Average Interest Rate": service.get_average_interest_rate(),
        "Average Income": service.get_average_income(),
        "Top States": service.get_top_states(5),
        "Top Purposes": service.get_top_purposes(5),
        "Grade Statistics": service.get_grade_statistics(),
        "Home Ownership Distribution": service.get_home_ownership_distribution(),
        "Verification Distribution": service.get_verification_distribution(),
        "Loan Status Distribution": service.get_loan_status_distribution(),
        "Bank Statistics (SBI)": service.get_bank_statistics("SBI"),
    }
    for label, frame in samples.items():
        LOGGER.info("%s\n%s", label, frame.to_string(index=False))


if __name__ == "__main__":
    main()
