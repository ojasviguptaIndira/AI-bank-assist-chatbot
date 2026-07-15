"""SQL query templates for analytics operations."""

from __future__ import annotations


class QueryTemplates:
    """Centralized SQL templates for analytics queries."""

    HIGHEST_INTEREST_RATE = """
        SELECT MAX(int_rate) AS highest_interest_rate
        FROM loans
    """
    LOWEST_INTEREST_RATE = """
        SELECT MIN(int_rate) AS lowest_interest_rate
        FROM loans
    """
    AVERAGE_INTEREST_RATE = """
        SELECT ROUND(AVG(int_rate), 2) AS average_interest_rate
        FROM loans
    """
    AVERAGE_INCOME = """
        SELECT ROUND(AVG(annual_inc), 2) AS average_income
        FROM loans
    """
    TOP_STATES = """
        SELECT addr_state, COUNT(*) AS loan_count
        FROM loans
        GROUP BY addr_state
        ORDER BY loan_count DESC
        LIMIT ?
    """
    TOP_PURPOSES = """
        SELECT purpose, COUNT(*) AS loan_count
        FROM loans
        GROUP BY purpose
        ORDER BY loan_count DESC
        LIMIT ?
    """
    GRADE_STATISTICS = """
        SELECT
            grade,
            COUNT(*) AS loan_count,
            ROUND(AVG(loan_amnt), 2) AS average_loan_amount,
            ROUND(AVG(int_rate), 2) AS average_interest_rate,
            ROUND(AVG(annual_inc), 2) AS average_income
        FROM loans
        GROUP BY grade
        ORDER BY grade
    """
    HOME_OWNERSHIP_DISTRIBUTION = """
        SELECT home_ownership, COUNT(*) AS loan_count
        FROM loans
        GROUP BY home_ownership
        ORDER BY loan_count DESC
    """
    VERIFICATION_DISTRIBUTION = """
        SELECT verification_status, COUNT(*) AS loan_count
        FROM loans
        GROUP BY verification_status
        ORDER BY loan_count DESC
    """
    LOAN_STATUS_DISTRIBUTION = """
        SELECT loan_status, COUNT(*) AS loan_count
        FROM loans
        GROUP BY loan_status
        ORDER BY loan_count DESC
    """
    BANK_STATISTICS = """
        SELECT *
        FROM bank_statistics
        WHERE LOWER(bank) = LOWER(?)
    """
