"""Orchestrate the end-to-end EDA workflow."""

from __future__ import annotations

import logging

import pandas as pd

from bank_loan_eda.bootstrap import DatasetBootstrapper
from bank_loan_eda.config import ProjectPaths
from bank_loan_eda.loan_analysis import LoanEDA
from bank_loan_eda.logging_utils import configure_logging
from bank_loan_eda.recommendations import RecommendationBuilder
from bank_loan_eda.reporting import ReportWriter, table_to_text
from bank_loan_eda.review_analysis import ReviewEDA


LOGGER = logging.getLogger(__name__)


class EDAPipeline:
    """Coordinate data loading, analysis, and output generation."""

    def __init__(self) -> None:
        configure_logging()
        self.paths = ProjectPaths()
        DatasetBootstrapper(self.paths).prepare()

    def run(self) -> None:
        LOGGER.info("Loading datasets")
        loan_frame = pd.read_csv(self.paths.loan_dataset, low_memory=False)
        review_frame = pd.read_csv(self.paths.review_dataset, low_memory=False)

        LOGGER.info("Running loan analysis")
        loan_eda = LoanEDA(loan_frame, self.paths.loan_plot_dir)
        loan_outputs = loan_eda.analyze()

        LOGGER.info("Running review analysis")
        review_eda = ReviewEDA(review_frame, self.paths.review_plot_dir)
        review_outputs = review_eda.analyze()

        LOGGER.info("Writing Excel workbooks")
        ReportWriter.write_excel(self.paths.excel_dir / "Loan_Dataset_Analysis.xlsx", loan_outputs)
        ReportWriter.write_excel(self.paths.excel_dir / "Review_Dataset_Analysis.xlsx", review_outputs)
        ReportWriter.write_excel(
            self.paths.excel_dir / "Business_Insights.xlsx",
            self._business_workbook(loan_eda.frame, loan_outputs, review_outputs),
        )

        LOGGER.info("Writing text reports")
        self._write_reports(loan_eda.frame, review_eda.frame, loan_outputs, review_outputs)
        LOGGER.info("EDA pipeline completed successfully")

    def _business_workbook(
        self,
        loan_frame: pd.DataFrame,
        loan_outputs: dict[str, pd.DataFrame],
        review_outputs: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        workbook = {
            "Loan Summary": loan_outputs["Business Insights"],
            "Grade Analysis": loan_outputs["Grade Analysis"],
            "Purpose Analysis": self._group_metric(loan_frame, "purpose"),
            "State Analysis": self._group_metric(loan_frame, "addr_state"),
            "Bank Ratings": review_outputs["Average Rating Per Bank"],
            "Sentiment Distribution": review_outputs["Bank Wise Sentiment Distribution"],
        }
        return workbook

    def _write_reports(
        self,
        loan_frame: pd.DataFrame,
        review_frame: pd.DataFrame,
        loan_outputs: dict[str, pd.DataFrame],
        review_outputs: dict[str, pd.DataFrame],
    ) -> None:
        ReportWriter.write_text(
            self.paths.reports_dir / "Dataset_Summary.txt",
            "Dataset Summary",
            [
                ("Loan Dataset Summary", table_to_text(loan_outputs["Dataset Summary"])),
                ("Review Dataset Summary", table_to_text(review_outputs["Dataset Summary"])),
                ("Loan Missing Report", table_to_text(loan_outputs["Missing Report"], 20)),
                ("Review Missing Report", table_to_text(review_outputs["Missing Report"], 20)),
            ],
        )
        ReportWriter.write_text(
            self.paths.reports_dir / "Business_Insights.txt",
            "Business Insights",
            [
                ("Loan Insights", table_to_text(loan_outputs["Business Insights"])),
                ("Review Insights", table_to_text(review_outputs["Business Insights"])),
                (
                    "Combined Narrative",
                    RecommendationBuilder.business_insight_text(
                        loan_outputs["Business Insights"],
                        review_outputs["Business Insights"],
                    ),
                ),
            ],
        )
        ReportWriter.write_text(
            self.paths.reports_dir / "Column_Recommendations.txt",
            "Column Recommendations",
            [
                ("Loan Columns", table_to_text(loan_outputs["Column Recommendations"])),
                ("Review Columns", table_to_text(review_outputs["Column Recommendations"])),
            ],
        )
        ReportWriter.write_text(
            self.paths.reports_dir / "MySQL_Schema_Recommendation.txt",
            "MySQL Schema Recommendation",
            [("Recommendation", RecommendationBuilder.mysql_schema(loan_frame, review_frame))],
        )
        ReportWriter.write_text(
            self.paths.reports_dir / "ML_Dataset_Recommendation.txt",
            "ML Dataset Recommendation",
            [("Recommendation", RecommendationBuilder.ml_dataset(loan_frame, review_frame))],
        )

    @staticmethod
    def _group_metric(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        required = {column, "loan_amnt", "int_rate"}
        if not required.issubset(frame.columns):
            return pd.DataFrame()
        grouped = frame.groupby(column, dropna=False).agg(
            loan_count=(column, "size"),
            average_loan_amount=("loan_amnt", "mean"),
            average_interest_rate=("int_rate", "mean"),
        )
        return grouped.reset_index().round(2).sort_values("loan_count", ascending=False)
