"""Application configuration and status helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class AppConfig:
    """Configuration for the chatbot application."""

    root: Path = Path(__file__).resolve().parents[2]
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    history_limit: int = 10
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    sample_questions: list[str] = field(default_factory=lambda: [
        "Highest interest rate",
        "Average income",
        "Top states",
        "Is SBI good?",
        "Common complaints about HDFC",
        "Compare SBI and HDFC",
        "Which bank has lower interest rate and better reviews?",
    ])

    @staticmethod
    def _is_placeholder_key(value: str) -> bool:
        normalized = value.strip().lower()
        return normalized in {
            "",
            "dummy",
            "dummy-key",
            "dummy_key",
            "dummy-api-key",
            "your_gemini_api_key",
            "your-google-api-key",
            "your_google_api_key",
        }

    @property
    def gemini_ready(self) -> bool:
        """Return whether a real Gemini key appears to be configured."""
        return not self._is_placeholder_key(self.gemini_api_key)

    def load_project_stats(self) -> dict[str, object]:
        """Load sidebar statistics from local artifacts."""
        loan_frame = pd.read_csv(self.root / "artifacts" / "loan_cleaned.csv", low_memory=False)
        review_frame = pd.read_csv(self.root / "artifacts" / "review_cleaned.csv", low_memory=False)
        metrics_path = self.root / "models" / "training_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        selected_model = metrics.get("selected_model", "Unavailable")
        accuracy = None
        for evaluation in metrics.get("evaluations", []):
            if evaluation.get("model_name") == selected_model:
                accuracy = evaluation.get("accuracy")
                break
        return {
            "loan_records": int(len(loan_frame)),
            "reviews": int(len(review_frame)),
            "banks": int(review_frame["bank"].nunique()),
            "model_accuracy": round(float(accuracy), 4) if accuracy is not None else None,
            "duckdb_status": (self.root / "database" / "loan.duckdb").exists(),
            "gemini_status": self.gemini_ready,
            "selected_model": selected_model,
        }
