"""Evaluation helpers for sentiment classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    """Structured model evaluation output."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, dict[str, float] | float]


class Evaluator:
    """Compute sentiment model evaluation metrics."""

    @staticmethod
    def evaluate(
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> EvaluationResult:
        return EvaluationResult(
            model_name=model_name,
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
            classification_report=classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        )
