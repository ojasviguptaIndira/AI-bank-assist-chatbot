"""Sentiment prediction service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sentiment.preprocessing import TextPreprocessor


@dataclass
class SentimentPredictor:
    """Load trained assets and predict sentiment for free text."""

    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self) -> None:
        self.models_dir = self.root / "models"
        self.model = joblib.load(self.models_dir / "sentiment_model.pkl")
        self.vectorizer = joblib.load(self.models_dir / "tfidf_vectorizer.pkl")
        self.label_encoder: LabelEncoder = joblib.load(self.models_dir / "label_encoder.pkl")
        self.preprocessor = TextPreprocessor()

    def predict(self, text: str) -> dict[str, object]:
        """Predict sentiment, confidence, and class probabilities."""
        processed = self.preprocessor.preprocess(text)
        vector = self.vectorizer.transform([processed])
        encoded_prediction = int(self.model.predict(vector)[0])
        probabilities = self._probabilities(vector)[0]
        labels = self.label_encoder.inverse_transform(np.arange(len(probabilities)))
        probability_map = {label: round(float(score), 4) for label, score in zip(labels, probabilities)}
        confidence = round(float(probabilities[encoded_prediction]), 4)
        sentiment = self.label_encoder.inverse_transform([encoded_prediction])[0]
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probability_map,
        }

    def _probabilities(self, vector) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(vector)
        scores = self.model.decision_function(vector)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
