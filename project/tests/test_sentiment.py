"""Tests for the sentiment engine."""

from __future__ import annotations

import unittest
from pathlib import Path

from sentiment.predictor import SentimentPredictor
from sentiment.trainer import SentimentTrainer


class SentimentEngineTests(unittest.TestCase):
    """Validate training outputs and prediction structure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parent.parent
        cls.trainer = SentimentTrainer(root=cls.root)
        cls.result = cls.trainer.train()
        cls.predictor = SentimentPredictor(root=cls.root)

    def test_model_artifacts_created(self) -> None:
        models_dir = self.root / "models"
        self.assertTrue((models_dir / "sentiment_model.pkl").exists())
        self.assertTrue((models_dir / "tfidf_vectorizer.pkl").exists())
        self.assertTrue((models_dir / "label_encoder.pkl").exists())

    def test_predictor_response_shape(self) -> None:
        result = self.predictor.predict("The bank service was smooth and very helpful.")
        self.assertIn("sentiment", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertGreater(len(result["probabilities"]), 0)

    def test_training_selects_supported_model(self) -> None:
        self.assertIn(self.result["best_model_name"], {"SGDClassifier", "LinearSVC"})


if __name__ == "__main__":
    unittest.main()
