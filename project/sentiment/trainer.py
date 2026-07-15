"""Train and export the sentiment classification pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sentiment.evaluation import EvaluationResult, Evaluator
from sentiment.preprocessing import TextPreprocessor


LOGGER = logging.getLogger(__name__)


@dataclass
class SentimentTrainer:
    """Train, compare, evaluate, and persist sentiment models."""

    root: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self) -> None:
        self.artifacts_dir = self.root / "artifacts"
        self.models_dir = self.root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_path = self.artifacts_dir / "sentiment_training.csv"
        self.preprocessor = TextPreprocessor()

    def train(self) -> dict[str, object]:
        """Train and export the best-performing sentiment model."""
        data = pd.read_csv(self.training_path)
        data = data.dropna(subset=["review", "sentiment"]).copy()
        data["processed_text"] = self.preprocessor.batch_preprocess(data["review"].astype(str).tolist())
        data = data[data["processed_text"].str.len() > 0].copy()

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data["sentiment"])
        x_train, x_test, y_train, y_test = train_test_split(
            data["processed_text"],
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        x_train_vectorized = vectorizer.fit_transform(x_train)
        x_test_vectorized = vectorizer.transform(x_test)

        candidates = {
            "SGDClassifier": SGDClassifier(loss="log_loss", random_state=42),
            "LinearSVC": LinearSVC(random_state=42),
        }
        results: list[tuple[EvaluationResult, object]] = []
        for name, model in candidates.items():
            model.fit(x_train_vectorized, y_train)
            predictions = model.predict(x_test_vectorized)
            results.append((Evaluator.evaluate(name, y_test, predictions), model))

        best_result, best_model = max(results, key=lambda item: (item[0].f1, item[0].accuracy))
        self._persist(best_model, vectorizer, label_encoder, best_result, results)
        LOGGER.info(
            "Selected %s with accuracy %.4f and F1 %.4f",
            best_result.model_name,
            best_result.accuracy,
            best_result.f1,
        )
        return {
            "best_model_name": best_result.model_name,
            "metrics": best_result,
        }

    def _persist(
        self,
        model: object,
        vectorizer: TfidfVectorizer,
        label_encoder: LabelEncoder,
        best_result: EvaluationResult,
        all_results: list[tuple[EvaluationResult, object]],
    ) -> None:
        joblib.dump(model, self.models_dir / "sentiment_model.pkl")
        joblib.dump(vectorizer, self.models_dir / "tfidf_vectorizer.pkl")
        joblib.dump(label_encoder, self.models_dir / "label_encoder.pkl")

        payload = {
            "selected_model": best_result.model_name,
            "evaluations": [
                {
                    "model_name": result.model_name,
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1": result.f1,
                    "confusion_matrix": result.confusion_matrix,
                    "classification_report": result.classification_report,
                }
                for result, _ in all_results
            ],
        }
        metrics_path = self.models_dir / "training_metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """Executable training entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    SentimentTrainer().train()


if __name__ == "__main__":
    main()
