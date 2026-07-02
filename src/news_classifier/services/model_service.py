from typing import Any

import numpy as np
from joblib import load

from news_classifier.config import MODEL_PATH, VECTORIZER_PATH
from news_classifier.preprocess import clean_text


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.vectorizer = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Файл модели не найден. Сначала выполните команду:\n"
                "python -m news_classifier.train"
            )

        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Файл векторизатора не найден. Сначала выполните команду:\n"
                "python -m news_classifier.train"
            )

        self.model = load(MODEL_PATH)
        self.vectorizer = load(VECTORIZER_PATH)

    def _get_probabilities(self, vectorized_text) -> dict[str, float]:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vectorized_text)[0]

            return {
                class_name: float(probability)
                for class_name, probability in zip(
                    self.model.classes_,
                    probabilities,
                )
            }

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(vectorized_text)

            if scores.ndim == 1:
                scores = np.array([scores])

            scores = scores[0]
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()

            return {
                class_name: float(probability)
                for class_name, probability in zip(
                    self.model.classes_,
                    probabilities,
                )
            }

        return {
            class_name: 0.0
            for class_name in self.model.classes_
        }

    def predict(self, text: str) -> dict[str, Any]:
        clean = clean_text(text)

        vectorized = self.vectorizer.transform([clean])

        prediction = self.model.predict(vectorized)[0]
        probabilities = self._get_probabilities(vectorized)

        return {
            "prediction": prediction,
            "probabilities": probabilities,
        }