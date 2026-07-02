from typing import Any

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

    def predict(self, text: str) -> dict[str, Any]:
        clean = clean_text(text)

        vectorized = self.vectorizer.transform([clean])

        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]

        class_probabilities = {
            class_name: float(probability)
            for class_name, probability in zip(
                self.model.classes_,
                probabilities,
            )
        }

        return {
            "prediction": prediction,
            "probabilities": class_probabilities,
        }