from typing import Dict, Any

from joblib import load

from news_classifier.config import MODEL_PATH, VECTORIZER_PATH
from news_classifier.preprocess import clean_text


def load_artifacts():
    model = load(MODEL_PATH)
    vectorizer = load(VECTORIZER_PATH)

    return model, vectorizer


def predict_category(text: str) -> Dict[str, Any]:
    model, vectorizer = load_artifacts()

    clean = clean_text(text)
    vectorized = vectorizer.transform([clean])

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    return {
        "prediction": prediction,
        "probabilities": class_probabilities,
    }


if __name__ == "__main__":
    user_text = input("Введите текст новости: ")

    result = predict_category(user_text)

    print(f"Предсказанная категория: {result['prediction']}")
    print("Вероятности:")

    for category, probability in result["probabilities"].items():
        print(f"- {category}: {probability:.4f}")