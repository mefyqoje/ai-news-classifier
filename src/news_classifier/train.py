import json

import pandas as pd

from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from news_classifier.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODELS_DIR,
    MODEL_PATH,
    VECTORIZER_PATH,
    METADATA_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    MAX_FEATURES,
    NGRAM_RANGE,
    MIN_DF,
    MAX_DF,
    MAX_ITER,
)
from news_classifier.logger import get_logger
from news_classifier.preprocess import clean_text


logger = get_logger(__name__)


def main() -> None:
    logger.info("Загрузка данных...")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Очистка текста...")

    df["clean_text"] = df["text"].apply(clean_text)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    X = df["clean_text"]
    y = df["category"]

    logger.info("Разделение данных на train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("TF-IDF векторизация текста...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    logger.info("Обучение модели Logistic Regression...")

    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train_tfidf, y_train)

    logger.info("Оценка качества модели...")

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 weighted: {f1:.4f}")

    print(classification_report(y_test, y_pred))

    logger.info("Сохранение модели и векторизатора...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)

    metadata = {
        "model": "Logistic Regression",
        "vectorizer": "TF-IDF",
        "accuracy": accuracy,
        "f1_weighted": f1,
        "dataset": "20 Newsgroups",
        "classes": list(model.classes_),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    logger.info("Готово.")


if __name__ == "__main__":
    main()