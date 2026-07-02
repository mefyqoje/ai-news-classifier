import json

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from news_classifier.config import (
    MAX_DF,
    MAX_FEATURES,
    MAX_ITER,
    METADATA_PATH,
    MIN_DF,
    MODEL_PATH,
    MODELS_DIR,
    NGRAM_RANGE,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TEST_SIZE,
    VECTORIZER_PATH,
)
from news_classifier.logger import get_logger
from news_classifier.preprocess import clean_text

logger = get_logger(__name__)


def main() -> None:
    mlflow.set_experiment("AI News Classifier")

    with mlflow.start_run():
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

        logger.info("Логирование параметров в MLflow...")

        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("ngram_range", str(NGRAM_RANGE))
        mlflow.log_param("min_df", MIN_DF)
        mlflow.log_param("max_df", MAX_DF)
        mlflow.log_param("max_iter", MAX_ITER)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1)

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

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(VECTORIZER_PATH))
        mlflow.log_artifact(str(METADATA_PATH))

        logger.info("Готово.")


if __name__ == "__main__":
    main()
