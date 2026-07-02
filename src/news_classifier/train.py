import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from news_classifier.config import (
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    VECTORIZER_PATH,
)
from news_classifier.config_loader import load_config
from news_classifier.logger import get_logger
from news_classifier.preprocess import clean_text

logger = get_logger(__name__)


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    config = load_config(config_path)

    data_config = config["data"]
    vectorizer_config = config["vectorizer"]
    model_config = config["model"]
    mlflow_config = config["mlflow"]

    mlflow.set_experiment(mlflow_config["experiment_name"])

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
            test_size=data_config["test_size"],
            random_state=data_config["random_state"],
            stratify=y,
        )

        logger.info("TF-IDF векторизация текста...")

        ngram_range = tuple(vectorizer_config["ngram_range"])

        vectorizer = TfidfVectorizer(
            max_features=vectorizer_config["max_features"],
            ngram_range=ngram_range,
            min_df=vectorizer_config["min_df"],
            max_df=vectorizer_config["max_df"],
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        logger.info("Обучение модели...")

        if model_config["name"] == "logistic_regression":
            model = LogisticRegression(max_iter=model_config["max_iter"])
        else:
            raise ValueError(f"Неизвестная модель: {model_config['name']}")

        model.fit(X_train_tfidf, y_train)

        logger.info("Оценка качества модели...")

        y_pred = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 weighted: {f1:.4f}")

        print(classification_report(y_test, y_pred))

        logger.info("Логирование эксперимента в MLflow...")

        mlflow.log_params(
            {
                "model": model_config["name"],
                "vectorizer": "TF-IDF",
                "max_features": vectorizer_config["max_features"],
                "ngram_range": str(ngram_range),
                "min_df": vectorizer_config["min_df"],
                "max_df": vectorizer_config["max_df"],
                "max_iter": model_config["max_iter"],
                "test_size": data_config["test_size"],
                "random_state": data_config["random_state"],
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1_weighted": f1,
            }
        )

        logger.info("Сохранение модели и векторизатора...")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        dump(model, MODEL_PATH)
        dump(vectorizer, VECTORIZER_PATH)

        metadata = {
            "model": model_config["name"],
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
        mlflow.log_artifact(str(config_path))

        logger.info("Готово.")


if __name__ == "__main__":
    main()