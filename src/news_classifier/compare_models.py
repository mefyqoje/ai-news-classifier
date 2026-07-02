import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from news_classifier.config import PROJECT_ROOT, RAW_DATA_PATH
from news_classifier.config_loader import load_config
from news_classifier.logger import get_logger
from news_classifier.model_factory import create_model
from news_classifier.preprocess import clean_text

logger = get_logger(__name__)


MODELS_TO_COMPARE = [
    {"name": "logistic_regression", "max_iter": 1000},
    {"name": "naive_bayes", "alpha": 1.0},
    {"name": "linear_svm", "max_iter": 1000},
]


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "train_config.yaml"
    config = load_config(config_path)

    data_config = config["data"]
    vectorizer_config = config["vectorizer"]
    mlflow_config = config["mlflow"]

    mlflow.set_experiment(mlflow_config["experiment_name"] + " - Model Comparison")

    logger.info("Загрузка данных...")
    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Очистка текста...")
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["category"]

    logger.info("Разделение данных...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config["test_size"],
        random_state=data_config["random_state"],
        stratify=y,
    )

    logger.info("TF-IDF векторизация...")
    ngram_range = tuple(vectorizer_config["ngram_range"])

    vectorizer = TfidfVectorizer(
        max_features=vectorizer_config["max_features"],
        ngram_range=ngram_range,
        min_df=vectorizer_config["min_df"],
        max_df=vectorizer_config["max_df"],
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = []

    for model_config in MODELS_TO_COMPARE:
        model_name = model_config["name"]

        logger.info(f"Обучение модели: {model_name}")

        with mlflow.start_run(run_name=model_name):
            model = create_model(model_config)
            model.fit(X_train_tfidf, y_train)

            y_pred = model.predict(X_test_tfidf)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            mlflow.log_params(model_config)
            mlflow.log_param("vectorizer", "TF-IDF")
            mlflow.log_param("max_features", vectorizer_config["max_features"])
            mlflow.log_param("ngram_range", str(ngram_range))
            mlflow.log_param("min_df", vectorizer_config["min_df"])
            mlflow.log_param("max_df", vectorizer_config["max_df"])

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_weighted", f1)

            results.append(
                {
                    "model": model_name,
                    "accuracy": accuracy,
                    "f1_weighted": f1,
                }
            )

            logger.info(
                f"{model_name}: accuracy={accuracy:.4f}, f1={f1:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(
        "f1_weighted",
        ascending=False,
    )

    print(results_df)


if __name__ == "__main__":
    main()