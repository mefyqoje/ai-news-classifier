from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def create_model(model_config: dict):
    model_name = model_config["name"]

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=model_config.get("max_iter", 1000)
        )

    if model_name == "naive_bayes":
        return MultinomialNB(
            alpha=model_config.get("alpha", 1.0)
        )

    if model_name == "linear_svm":
        return LinearSVC(
            max_iter=model_config.get("max_iter", 1000)
        )

    raise ValueError(f"Неизвестная модель: {model_name}")