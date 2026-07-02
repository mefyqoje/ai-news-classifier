import json

import pandas as pd

from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
)
from news_classifier.preprocess import clean_text


def main():
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    X = df["clean_text"]
    y = df["category"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 weighted: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    print("Saving artifacts...")
    MODELS_DIR.mkdir(exist_ok=True)

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

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()