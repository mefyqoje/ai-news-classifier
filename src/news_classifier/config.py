from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "news_dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "news_dataset_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2