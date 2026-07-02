from fastapi import FastAPI
from pydantic import BaseModel

from news_classifier.predict import predict_category


app = FastAPI(
    title="AI News Classifier API",
    description="API для классификации новостных текстов",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "message": "AI News Classifier API запущен"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    result = predict_category(request.text)

    return result