from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from news_classifier.services.model_service import ModelService

app = FastAPI(
    title="AI News Classifier API",
    description="API для классификации новостных текстов",
    version="1.0.0",
)

model_service = ModelService()


class PredictionRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "AI News Classifier API запущен"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Текст новости не должен быть пустым.",
        )

    result = model_service.predict(request.text)

    return result
