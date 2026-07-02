from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "AI News Classifier API запущен"}


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"text": "Doctors discovered a new treatment for heart disease."},
    )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probabilities" in response.json()


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 400