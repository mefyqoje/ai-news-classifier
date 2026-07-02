from news_classifier.services.model_service import ModelService


def test_model_service_prediction():
    service = ModelService()

    result = service.predict(
        "Doctors discovered a new treatment for heart disease."
    )

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], dict)