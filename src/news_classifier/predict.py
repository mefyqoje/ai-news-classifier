from news_classifier.services.model_service import ModelService


def predict_category(text: str):
    service = ModelService()

    return service.predict(text)


if __name__ == "__main__":
    user_text = input("Введите текст новости: ")

    result = predict_category(user_text)

    print(f"Предсказанная категория: {result['prediction']}")
    print("Вероятности:")

    for category, probability in result["probabilities"].items():
        print(f"- {category}: {probability:.4f}")