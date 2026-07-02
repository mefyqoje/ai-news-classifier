import streamlit as st

from news_classifier.predict import predict_category

st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="centered",
)

st.title("AI News Classifier")
st.write("Введите текст новости, и модель определит ее категорию.")

text = st.text_area(
    "Текст новости", height=200, placeholder="Вставьте текст новости сюда..."
)

if st.button("Классифицировать"):
    if not text.strip():
        st.warning("Введите текст новости.")
    else:
        result = predict_category(text)

        st.subheader("Предсказанная категория")
        st.success(result["prediction"])

        st.subheader("Вероятности по категориям")

        probabilities = result["probabilities"]

        for category, probability in sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            st.write(f"{category}: {probability:.4f}")
            st.progress(probability)
