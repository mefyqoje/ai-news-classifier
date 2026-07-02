from news_classifier.preprocess import clean_text


def test_clean_text_returns_string():
    text = "Hello!!! This is TEST."

    result = clean_text(text)

    assert isinstance(result, str)


def test_clean_text_removes_punctuation():
    text = "Hello!!! World???"

    result = clean_text(text)

    assert result == "hello world"


def test_clean_text_removes_stopwords():
    text = "This is a simple test"

    result = clean_text(text)

    assert "is" not in result
    assert "a" not in result