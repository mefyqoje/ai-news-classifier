FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords punkt wordnet omw-1.4

COPY data ./data
COPY src ./src
COPY app ./app

ENV PYTHONPATH=/app/src

RUN python -m news_classifier.train

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]