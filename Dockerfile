FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords punkt wordnet omw-1.4

RUN python -m spacy download en_core_web_sm

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]