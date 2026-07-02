# 📰 AI News Classifier

> End-to-End ML-проект для классификации новостных текстов с использованием классических методов NLP и Machine Learning.

---

## 📖 О проекте

**AI News Classifier** — это полноценный ML-сервис, демонстрирующий весь цикл разработки модели машинного обучения:

- исследование данных (EDA);
- очистка и предобработка текста;
- обучение нескольких моделей;
- сравнение качества моделей;
- сохранение лучшей модели;
- REST API на FastAPI;
- веб-интерфейс на Streamlit;
- Docker-контейнеризация;
- отслеживание экспериментов через MLflow;
- автоматическое тестирование;
- CI/CD через GitHub Actions.

Проект создавался как часть портфолио ML Engineer и демонстрирует навыки разработки production-ready ML-приложений.

---

# 🚀 Возможности

- классификация новостных текстов;
- автоматическая очистка текста;
- лемматизация;
- удаление стоп-слов;
- TF-IDF векторизация;
- сравнение нескольких моделей;
- сохранение модели;
- REST API;
- Web UI;
- Docker;
- MLflow;
- Unit Tests;
- GitHub Actions.

---

# 🏗 Архитектура проекта

```text
ai-news-classifier
│
├── app
│   ├── api.py
│   └── streamlit_app.py
│
├── configs
│   └── train_config.yaml
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── notebooks
│   └── 01_eda.ipynb
│
├── src
│   └── news_classifier
│       │
│       ├── services
│       │     └── model_service.py
│       │
│       ├── config.py
│       ├── config_loader.py
│       ├── logger.py
│       ├── model_factory.py
│       ├── preprocess.py
│       ├── predict.py
│       ├── train.py
│       └── compare_models.py
│
├── tests
│
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# 📚 Используемый датасет

Используется классический датасет **20 Newsgroups**.

Категории:

- alt.atheism
- comp.graphics
- rec.sport.baseball
- sci.med
- talk.politics.misc

---

# ⚙️ Предобработка текста

Перед обучением каждый документ проходит несколько этапов обработки:

- перевод в нижний регистр;
- удаление HTML и URL;
- удаление цифр;
- удаление знаков препинания;
- удаление стоп-слов;
- лемматизация;
- удаление лишних пробелов.

После этого текст преобразуется в TF-IDF представление.

---

# 🤖 Используемые модели

В проекте реализована возможность обучения нескольких моделей.

Поддерживаются:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

Выбор модели осуществляется через YAML-конфигурацию.

---

# 📊 Сравнение моделей

В проекте реализован отдельный модуль сравнения моделей.

Запуск:

```powershell
$env:PYTHONPATH="src"
python -m news_classifier.compare_models
```

После выполнения результаты автоматически сохраняются в MLflow.

---

# 📈 MLflow

Каждый запуск обучения автоматически сохраняет:

- параметры обучения;
- используемую модель;
- TF-IDF параметры;
- accuracy;
- F1-score;
- обученную модель;
- векторизатор;
- YAML-конфигурацию.

Запуск интерфейса:

```bash
mlflow ui
```

После запуска:

```
http://127.0.0.1:5000
```

---

# 🌐 REST API

Проект предоставляет REST API на FastAPI.

Запуск:

```powershell
$env:PYTHONPATH="src"

uvicorn app.api:app --reload
```

Swagger:

```
http://127.0.0.1:8000/docs
```

---

## POST /predict

Пример запроса

```json
{
    "text": "Doctors discovered a new treatment for heart disease."
}
```

Ответ

```json
{
    "prediction": "sci.med",
    "probabilities":
    {
        "alt.atheism": 0.01,
        "comp.graphics": 0.02,
        "rec.sport.baseball": 0.01,
        "sci.med": 0.91,
        "talk.politics.misc": 0.05
    }
}
```

---

# 💻 Web Interface

Для удобства пользователей реализован интерфейс на Streamlit.

Запуск:

```powershell
streamlit run app/streamlit_app.py
```

---

# 🐳 Docker

Сборка контейнера:

```bash
docker build -t ai-news-classifier .
```

Запуск:

```bash
docker run -p 8000:8000 ai-news-classifier
```

После запуска API доступен по адресу:

```
http://127.0.0.1:8000/docs
```

---

# 🧪 Тестирование

Проект содержит unit-тесты.

Покрываются:

- очистка текста;
- сервис модели;
- REST API.

Запуск:

```bash
pytest
```

---

# ✅ Проверка качества кода

Форматирование:

```bash
black .
```

Проверка:

```bash
ruff check .
```

Автоматическое исправление:

```bash
ruff check . --fix
```

---

# ⚙️ GitHub Actions

После каждого push автоматически выполняются:

- установка зависимостей;
- проверка Ruff;
- проверка Black;
- запуск тестов.

---

# 🚀 Быстрый старт

## 1. Клонировать репозиторий

```bash
git clone https://github.com/<username>/ai-news-classifier.git

cd ai-news-classifier
```

---

## 2. Создать виртуальное окружение

```bash
python -m venv venv
```

---

## 3. Активировать

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux:

```bash
source venv/bin/activate
```

---

## 4. Установить зависимости

```bash
pip install -r requirements.txt
```

---

## 5. Обучить модель

```powershell
$env:PYTHONPATH="src"

python -m news_classifier.train
```

---

## 6. Запустить API

```powershell
$env:PYTHONPATH="src"

uvicorn app.api:app --reload
```

---

## 7. Запустить Streamlit

```powershell
$env:PYTHONPATH="src"

streamlit run app/streamlit_app.py
```

---

# 📦 Используемые технологии

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- NLTK
- FastAPI
- Streamlit
- MLflow
- Docker
- Pytest
- Ruff
- Black
- GitHub Actions

---

# 📌 Что демонстрирует проект

Проект демонстрирует навыки:

- NLP;
- Text Preprocessing;
- Feature Engineering;
- TF-IDF;
- Machine Learning;
- Model Selection;
- MLflow;
- REST API;
- Docker;
- Unit Testing;
- CI/CD;
- Проектирование архитектуры ML-приложений.

---

# 📈 Планы развития

В дальнейшем планируется добавить:

- поддержку BERT и других Transformer-моделей;
- автоматический подбор гиперпараметров (Optuna);
- регистрацию моделей в MLflow Model Registry;
- мониторинг качества модели;
- деплой в облако (Render/Railway);
- поддержку пакетной классификации.

---

# 👨‍💻 Автор

Проект разработан в рамках формирования портфолио ML Engineer и демонстрирует полный цикл разработки прикладного ML-сервиса: от исследования данных до контейнеризации, тестирования и автоматизации.