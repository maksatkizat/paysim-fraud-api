# PaySim Fraud Detection API

Приоритизация алертов мобильного фрода для compliance-команд.
Модель: Random Forest на датасете PaySim (Kaggle).

## Запуск через Docker
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api

Swagger UI: http://127.0.0.1:8000/docs

## Стек
FastAPI, scikit-learn 1.6.1, Docker
