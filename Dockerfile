# 1. Базовый образ: лёгкий Python 3.12
FROM python:3.12-slim

# 2. Рабочая папка внутри контейнера
WORKDIR /app

# 3. Сначала копируем только requirements (для кэширования)
COPY requirements.txt .

# 4. Ставим библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем код и модель в контейнер
COPY app.py .
COPY fraud_model.joblib .
COPY scaler.joblib .

# 6. Открываем порт 8000
EXPOSE 8000

# 7. Команда запуска сервиса
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
