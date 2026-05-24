import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Загрузка модели и scaler (один раз при старте) ---
model = joblib.load('fraud_model.joblib')
scaler = joblib.load('scaler.joblib')

# --- 2. Создаём приложение ---
app = FastAPI(title="Fraud Alert Prioritization API")

# --- 3. Форма заказа: что шлёт клиент (7 базовых полей) ---
class Transaction(BaseModel):
    step: int
    type: int            # 1 = TRANSFER, 0 = CASH_OUT
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

# --- 4. Пинг: жив ли сервис ---
@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud API is running"}

# --- 5. Главная ручка: предсказание ---
@app.post("/predict")
def predict(tx: Transaction):
    # 5.1 Досчитываем 4 инженерные фичи (как в разделе 3.4)
    errorBalanceOrig = tx.oldbalanceOrg - tx.amount - tx.newbalanceOrig
    errorBalanceDest = tx.newbalanceDest - tx.oldbalanceDest - tx.amount
    isZeroBalanceOrig = 1 if tx.newbalanceOrig == 0 else 0
    isZeroBalanceDest = 1 if (tx.oldbalanceDest == 0 and tx.newbalanceDest == 0) else 0

    # 5.2 Собираем 11 признаков В ТОМ ЖЕ ПОРЯДКЕ, что при обучении
    features = np.array([[
        tx.step,
        tx.type,
        tx.amount,
        tx.oldbalanceOrg,
        tx.newbalanceOrig,
        tx.oldbalanceDest,
        tx.newbalanceDest,
        errorBalanceOrig,
        errorBalanceDest,
        isZeroBalanceOrig,
        isZeroBalanceDest
    ]])

    # 5.3 Масштабируем тем же scaler, затем предсказываем
    features_scaled = scaler.transform(features)
    proba = float(model.predict_proba(features_scaled)[0][1])
    prediction = int(proba >= 0.5)

    # 5.4 Возвращаем ответ
    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": prediction
    }
