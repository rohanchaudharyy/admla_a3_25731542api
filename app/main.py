from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import datetime
import numpy as np
from joblib import load
import os

app = FastAPI(
    title="Ethereum Next-Day High Prediction API",
    description="API to predict Ethereum's next-day high price based on a given date input.",
    version="1.0.0"
)

# === Load model ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/LinearRegression_ethereum_nextday_20251030_1650.pkl")
try:
    model_data = load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    features = model_data["features"]
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    features = []

# === Feature builder (from date) ===
def create_features(input_date: datetime.datetime):
    df = pd.DataFrame({
        "year": [input_date.year],
        "month": [input_date.month],
        "day": [input_date.day],
        "weekday": [input_date.weekday()],
        "hour": [input_date.hour]
    })
    # Fill other numerical placeholders (dummy values)
    df["open"] = 0
    df["low"] = 0
    df["close"] = 0
    df["volume"] = 0
    df["marketcap"] = 0
    return df[features] if features else df

# ============================================================
# Root endpoint
# ============================================================
@app.get("/")
def home():
    return {
        "project": "Ethereum Next-Day High Prediction API",
        "description": "Predicts Ethereum's next-day high price using a trained Linear Regression model.",
        "endpoints": {
            "/": "Project overview (this page)",
            "/health/": "Health check endpoint",
            "/predict/ethereum?date=YYYY-MM-DD": "Predicts Ethereum next-day high based on a date input"
        },
        "expected_input": {
            "date": "string in 'YYYY-MM-DD' format"
        },
        "output_format": {
            "input_date": "string",
            "next_day_high_prediction": "float"
        },
        "github_repo": "https://github.com/rohanchaudharyy/admla_a3_25731542api.git"
    }

# ============================================================
# Health check
# ============================================================
@app.get("/health/")
def health_check():
    return {"status": "API is running fine!", "code": 200}

# ============================================================
# Prediction endpoint
# ============================================================
@app.get("/predict/ethereum")
def predict(date: str = Query(..., description="Input date in YYYY-MM-DD format")):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    try:
        input_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    features_df = create_features(input_date)
    scaled = scaler.transform(features_df)
    prediction = model.predict(scaled)[0]
    next_day = (input_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    return {
        "input_date": date,
        "prediction": {
            "next_day": next_day,
            "next_day_high_prediction": round(float(prediction), 2)
        }
    }
