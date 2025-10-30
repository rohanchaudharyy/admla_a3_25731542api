from fastapi import FastAPI, HTTPException
import datetime
import numpy as np
import joblib
import os
import pandas as pd

app = FastAPI(
    title="Ethereum Next-Day High Prediction API",
    description="Predict Ethereum next-day high price using a trained Linear Regression model.",
    version="1.0.0"
)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/LinearRegression_ethereum_nextday_20251030_1650.pkl")
try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    features_list = model_data["features"]
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    scaler = None
    features_list = []
    print(f"❌ Error loading model: {e}")

# Function to create features from date
def create_features(input_date: datetime.datetime):
    df = pd.DataFrame({col: [0] for col in features_list})
    df['year'] = input_date.year
    df['month'] = input_date.month
    df['day'] = input_date.day
    df['weekday'] = input_date.weekday()
    df['hour'] = input_date.hour
    return df

# Root endpoint
@app.get("/")
def root():
    return {
        "project": "Ethereum Next-Day High Prediction API",
        "description": "Predicts Ethereum's next-day high price based on date input (YYYY-MM-DD).",
        "endpoints": {
            "/health/": "Health check endpoint",
            "/predict/{token}/{date}": "Predict next-day high of {token} on a specific date"
        },
        "input": {"token": "e.g., ETH", "date": "YYYY-MM-DD"},
        "output": {"next_day_high_prediction": "float"}
    }

# Health check
@app.get("/health/")
def health_check():
    return {"status": "API is running!"}

# Prediction endpoint
@app.get("/predict/{token}/{date}")
def predict(token: str, date: str):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        input_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Create features and scale
    input_features = create_features(input_date)
    input_scaled = scaler.transform(input_features)

    # Make prediction
    try:
        prediction = model.predict(input_scaled)[0]
        return {"token": token, "input_date": date, "next_day_high_prediction": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
