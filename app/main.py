from fastapi import FastAPI, HTTPException
import datetime
import numpy as np
import joblib
import os
import pandas as pd
import requests

app = FastAPI(
    title="Ethereum Next-Day High Prediction API",
    description="Predict Ethereum next-day high price using a trained Linear Regression model and real-time historical data from CoinGecko.",
    version="1.0.0"
)

# =====================================================
# 1. Load Model + Scaler + Feature List
# =====================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/LinearRegression_ethereum_nextday_20251030_1650.pkl")

try:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    scaler = model_data["scaler"]
    features_list = model_data["features"]
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    model = None
    scaler = None
    features_list = []
    print(f"Error loading model: {e}")

# =====================================================
# 2. Fetch Historical Data from CoinGecko
# =====================================================
def fetch_historical_data(token: str = "ethereum", days: int = 30) -> pd.DataFrame:
    """Fetch recent OHLC data for a token from the CoinGecko API."""
    url = f"https://api.coingecko.com/api/v3/coins/{token}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data from CoinGecko API: {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["hour"] = df["date"].dt.hour
    return df

# =====================================================
# 3. Create Features for the Model
# =====================================================
def create_features(input_date: datetime.datetime, token: str = "ethereum") -> pd.DataFrame:
    """Fetch recent data and prepare features for model prediction."""
    df = fetch_historical_data(token, days=30)

    # Find the most recent available day before the input date
    df = df[df["date"].dt.date <= input_date.date()]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No historical data available before {input_date.date()}")

    latest_row = df.iloc[-1]  # Use latest available day’s OHLC
    feature_row = {
        "open": latest_row["open"],
        "high": latest_row["high"],
        "low": latest_row["low"],
        "close": latest_row["close"],
        "volume": 10,         
        "marketcap": 1,   
        "year": latest_row["year"],
        "month": latest_row["month"],
        "day": latest_row["day"],
        "weekday": latest_row["weekday"],
        "hour": latest_row["hour"]
    }

    # Create DataFrame aligned to model’s expected columns
    df_features = pd.DataFrame([feature_row])
    df_features = df_features[[c for c in features_list if c in df_features.columns]]
    return df_features

# =====================================================
# 4. Root Endpoint
# =====================================================
@app.get("/")
def root():
    return {
        "project": "Ethereum Next-Day High Prediction API",
        "description": "Predicts Ethereum's next-day high price based on live historical data fetched from CoinGecko.",
        "endpoints": {
            "/health/": "Health check endpoint",
            "/predict/{token}/{date}": "Predict next-day high of {token} on a specific date"
        },
        "input": {"token": "e.g., ethereum", "date": "YYYY-MM-DD"},
        "output": {"next_day_high_prediction": "float (USD)"}
    }

# =====================================================
# 5. Health Check
# =====================================================
@app.get("/health/")
def health_check():
    return {"status": "API is running and model loaded" if model else "Model not loaded"}

# =====================================================
# 6. Prediction Endpoint
# =====================================================
@app.get("/predict/{token}/{date}")
def predict(token: str, date: str):
    """Predict next-day high price for a cryptocurrency."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        input_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    try:
        input_features = create_features(input_date, token)
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]

        next_day = input_date + datetime.timedelta(days=1)

        return {
            "token": token,
            "input_date": date,
            "predicted_date": next_day.strftime("%Y-%m-%d"),
            "next_day_high_prediction_usd": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
