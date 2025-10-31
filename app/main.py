from fastapi import FastAPI, HTTPException
import datetime
import numpy as np
import joblib
import os
import pandas as pd
import requests

app = FastAPI(
    title="Ethereum Next-Day High Prediction API",
    description="Predict Ethereum next-day high price using a trained Linear Regression model and Kraken historical data.",
    version="1.1.0"
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
# 2. Fetch Historical Data from Kraken
# =====================================================
def fetch_historical_data(token: str = "ETH", days: int = 30) -> pd.DataFrame:
    """
    Fetch historical OHLC daily data from Kraken for ETH-USD.
    """
    pair = "XETHZUSD"
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440&since={int((datetime.datetime.now() - datetime.timedelta(days=days)).timestamp())}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {response.status_code}")
    
    data = response.json()
    if data.get("error"):
        raise HTTPException(status_code=500, detail=f"Kraken API error: {data['error']}")
    
    ohlc = data["result"][pair]
    df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df[["open","high","low","close","vwap","volume"]] = df[["open","high","low","close","vwap","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["weekday"] = df["time"].dt.weekday
    df["hour"] = 0  # daily data has no hour
    return df

# =====================================================
# 3. Create Features for the Model
# =====================================================
def create_features(input_date: datetime.datetime, token: str = "ETH") -> pd.DataFrame:
    df = fetch_historical_data(token, days=30)
    df = df[df["time"].dt.date <= input_date.date()]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No historical data available before {input_date.date()}")
    
    latest_row = df.iloc[-1]
    feature_row = {
        "open": latest_row["open"],
        "high": latest_row["high"],
        "low": latest_row["low"],
        "close": latest_row["close"],
        "volume": latest_row["volume"],  
        "marketcap": 1,  # placeholder
        "year": latest_row["year"],
        "month": latest_row["month"],
        "day": latest_row["day"],
        "weekday": latest_row["weekday"],
        "hour": latest_row["hour"]
    }
    
    df_features = pd.DataFrame([feature_row])
    for col in features_list:
        if col not in df_features.columns:
            df_features[col] = 0
    df_features = df_features[features_list]
    return df_features

# =====================================================
# 4. Root Endpoint
# =====================================================
@app.get("/")
def root():
    return {
        "project": "Ethereum Next-Day High Prediction API",
        "description": "Predicts Ethereum's next-day high price based on Kraken historical data.",
        "endpoints": {
            "/health/": "Health check endpoint",
            "/predict/{token}/{date}": "Predict next-day high of {token} on a specific date"
        },
        "input": {"token": "e.g., ETH", "date": "YYYY-MM-DD"},
        "output": {"next_day_high_prediction_usd": "float"}
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
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
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
