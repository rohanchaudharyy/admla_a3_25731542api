from fastapi import FastAPI, HTTPException
import datetime
import numpy as np
import joblib
import os
import pandas as pd
import requests

app = FastAPI(
    title="Ethereum Next-Day High Prediction API",
    description="Predict Ethereum next-day high price using a trained Linear Regression model and Kraken OHLC data.",
    version="1.0.1"
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
def fetch_historical_data(token="ETH", days=30) -> pd.DataFrame:
    """
    Fetch historical OHLC data from Kraken public API.
    """
    pair = f"X{token}ZUSD"
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {response.status_code}")

    data = response.json()
    if "error" in data and len(data["error"]) > 0:
        raise HTTPException(status_code=500, detail=f"Kraken API error: {data['error']}")

    ohlc = data["result"][pair]
    df = pd.DataFrame(ohlc, columns=[
        "time","open","high","low","close","vwap","volume","count"
    ])
    df[["open","high","low","close","vwap","volume"]] = df[["open","high","low","close","vwap","volume"]].astype(float)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["hour"] = 0  # daily data has no hour
    df = df.tail(days)
    return df

# =====================================================
# 3. Prepare Features for Prediction
# =====================================================
def create_features(input_date: datetime.datetime, token="ETH") -> pd.DataFrame:
    df = fetch_historical_data(token, days=30)
    df = df[df["date"].dt.date <= input_date.date()]
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
            df_features[col] = 0  # fill missing columns
    df_features = df_features[features_list]  # reorder
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
        "input": {"token": "ETH", "date": "YYYY-MM-DD"},
        "output": {"next_day_high_prediction_usd": "float"}
    }

# =====================================================
# 5. Health Check
# =====================================================
@app.get("/health/")
def health_check():
    return {"status": "API running", "model_loaded": model is not None}

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

# =====================================================
# 7. Run Uvicorn for Render
# =====================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
