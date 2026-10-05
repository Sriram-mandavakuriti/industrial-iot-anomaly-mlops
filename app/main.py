from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Industrial IoT Anomaly Detection API",
    description="API for detecting machine anomalies using Isolation Forest",
    version="1.0.0"
)

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "isolation_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.get("/")
def home():
    return {
        "message": "Industrial IoT Anomaly Detection API is running",
        "status": "success"
    }


@app.post("/predict")
def predict(
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float
):

    input_data = pd.DataFrame([{
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Anomaly score
    anomaly_score = model.decision_function(input_scaled)[0]

    if prediction == -1:
        result = "ANOMALY"
    else:
        result = "NORMAL"

    return {
        "prediction": result,
        "anomaly_score": round(float(anomaly_score), 4),
        "sensor_data": {
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear
        }
    }