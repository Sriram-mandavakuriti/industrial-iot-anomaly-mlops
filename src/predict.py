import joblib



# =========================================
# 1. Load trained model and scaler
# =========================================

MODEL_PATH = "../models/isolation_forest.pkl"
SCALER_PATH = "../models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully.")
print("Scaler loaded successfully.")


# =========================================
# 2. New machine sensor data
# =========================================

air_temperature = 300.0
process_temperature = 310.0
rotational_speed = 1500
torque = 40.0
tool_wear = 100


# =========================================
# 3. Create input
# =========================================

import pandas as pd

input_data = pd.DataFrame([{
    "Air temperature [K]": air_temperature,
    "Process temperature [K]": process_temperature,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])


# =========================================
# 4. Scale input
# =========================================

input_scaled = scaler.transform(input_data)


# =========================================
# 5. Predict
# =========================================

prediction = model.predict(input_scaled)[0]

anomaly_score = model.decision_function(input_scaled)[0]


# =========================================
# 6. Display result
# =========================================

print("\n========================================")
print("MACHINE ANOMALY PREDICTION")
print("========================================")

print("\nInput Sensor Values:")

print("Air Temperature:", air_temperature, "K")
print("Process Temperature:", process_temperature, "K")
print("Rotational Speed:", rotational_speed, "rpm")
print("Torque:", torque, "Nm")
print("Tool Wear:", tool_wear, "min")

print("\nAnomaly Score:", round(anomaly_score, 4))

print("\nPrediction:")

if prediction == -1:
    print("⚠️ ANOMALY DETECTED")
else:
    print("✅ NORMAL MACHINE CONDITION")

print("\n========================================")