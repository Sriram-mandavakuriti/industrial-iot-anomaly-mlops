import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


# =========================================
# 1. Load processed dataset
# =========================================

DATA_PATH = "../data/processed/processed_data.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


# =========================================
# 2. Select features and target
# =========================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]


# =========================================
# 3. Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================
# 4. Feature Scaling
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================
# 5. Contamination experiments
# =========================================

contamination_values = [
    0.02,
    0.03,
    0.034,
    0.04,
    0.05,
    0.07,
    0.10
]


print("\n========================================")
print("ISOLATION FOREST EXPERIMENT")
print("========================================")

print(
    f"{'Contamination':<15}"
    f"{'Precision':<15}"
    f"{'Recall':<15}"
    f"{'F1-Score':<15}"
    f"{'Anomalies':<15}"
)
experiment_results = []

print("-" * 75)


# =========================================
# 6. Train and evaluate each model
# =========================================

for contamination in contamination_values:

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled)

    predictions = model.predict(X_test_scaled)

    # Convert:
    # Normal  = 0
    # Anomaly = 1

    anomaly_predictions = (
        predictions == -1
    ).astype(int)

    precision = precision_score(
        y_test,
        anomaly_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        anomaly_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        anomaly_predictions,
        zero_division=0
    )

    anomalies = anomaly_predictions.sum()
    experiment_results.append({
        "contamination": contamination,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "anomalies_detected": anomalies
    })

    print(
        f"{contamination:<15.3f}"
        f"{precision:<15.3f}"
        f"{recall:<15.3f}"
        f"{f1:<15.3f}"
        f"{anomalies:<15}"
    )

 # =========================================
    # 7. Save experiment results
    # =========================================
os.makedirs("../reports", exist_ok=True)
results_df = pd.DataFrame(experiment_results)

results_df.to_csv(
    "../reports/experiment_results.csv",
    index=False
)

print("\nExperiment results saved to:")
print("../reports/experiment_results.csv")
print("\n========================================")
print("EXPERIMENT COMPLETED")
print("========================================")