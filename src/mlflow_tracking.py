import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
mlflow.set_tracking_uri("sqlite:///../mlflow.db")

# =========================================
# 1. Load dataset
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
# 5. MLflow Configuration
# =========================================

mlflow.set_experiment("Industrial-IoT-Anomaly-Detection")

print("\nMLflow experiment:")
print("Industrial-IoT-Anomaly-Detection")


# =========================================
# 6. Model parameters
# =========================================

n_estimators = 200
contamination = 0.10
random_state = 42


# =========================================
# 7. Start MLflow Run
# =========================================

with mlflow.start_run():

    print("\nStarting MLflow run...")

    # -------------------------------------
    # Create model
    # -------------------------------------

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )

    # -------------------------------------
    # Train model
    # -------------------------------------

    model.fit(X_train_scaled)

    # -------------------------------------
    # Predictions
    # -------------------------------------

    predictions = model.predict(X_test_scaled)

    anomaly_predictions = (
        predictions == -1
    ).astype(int)

    # -------------------------------------
    # Calculate metrics
    # -------------------------------------

    accuracy = accuracy_score(
        y_test,
        anomaly_predictions
    )

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

    anomalies = int(
        anomaly_predictions.sum()
    )

    # -------------------------------------
    # Log parameters
    # -------------------------------------

    mlflow.log_param(
        "model",
        "Isolation Forest"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "contamination",
        contamination
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "features",
        ", ".join(features)
    )

    # -------------------------------------
    # Log metrics
    # -------------------------------------

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "anomalies_detected",
        anomalies
    )

    # -------------------------------------
    # Log model
    # -------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="isolation_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    # -------------------------------------
    # Display results
    # -------------------------------------

    print("\n========================================")
    print("MLFLOW RUN COMPLETED")
    print("========================================")

    print("\nParameters:")
    print("Model:", "Isolation Forest")
    print("Estimators:", n_estimators)
    print("Contamination:", contamination)

    print("\nMetrics:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Anomalies:", anomalies)

    print("\nMLflow run ID:")
    print(mlflow.active_run().info.run_id)

    print("\nMLflow tracking completed.")