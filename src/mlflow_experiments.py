import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================================
# 1. Project paths
# =========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "processed_data.csv"
)

MLFLOW_DB = os.path.join(
    PROJECT_ROOT,
    "mlflow.db"
)


# =========================================
# 2. Configure MLflow
# =========================================

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.replace(os.sep, '/')}"
)

mlflow.set_experiment(
    "Industrial-IoT-Anomaly-Detection"
)


# =========================================
# 3. Load dataset
# =========================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# =========================================
# 4. Features and target
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
# 5. Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================
# 6. Feature Scaling
# =========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================
# 7. Experiment configurations
# =========================================

contamination_values = [
    0.020,
    0.030,
    0.034,
    0.040,
    0.050,
    0.070,
    0.100
]


# =========================================
# 8. Run experiments
# =========================================

print("\n========================================")
print("MLFLOW EXPERIMENT TRACKING")
print("========================================")

for contamination in contamination_values:

    print(
        f"\nRunning contamination = {contamination}"
    )

    with mlflow.start_run(
        run_name=f"IsolationForest_{contamination}"
    ):

        # ---------------------------------
        # Model
        # ---------------------------------

        n_estimators = 200
        random_state = 42

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

        # ---------------------------------
        # Train
        # ---------------------------------

        model.fit(X_train_scaled)

        # ---------------------------------
        # Predict
        # ---------------------------------

        predictions = model.predict(
            X_test_scaled
        )

        anomaly_predictions = (
            predictions == -1
        ).astype(int)

        # ---------------------------------
        # Metrics
        # ---------------------------------

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

        # ---------------------------------
        # Log parameters
        # ---------------------------------

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
            "feature_count",
            len(features)
        )

        # ---------------------------------
        # Log metrics
        # ---------------------------------

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

        # ---------------------------------
        # Log model
        # ---------------------------------

        mlflow.sklearn.log_model(
            model,
            name="isolation_forest_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ---------------------------------
        # Display result
        # ---------------------------------

        print(
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f} | "
            f"Anomalies: {anomalies}"
        )


print("\n========================================")
print("ALL MLFLOW EXPERIMENTS COMPLETED")
print("========================================")