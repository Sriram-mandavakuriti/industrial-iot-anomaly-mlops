# 🚀 Industrial IoT Anomaly Detection using MLOps

### An End-to-End Machine Learning Operations Project for Predictive Maintenance

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Isolation%20Forest-orange" />
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow" />
  <img src="https://img.shields.io/badge/DVC-Data%20Versioning-purple" />
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker" />
  <img src="https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions" />
</p>

---

## 📌 Project Overview

Industrial machines continuously generate sensor data such as air temperature, process temperature, rotational speed, torque, and tool wear. Unusual changes in these measurements can indicate abnormal operating conditions and potential equipment problems.

This project implements an **Industrial IoT Anomaly Detection System using Machine Learning and MLOps practices** to identify unusual machine operating conditions.

The system uses the **Isolation Forest algorithm** to detect anomalies and integrates experiment tracking, data version control, data drift monitoring, API development, interactive visualization, containerization, and automated testing.

The project demonstrates how machine learning models can be developed, evaluated, deployed as services, and maintained through an organized MLOps workflow.

## 🎯 Project Objectives

* Detect anomalies in industrial machine sensor data.
* Perform data preprocessing and exploratory data analysis.
* Train an Isolation Forest anomaly detection model.
* Track machine learning experiments using MLflow.
* Manage dataset versions using DVC.
* Monitor changes in data distributions through data drift analysis.
* Serve predictions through a FastAPI REST API.
* Visualize results using a Streamlit dashboard.
* Containerize the application using Docker.
* Automate testing and Docker image builds using GitHub Actions.

## 🏗️ System Architecture

```text
                 Industrial IoT Dataset
                          |
                          v
                 Data Loading and EDA
                          |
                          v
                  Data Preprocessing
                          |
                          v
                    Feature Scaling
                          |
                          v
                Isolation Forest Model
                          |
                          v
                   Model Evaluation
                          |
               +----------+----------+
               |                     |
               v                     v
         MLflow Tracking       Saved Model (.pkl)
                                     |
                                     v
                              FastAPI REST API
                                     |
                                     v
                              Streamlit Dashboard

             Supporting MLOps Components
             ----------------------------
             DVC            - Data Version Control
             Data Drift     - Data Distribution Monitoring
             Docker         - Containerization
             GitHub Actions - Automated Testing and Builds
```

## 🧰 Technology Stack

| Category             | Technologies                             |
| -------------------- | ---------------------------------------- |
| Programming Language | Python                                   |
| Dataset              | AI4I 2020 Predictive Maintenance Dataset |
| Data Processing      | Pandas, NumPy                            |
| Data Visualization   | Matplotlib, Seaborn, Plotly              |
| Machine Learning     | Scikit-learn                             |
| Anomaly Detection    | Isolation Forest                         |
| Experiment Tracking  | MLflow                                   |
| Data Version Control | DVC                                      |
| API Development      | FastAPI                                  |
| Dashboard            | Streamlit                                |
| Model Serialization  | Joblib, `.pkl` files                     |
| Containerization     | Docker                                   |
| CI/CD Automation     | GitHub Actions                           |
| Automated Testing    | Pytest                                   |
| Version Control      | Git, GitHub                              |

## 📊 Dataset Description

The project uses the **AI4I 2020 Predictive Maintenance Dataset**, which contains industrial machine operating measurements and failure-related information.

The original dataset contains approximately **10,000 records and 14 columns**.

### Important Features

| Feature                 | Description                                  |
| ----------------------- | -------------------------------------------- |
| Air temperature [K]     | Ambient air temperature                      |
| Process temperature [K] | Machine process temperature                  |
| Rotational speed [rpm]  | Machine rotational speed                     |
| Torque [Nm]             | Applied machine torque                       |
| Tool wear [min]         | Accumulated tool wear                        |
| Machine failure         | Indicates whether a machine failure occurred |

Five numerical sensor features are used by the anomaly detection model.

The dataset was divided into training and testing sets for model development and evaluation.

**Note:** Anomaly detection identifies unusual operating patterns. An anomaly is not necessarily a confirmed machine failure.

## 🤖 Machine Learning Model

### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm that identifies anomalies by isolating observations that differ from the majority of the data.

The model returns one of two prediction classes:

* **NORMAL:** The observation is not classified as an anomaly.
* **ANOMALY:** The observation is classified as an anomaly.

The trained model and fitted feature scaler are serialized and saved for reuse during prediction.

### Saved Model Artifacts

* `models/isolation_forest.pkl` — Trained Isolation Forest model.
* `models/scaler.pkl` — Fitted feature scaler.

Saving these artifacts avoids retraining the model every time the API starts.

## 📈 Model Evaluation

The model evaluation process includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Anomaly detection results

Because machine failures are relatively rare, accuracy alone is not sufficient to assess performance. Precision, recall, and F1-score help evaluate how effectively the model identifies unusual observations.

Model parameters and evaluation metrics can also be tracked across experiments using MLflow.

## ⚙️ MLOps Implementation

### 1. MLflow — Experiment Tracking

MLflow records machine learning experiment information, including parameters and evaluation metrics.

**Benefits:**

* Track experiments.
* Compare model configurations.
* Maintain experiment records.
* Support reproducible model development.

### 2. DVC — Data Version Control

DVC tracks the raw dataset through a `.dvc` metadata file, keeping large data files separate from normal Git version control.

**Benefits:**

* Track dataset versions.
* Maintain dataset references alongside source code.
* Support reproducible data workflows.

Retrieving the dataset from another environment requires the dataset itself or an accessible configured DVC remote.

### 3. Data Drift Monitoring

Data drift monitoring examines changes in data distributions that may affect a deployed machine learning model.

The project includes a drift analysis script and an HTML report.

* **Script:** `src/data_drift.py`
* **Report:** `reports/data_drift_report.html`

Data drift can indicate that incoming data differs from reference data. It does not automatically mean that model performance has decreased.

### 4. FastAPI — Prediction Service

FastAPI exposes the trained anomaly detection model through REST API endpoints.

| Method | Endpoint   | Purpose                                       |
| ------ | ---------- | --------------------------------------------- |
| GET    | `/`        | Check API status                              |
| POST   | `/predict` | Predict whether sensor readings are anomalous |

The prediction endpoint accepts five sensor measurements and returns the predicted class, anomaly score, and sensor values.

Interactive API documentation is available through Swagger UI.

### 5. Streamlit — Interactive Dashboard

The Streamlit dashboard provides a user interface for exploring the application's available information and results.

It makes the project easier to demonstrate without requiring users to interact directly with Python scripts.

### 6. Docker — Containerization

Docker packages the API, model artifacts, and required dependencies into a container image.

This provides a consistent environment for running the prediction service.

### 7. GitHub Actions — CI/CD Workflow

GitHub Actions automates the testing and Docker image build process.

The workflow includes:

1. Checking out the repository.
2. Setting up Python 3.12.
3. Installing dependencies.
4. Running automated API tests.
5. Setting up Docker Buildx.
6. Building the Docker image.

The workflow has successfully completed a test run with all three API tests passing and the Docker image build succeeding.

**Current scope:** The workflow automates testing and image building. Automatic Docker Hub publishing and production deployment are not currently configured.

## 📁 Project Structure

```text
industrial-iot-anomaly-mlops/
│
├── .dvc/
│   └── config
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── app/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv.dvc
│   └── processed/
│
├── frontend/
│   └── dashboard.py
│
├── models/
│   ├── isolation_forest.pkl
│   └── scaler.pkl
│
├── reports/
│   ├── confusion_matrix.png
│   ├── data_drift_report.html
│   ├── experiment_results.csv
│   └── model_predictions.csv
│
├── src/
│   ├── data_drift.py
│   ├── eda.py
│   ├── evaluate_model.py
│   ├── experiment_model.py
│   ├── load_data.py
│   ├── mlflow_experiments.py
│   ├── mlflow_tracking.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── preprocess.py
│   └── train_model.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── .dvcignore
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
```

## 🚀 Installation and Setup

### Prerequisites

Install the following tools:

* Python 3.12
* Git
* Docker Desktop
* DVC, when needed for dataset retrieval and version control

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sriram-mandavakuriti/industrial-iot-anomaly-mlops.git
cd industrial-iot-anomaly-mlops
```

### Step 2: Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

If DVC is configured with an accessible remote:

```bash
dvc pull
```

If no remote is configured, obtain the dataset separately and place it at the location expected by the data-loading scripts.

### Step 5: Run Automated Tests

```bash
python -m pytest -v
```

## 🖥️ Running the Application

### Run the FastAPI Service

From the project root:

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

Check the API status:

http://127.0.0.1:8000/

### Run the Streamlit Dashboard

Open another terminal and run:

```bash
streamlit run frontend/dashboard.py
```

Open the local URL displayed in the terminal.

### Launch MLflow

If MLflow tracking data is stored in the local `mlruns` directory, run:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open:

http://127.0.0.1:5000

If the project uses a different tracking URI, use the same configuration as the experiment scripts.

### Generate the Data Drift Report

```bash
python src/data_drift.py
```

After successful execution, open:

`reports/data_drift_report.html`

### Check DVC Status

```bash
dvc status
```

## 🐳 Running with Docker

Ensure Docker Desktop is running before executing these commands.

### Step 1: Build the Docker Image

```bash
docker build -t industrial-iot-anomaly-api .
```

### Step 2: Start the Container

```bash
docker run -d --name industrial-iot-api -p 8000:8000 industrial-iot-anomaly-api
```

### Step 3: Open the API Documentation

http://127.0.0.1:8000/docs

### Useful Docker Commands

Check running containers:

```bash
docker ps
```

View container logs:

```bash
docker logs industrial-iot-api
```

Stop the container:

```bash
docker stop industrial-iot-api
```

Start the existing container:

```bash
docker start industrial-iot-api
```

Remove a stopped container:

```bash
docker rm industrial-iot-api
```

If a container with the same name already exists, stop and remove it before creating a replacement.

## 🧪 Automated Testing

The project includes three API tests covering:

* API health check.
* Successful prediction response.
* Validation of a missing required parameter.

Run the test suite:

```bash
python -m pytest -v
```

All three tests passed in a previous test run. Run the tests again after making changes to verify the current project state.

## 🔄 Continuous Integration Workflow

The GitHub Actions workflow is located at:

`.github/workflows/ci-cd.yml`

The workflow runs on configured pushes and pull requests targeting the `main` and `master` branches.

To trigger another workflow run:

```bash
git add .
git commit -m "Update project"
git push
```

Check the **Actions** tab in the GitHub repository to view the workflow status and logs.

## 🔮 Future Enhancements

Potential improvements include:

* Automatic Docker Hub publishing.
* Deployment to a cloud hosting platform.
* Scheduled data drift monitoring.
* Model performance monitoring using newly labeled data.
* Automated model retraining and validation.
* API authentication and rate limiting.
* Model registry and controlled model promotion.
* Enhanced dashboard monitoring and alerts.

These are proposed enhancements and are not all implemented in the current version.

## 👨‍💻 Author

**Sriram Mandavakuriti**

AI & Data Science | Machine Learning | Data Analytics | MLOps

* **GitHub:** [Sriram Mandavakuriti](https://github.com/Sriram-mandavakuriti)
* **Project Repository:** [Industrial IoT Anomaly Detection MLOps](https://github.com/Sriram-mandavakuriti/industrial-iot-anomaly-mlops)
* **LinkedIn:** [Sriram Mandavakuriti](https://www.linkedin.com/in/sriram-mandavakuriti-648546357/)

## 📄 License

No license has been specified for this repository. Add an appropriate open-source license if you intend to permit others to reuse or distribute the project.

---

<p align="center">
  <strong>Building reliable machine learning applications through MLOps.</strong>
</p>
