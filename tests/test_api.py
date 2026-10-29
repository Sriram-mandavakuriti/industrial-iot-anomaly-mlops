from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Industrial IoT Anomaly Detection API is running" in data["message"]


def test_predict_endpoint():
    response = client.post("/predict", params={
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["NORMAL", "ANOMALY"]
    assert isinstance(data["anomaly_score"], (int, float))
    sensor_data = data["sensor_data"]
    assert sensor_data["air_temperature"] == 298.1
    assert sensor_data["process_temperature"] == 308.6
    assert sensor_data["rotational_speed"] == 1500
    assert sensor_data["torque"] == 40.0
    assert sensor_data["tool_wear"] == 100


def test_predict_missing_parameter():
    response = client.post("/predict", params={
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500,
        "torque": 40.0
    })
    assert response.status_code == 422
