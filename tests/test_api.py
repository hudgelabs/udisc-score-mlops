from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Online"

def test_predict_endpoint():
    # Replace with a course name that exists in your model
    test_data = {
        "CourseName": "Mount Bradford Preserve",
        "Month": 5,
        "HourOfDay": 10,
        "Duration_Min": 120.0
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "predicted_total_score" in response.json()