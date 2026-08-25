"""
test_api.py
-----------
Pruebas unitarias e integración para la API de FastAPI (app.py).
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Online"
    assert "model_name" in data
    assert "production_version" in data


def test_predict_endpoint():
    payload = {
        "data": [
            {
                "age": 39,
                "workclass": "State-gov",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 1
    assert len(data["results"]) == 1
    res = data["results"][0]
    assert res["prediction_code"] in [0, 1]
    assert res["diagnosis"] in ["<=50K", ">50K"]
    assert "confidence_score" in res
    assert "probabilities_detail" in res
