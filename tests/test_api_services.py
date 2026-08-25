import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

PROJ_ROOT = Path(__file__).resolve().parents[1]
model_service_dir = str(PROJ_ROOT / "model_service")
api_service_dir = str(PROJ_ROOT / "api_service")

if model_service_dir not in sys.path:
    sys.path.append(model_service_dir)
if api_service_dir not in sys.path:
    sys.path.append(api_service_dir)

import importlib.util

def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_module_from_path("model_service_app", PROJ_ROOT / "model_service" / "app.py")
gateway_module = load_module_from_path("api_service_app", PROJ_ROOT / "api_service" / "app.py")

model_app = model_module.app
gateway_app = gateway_module.app


@pytest.fixture
def sample_payload():
    return {
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
                "native-country": "United-States",
            }
        ]
    }


# =====================================================
# PRUEBAS DEL SERVICIO DE MODELOS (model_service)
# =====================================================
def test_model_service_root():
    with TestClient(model_app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Model Service"
        assert "metadata" in data


def test_model_service_health():
    with TestClient(model_app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "Healthy"


def test_model_service_predict(sample_payload):
    with TestClient(model_app) as client:
        response = client.post("/predict", json=sample_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["diagnosis"] in ["<=50K", ">50K"]
        assert "confidence_score" in data["results"][0]


# =====================================================
# PRUEBAS DEL GATEWAY FASTAPI (api_service)
# =====================================================
def test_gateway_root():
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json = MagicMock(return_value={"service": "Model Service", "status": "Online"})

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_res)):
        with TestClient(gateway_app) as client:
            response = client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["service"] == "FastAPI Gateway"
            assert data["status"] == "Online"


def test_gateway_validation_error():
    """Verifica que el Gateway de FastAPI valide el esquema Pydantic antes de reenviar."""
    invalid_payload = {"data": [{"age": "no-es-un-numero"}]}
    with TestClient(gateway_app) as client:
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422  # Unprocessable Entity de Pydantic/FastAPI


def test_gateway_forwarding(sample_payload):
    """Verifica que el Gateway FastAPI serializa y procesa respuestas del contenedor de modelos."""
    mock_model_response = {
        "model_metadata": {"name": "Clasificación de Salarios", "version": "1.0.0"},
        "total_predictions": 1,
        "results": [{"index": 0, "prediction_code": 0, "diagnosis": "<=50K", "confidence_score": 95.5}],
    }
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.text = ""
    mock_res.json = MagicMock(return_value=mock_model_response)

    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_res)):
        with TestClient(gateway_app) as client:
            response = client.post("/predict", json=sample_payload)
            assert response.status_code == 200
            data = response.json()
            assert data["total_predictions"] == 1
            assert data["results"][0]["diagnosis"] == "<=50K"
