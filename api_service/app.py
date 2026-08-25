import os
from typing import Any, Dict, List, Optional
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://modelos:5000")

app = FastAPI(
    title="API de Clasificación de Salarios (Adult Census Income)",
    description="Microservicio Gateway de FastAPI que valida solicitudes y se comunica con el contenedor de modelos.",
    version="3.0.0",
)


# ==========================================
# ESQUEMAS DE ENTRADA (Pydantic V2)
# ==========================================
class SalaryFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., description="Edad del individuo", json_schema_extra={"example": 39})
    workclass: str = Field(..., description="Clase/sector laboral", json_schema_extra={"example": "State-gov"})
    education_num: int = Field(..., alias="education-num", description="Años de educación", json_schema_extra={"example": 13})
    marital_status: str = Field(..., alias="marital-status", description="Estado civil", json_schema_extra={"example": "Never-married"})
    occupation: str = Field(..., description="Ocupación o puesto de trabajo", json_schema_extra={"example": "Adm-clerical"})
    relationship: str = Field(..., description="Relación familiar", json_schema_extra={"example": "Not-in-family"})
    race: str = Field(..., description="Raza o grupo étnico", json_schema_extra={"example": "White"})
    sex: str = Field(..., description="Género (Male/Female)", json_schema_extra={"example": "Male"})
    capital_gain: int = Field(..., alias="capital-gain", description="Ganancias de capital", json_schema_extra={"example": 2174})
    capital_loss: int = Field(..., alias="capital-loss", description="Pérdidas de capital", json_schema_extra={"example": 0})
    hours_per_week: int = Field(..., alias="hours-per-week", description="Horas semanales trabajadas", json_schema_extra={"example": 40})
    native_country: str = Field(..., alias="native-country", description="País de origen", json_schema_extra={"example": "United-States"})


class PredictionRequest(BaseModel):
    data: List[SalaryFeatures]


# ==========================================
# ENDPOINTS
# ==========================================
@app.get("/", tags=["General"])
async def root():
    """Verifica el estado del gateway FastAPI y consulta el microservicio de modelos."""
    model_info = {"status": "Unreachable"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{MODEL_SERVICE_URL}/")
            if res.status_code == 200:
                model_info = res.json()
    except Exception as e:
        model_info = {"status": "Error connecting to model service", "detail": str(e)}

    return {
        "service": "FastAPI Gateway",
        "status": "Online",
        "model_service_url": MODEL_SERVICE_URL,
        "model_service_status": model_info,
    }


@app.get("/health", tags=["Salud"])
async def health_check():
    """Health check endpoint para orquestadores y monitoreo."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            res = await client.get(f"{MODEL_SERVICE_URL}/health")
            if res.status_code == 200:
                return {"status": "Healthy", "gateway": "OK", "model_service": "OK"}
    except Exception:
        pass
    return {"status": "Degraded", "gateway": "OK", "model_service": "Unavailable"}


@app.post("/predict", tags=["Predicción"])
async def predict(payload: PredictionRequest):
    """Recibe las variables demográficas, las valida y realiza inferencia mediante el servicio de modelos."""
    raw_payload = [item.model_dump(by_alias=True) for item in payload.data]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{MODEL_SERVICE_URL}/predict",
                json={"data": raw_payload},
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error en el servicio de modelos: {response.text}",
            )

        return response.json()

    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No se pudo conectar con el microservicio de modelos en {MODEL_SERVICE_URL}. Verifica que el contenedor 'modelos' esté en ejecución.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Tiempo de espera agotado al consultar el microservicio de modelos.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al procesar la solicitud: {str(e)}",
        )
