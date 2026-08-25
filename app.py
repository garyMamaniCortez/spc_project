from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Importamos las funciones centralizadas de MLOps
from src.api.model_loader import (
    MODEL_NAME,
    get_model_metadata,
    load_feature_columns,
    load_model,
)

# Globales para el ciclo de vida del modelo
model_bundle: Optional[Dict[str, Any]] = None
model_version_info: Dict[str, str] = {"version": "Desconocida", "run_id": "Desconocido"}
feature_columns: List[str] = []


def _initialize_model():
    global model_bundle, model_version_info, feature_columns
    model_bundle = load_model()
    model_version_info = get_model_metadata()
    feature_columns = load_feature_columns()

    if model_bundle and "model" in model_bundle:
        print(f"[FastAPI] ¡Modelo v{model_version_info['version']} cargado exitosamente en producción!")
    else:
        print("[FastAPI ERROR] El modelo inició en None.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialización al arrancar
    _initialize_model()
    yield
    # Limpieza al apagar (si aplica)


# ==========================================
# 1. INICIALIZACIÓN DE FASTAPI
# ==========================================
app = FastAPI(
    title="API de Clasificación de Salarios (Adult Census Income)",
    description="API MLOps robusta con arquitectura modular para predecir si un individuo gana <=50K o >50K.",
    version="2.5.0",
    lifespan=lifespan,
)

# ==========================================
# 2. ESQUEMAS DE ENTRADA (Pydantic V2)
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
# 3. HELPER PREPROCESAMIENTO DE ENTRADA
# ==========================================
def preprocess_input(raw_df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """Aplica One-Hot Encoding a la entrada recibida y alinea las columnas con el dataset de entrenamiento."""
    column_mapping = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country",
    }
    df = raw_df.rename(columns=column_mapping)
    categorical_cols = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    encoded_df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)

    if expected_cols:
        encoded_df = encoded_df.reindex(columns=expected_cols, fill_value=0)

    return encoded_df


# ==========================================
# 4. ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "model_name": MODEL_NAME,
        "production_version": model_version_info["version"],
        "run_id": model_version_info["run_id"],
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    if model_bundle is None or "model" not in model_bundle:
        # Fallback de inicialización si es llamado sin lifespan (ej. algunas pruebas)
        _initialize_model()

    if model_bundle is None or "model" not in model_bundle:
        raise HTTPException(
            status_code=500,
            detail="El modelo no está cargado en memoria o no se encontró en el Model Registry.",
        )

    try:
        raw_items = [item.model_dump(by_alias=True) for item in payload.data]
        raw_df = pd.DataFrame(raw_items)

        processed_df = preprocess_input(raw_df, feature_columns)

        scaler = model_bundle["scaler"]
        X_scaled = scaler.transform(processed_df)

        model = model_bundle["model"]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

        results = []
        for i, pred in enumerate(predictions):
            class_label = ">50K" if pred == 1 else "<=50K"
            prob_le_50k = float(probabilities[i][0]) if probabilities is not None else None
            prob_gt_50k = float(probabilities[i][1]) if probabilities is not None else None
            confidence = float(max(probabilities[i])) if probabilities is not None else None

            results.append(
                {
                    "index": i,
                    "prediction_code": int(pred),
                    "diagnosis": class_label,
                    "confidence_score": round(confidence * 100, 2) if confidence is not None else None,
                    "probabilities_detail": {
                        "<=50K": round(prob_le_50k, 4) if prob_le_50k is not None else None,
                        ">50K": round(prob_gt_50k, 4) if prob_gt_50k is not None else None,
                    },
                }
            )

        return {
            "model_metadata": {
                "name": MODEL_NAME,
                "version": model_version_info["version"],
                "run_id": model_version_info["run_id"],
            },
            "total_predictions": len(predictions),
            "results": results,
            "message": "Inferencia completada con éxito.",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")
