from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from loguru import logger
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_loader import (
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
        logger.info(f"[ModelService] ¡Modelo v{model_version_info.get('version')} cargado exitosamente!")
    else:
        logger.error("[ModelService] Error: No se pudo cargar el modelo en memoria.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _initialize_model()
    yield


app = FastAPI(
    title="Microservicio Interno de Modelos ML (Adult Census Income)",
    description="Servicio interno de inferencia de Machine Learning con XGBoost / Red Neuronal.",
    version="3.0.0",
    lifespan=lifespan,
)


class RawPredictionRequest(BaseModel):
    data: List[Dict[str, Any]]


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


@app.get("/")
def read_root():
    if model_bundle is None or "model" not in model_bundle:
        _initialize_model()
    return {
        "service": "Model Service",
        "status": "Online" if (model_bundle and "model" in model_bundle) else "Degraded",
        "model_name": MODEL_NAME,
        "metadata": model_version_info,
        "features_loaded": len(feature_columns),
    }


@app.get("/health")
def health():
    if model_bundle is None or "model" not in model_bundle:
        _initialize_model()
    if model_bundle and "model" in model_bundle:
        return {"status": "Healthy", "model": "Ready"}
    return {"status": "Unhealthy", "model": "Not Loaded"}


@app.post("/predict")
def predict(payload: RawPredictionRequest):
    if model_bundle is None or "model" not in model_bundle:
        _initialize_model()

    if model_bundle is None or "model" not in model_bundle:
        raise HTTPException(
            status_code=500,
            detail="El modelo no está disponible en memoria en el contenedor de modelos.",
        )

    try:
        raw_df = pd.DataFrame(payload.data)
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
                "version": model_version_info.get("version", "1.0.0"),
                "run_id": model_version_info.get("run_id", "xgboost_tuned_production"),
            },
            "total_predictions": len(predictions),
            "results": results,
            "message": "Inferencia completada con éxito desde el microservicio de modelos.",
        }

    except Exception as e:
        logger.error(f"[ModelService] Error en inferencia: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")
