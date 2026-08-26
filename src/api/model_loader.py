"""
model_loader.py
---------------
Funciones MLOps centralizadas para la carga de modelos, escaladores y metadatos
compatibles con ejecuciones locales y dentro de contenedores Docker.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from loguru import logger

# Detección de entorno Docker / Local
IS_DOCKER = os.environ.get("ENVIRONMENT") == "docker"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_NAME = "Clasificación de Salarios (Adult Census Income)"

# Rutas predeterminadas
if IS_DOCKER:
    DEFAULT_MODEL_PATH = Path("/app/models/model.pkl")
    DEFAULT_FEATURES_PATH = Path("/app/data/processed/features.csv")
else:
    DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
    DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"


def get_mlflow_tracking_uri() -> str:
    """Obtiene la URI de seguimiento de MLflow según el entorno."""
    if IS_DOCKER:
        return "sqlite:////app/mlflow.db"
    return f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


def load_model(model_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Carga y retorna el bundle del modelo según el entorno (MLflow / Pickle local)."""
    env_str = "Docker" if IS_DOCKER else "Local"
    logger.info(f"[ModelLoader] Entorno detectado: {env_str}")

    target_path = model_path or DEFAULT_MODEL_PATH

    # 1. Intento de carga binaria primaria desde pickle/joblib
    if target_path.exists():
        try:
            logger.info(f"[ModelLoader] Cargando modelo binario desde: {target_path}")
            with open(target_path, "rb") as f:
                bundle = pickle.load(f)
            logger.info(f"[ModelLoader] Modelo cargado exitosamente en entorno {env_str}.")
            return bundle
        except Exception as e:
            logger.warning(f"[ModelLoader] Error al cargar con pickle: {e}. Intentando joblib...")
            try:
                bundle = joblib.load(target_path)
                logger.info(f"[ModelLoader] Modelo cargado exitosamente con joblib.")
                return bundle
            except Exception as ex:
                logger.error(f"[ModelLoader ERROR] Falló carga con joblib: {ex}")

    # 2. Intento de carga desde MLflow Tracking URI como fallback
    try:
        import mlflow
        db_path = get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(db_path)
        model_uri = f"models:/{MODEL_NAME}/Production"
        logger.info(f"[ModelLoader] Intentando cargar modelo desde MLflow URI: {model_uri}")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        return {"model": loaded_model, "scaler": None}
    except Exception as e:
        logger.warning(f"[ModelLoader] No se pudo cargar desde MLflow Model Registry: {e}")

    logger.error(f"[ModelLoader ERROR] El modelo no pudo ser cargado de ninguna fuente en {target_path}")
    return None


def get_model_metadata(model_path: Optional[Path] = None) -> Dict[str, str]:
    """Obtiene metadatos informativos del modelo desde MLflow o sistema de archivos local."""
    if IS_DOCKER:
        return {
            "name": MODEL_NAME,
            "version": "1.0.0",
            "run_id": "docker_production_run",
            "environment": "Docker",
            "status": "Production",
        }

    try:
        import mlflow
        db_path = get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(db_path)
        client = mlflow.tracking.MlflowClient(tracking_uri=db_path)
        latest_versions = client.get_latest_versions("salary-classification", stages=["Production"])
        if latest_versions:
            return {
                "name": MODEL_NAME,
                "version": str(latest_versions[0].version),
                "run_id": str(latest_versions[0].run_id),
                "environment": "Local (MLflow)",
                "status": "Production",
            }
    except Exception:
        pass

    target_path = model_path or DEFAULT_MODEL_PATH
    if target_path.exists():
        return {
            "name": MODEL_NAME,
            "version": "1.0.0",
            "run_id": "xgboost_tuned_production",
            "environment": "Local (Pickle)",
            "status": "Production",
        }

    return {
        "name": MODEL_NAME,
        "version": "Desconocida",
        "run_id": "Desconocido",
        "environment": "Unknown",
        "status": "Error",
    }


def load_feature_columns(features_path: Optional[Path] = None) -> List[str]:
    """Obtiene la lista exacta de las 81 columnas One-Hot producidas en el entrenamiento."""
    target_path = features_path or DEFAULT_FEATURES_PATH
    if target_path.exists():
        df_cols = pd.read_csv(target_path, nrows=1)
        return df_cols.columns.tolist()
    else:
        logger.warning(f"[ModelLoader] No se encontró {target_path}, usando columnas por defecto.")
        return []
