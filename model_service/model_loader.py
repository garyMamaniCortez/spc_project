"""
model_loader.py
---------------
Funciones de carga del modelo, escalador y definición de columnas
para el microservicio de inferencia de modelos.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
_env_model = os.getenv("MODEL_PATH")
_env_features = os.getenv("FEATURES_PATH")

if _env_model:
    DEFAULT_MODEL_PATH = Path(_env_model)
elif (BASE_DIR / "models" / "model.pkl").exists():
    DEFAULT_MODEL_PATH = BASE_DIR / "models" / "model.pkl"
else:
    DEFAULT_MODEL_PATH = BASE_DIR.parent / "models" / "model.pkl"

if _env_features:
    DEFAULT_FEATURES_PATH = Path(_env_features)
elif (BASE_DIR / "data" / "processed" / "features.csv").exists():
    DEFAULT_FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.csv"
else:
    DEFAULT_FEATURES_PATH = BASE_DIR.parent / "data" / "processed" / "features.csv"

MODEL_NAME = "Clasificación de Salarios (Adult Census Income)"


def load_model(model_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Carga el bundle del modelo (modelo + scaler) desde un archivo pickle."""
    target_path = model_path or DEFAULT_MODEL_PATH
    if not target_path.exists():
        logger.error(f"[ModelLoader] El archivo de modelo no existe en: {target_path}")
        return None

    try:
        with open(target_path, "rb") as f:
            bundle = pickle.load(f)
        logger.info(f"[ModelLoader] Modelo cargado exitosamente desde: {target_path}")
        return bundle
    except Exception as e:
        logger.error(f"[ModelLoader] Error al cargar el modelo: {str(e)}")
        return None


def get_model_metadata(model_path: Optional[Path] = None) -> Dict[str, str]:
    """Obtiene metadatos informativos sobre la versión de producción del modelo."""
    target_path = model_path or DEFAULT_MODEL_PATH
    if target_path.exists():
        return {
            "name": MODEL_NAME,
            "version": "1.0.0",
            "run_id": "xgboost_tuned_production",
            "model_file": target_path.name,
            "status": "Production",
        }
    return {
        "name": MODEL_NAME,
        "version": "Desconocida",
        "run_id": "Desconocido",
        "model_file": "No encontrado",
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
