"""
model_loader.py
---------------
Funciones centralizadas de MLOps para la carga del modelo entrenado,
escalador y metadatos desde el directorio de modelos.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from spc_module.config import MODELS_DIR, PROCESSED_DATA_DIR

MODEL_NAME = "Clasificación de Salarios (Adult Census Income)"
DEFAULT_MODEL_PATH = MODELS_DIR / "model.pkl"
DEFAULT_FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"


def load_model(model_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Carga el bundle del modelo (modelo + scaler) desde un archivo pickle.

    Returns:
        dict con llaves 'model' y 'scaler', o None si falla la carga.
    """
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
            "version": "1.0.0",
            "run_id": "xgboost_tuned_production",
            "model_file": target_path.name,
            "status": "Production"
        }
    return {
        "version": "Desconocida",
        "run_id": "Desconocido",
        "model_file": "No encontrado",
        "status": "Error"
    }


def load_feature_columns(features_path: Optional[Path] = None) -> List[str]:
    """Obtiene la lista exacta de las 81 columnas One-Hot producidas en el entrenamiento."""
    target_path = features_path or DEFAULT_FEATURES_PATH
    if target_path.exists():
        df_cols = pd.read_csv(target_path, nrows=1)
        return df_cols.columns.tolist()
    else:
        logger.warning(f"[ModelLoader] No se encontró {target_path}, usando columnas por defecto.")
        # Retorna lista de columnas fallback si features.csv no está presente
        return []
