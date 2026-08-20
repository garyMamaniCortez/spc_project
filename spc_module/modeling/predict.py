"""
predict.py
----------
Carga el modelo entrenado (junto con el scaler usado en train.py),
predice sobre un set de features y guarda las predicciones en CSV.
"""

import pickle
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from spc_module.config import MODELS_DIR, PROCESSED_DATA_DIR
from spc_module.eda.loader import CSVDataLoader

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    logger.info(f"Cargando modelo y scaler desde: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]

    logger.info(f"Cargando features desde: {features_path}")
    X = CSVDataLoader(features_path).load()
    X_scaled = scaler.transform(X)

    logger.info("Realizando predicciones...")
    predictions = model.predict(X_scaled)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(predictions_path, index=False)

    logger.success(f"Predicciones guardadas en: {predictions_path}")


if __name__ == "__main__":
    app()
