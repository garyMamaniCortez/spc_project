"""
train.py
--------
Carga las tablas ya procesadas (features/labels de train y test),
escala, entrena la red neuronal, evalúa y guarda el modelo entrenado
para que predict.py pueda usarlo después.
"""

import pickle
from pathlib import Path

from loguru import logger
from sklearn.preprocessing import StandardScaler
import typer

from spc_module.config import MODELS_DIR, PROCESSED_DATA_DIR
from spc_module.eda.loader import CSVDataLoader
from spc_module.modeling.evaluator import Evaluator
from spc_module.modeling.models import NeuralNetworkModel

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    figures_path: Path = PROCESSED_DATA_DIR.parent.parent / "reports" / "figures" / "confusion_matrices.png",
    # -----------------------------------------
):
    logger.info("Cargando datos ya procesados...")
    X_train = CSVDataLoader(features_path).load()
    y_train = CSVDataLoader(labels_path).load().iloc[:, 0]  # asume una sola columna de etiqueta
    X_test = CSVDataLoader(test_features_path).load()
    y_test = CSVDataLoader(test_labels_path).load().iloc[:, 0]

    logger.info("Escalando features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Entrenando modelo...")
    model = NeuralNetworkModel()
    model.fit(X_train_scaled, y_train)
    logger.success(f"Entrenamiento terminado: {model}")

    logger.info("Evaluando modelo sobre el set de prueba...")
    y_pred = model.predict(X_test_scaled)
    evaluator = Evaluator()
    evaluator.evaluate(model.name, y_test, y_pred)
    evaluator.print_summary(model.name)
    evaluator.compare()

    figures_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.plot_confusion_matrices(str(figures_path))
    logger.info(f"Matriz de confusión guardada en: {figures_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    logger.success(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    app()