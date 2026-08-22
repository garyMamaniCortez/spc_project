"""
train.py
--------
Carga las tablas ya procesadas (features/labels de train y test),
escala las columnas numéricas continuas,
entrena varios modelos (red neuronal MLP y XGBoost), evalúa cada uno,
registra todo en MLflow (parámetros, métricas, artefactos y el modelo)
y guarda los modelos entrenados en disco para que ``predict.py`` los
use después.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import yaml

from loguru import logger
import mlflow
import mlflow.tensorflow
import mlflow.xgboost
import pandas as pd
import typer

from spc_module.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    NUMERICAL_COLUMNS,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REPORTS_DIR,
)
from spc_module.eda.loader import CSVDataLoader
from spc_module.modeling.evaluator import Evaluator
from spc_module.modeling.models import BaseModel, NeuralNetworkModel, XGBoostModel
from spc_module.preprocessing.scaling import StandardNumericalScaler

app = typer.Typer()

# Registro de modelos a entrenar y comparar. Añadir un modelo nuevo al
# proyecto es tan simple como agregar una entrada aquí (Open/Closed
# Principle): el resto de train.py no necesita cambiar.
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "mlp": NeuralNetworkModel,
    "xgboost": XGBoostModel,
}


def _loggable_params(params: dict) -> dict[str, str]:
    """Convierte hiperparámetros a un formato seguro para ``mlflow.log_params``."""
    return {k: str(v) for k, v in params.items() if v is not None}


def _train_and_log_model(
    key: str,
    model: BaseModel,
    x_train,
    y_train,
    x_test,
    y_test,
    evaluator: Evaluator,
    models_dir: Path,
    scaler: StandardNumericalScaler,
) -> dict:
    """Entrena un modelo, lo evalúa y registra todo en un run de MLflow."""
    with mlflow.start_run(run_name=model.name):
        mlflow.set_tag("model_key", key)
        mlflow.log_params(_loggable_params(model.get_params()))

        logger.info(f"Entrenando: {model.name}...")
        model.fit(x_train, y_train)
        logger.success(f"Entrenamiento terminado: {model}")

        y_pred = model.predict(x_test)
        metrics = evaluator.evaluate(model.name, y_test, y_pred)
        evaluator.print_summary(model.name)

        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = Path(tmp_dir) / f"confusion_matrix_{key}.png"
            evaluator.plot_single_confusion_matrix(model.name, str(cm_path))
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

        if key == "xgboost":
            mlflow.xgboost.log_model(model.model, name="model")
        else:
            mlflow.tensorflow.log_model(model.model, name="model")

        model_path = models_dir / f"model_{key}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        mlflow.log_artifact(str(model_path), artifact_path="bundle")
        logger.success(f"Modelo '{key}' guardado en: {model_path}")

        return metrics


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    figures_path: Path = REPORTS_DIR / "figures" / "confusion_matrices.png",
    # -----------------------------------------
    models: str = typer.Option(
        "mlp,xgboost",
        help="Modelos a entrenar, separados por coma (mlp,xgboost).",
    ),
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    """Entrena, evalúa y registra en MLflow todos los modelos configurados."""
    # Cargar parámetros desde params.yaml si existe
    params_file = PROJ_ROOT / "params.yaml"
    params_cfg = {}
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            params_cfg = yaml.safe_load(f) or {}

    train_cfg = params_cfg.get("train", {})
    if "models" in train_cfg:
        models = train_cfg["models"]

    model_keys = [m.strip() for m in models.split(",") if m.strip()]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI: {tracking_uri} | experimento: {experiment_name}")

    logger.info("Cargando datos ya procesados...")
    x_train = CSVDataLoader(features_path).load()
    y_train = CSVDataLoader(labels_path).load().iloc[:, 0]  # asume una sola columna de etiqueta
    x_test = CSVDataLoader(test_features_path).load()
    y_test = CSVDataLoader(test_labels_path).load().iloc[:, 0]

    # --- Escalado correcto: SOLO columnas numéricas continuas ---
    scaler = StandardNumericalScaler(columns=NUMERICAL_COLUMNS)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    evaluator = Evaluator()
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    trained_models: dict[str, BaseModel] = {}
    for key in model_keys:
        model_cls = MODEL_REGISTRY[key]
        model_kwargs = train_cfg.get(key, {})
        try:
            model_instance = model_cls(**model_kwargs)
        except Exception:
            model_instance = model_cls()

        metrics = _train_and_log_model(
            key=key,
            model=model_instance,
            x_train=x_train_scaled,
            y_train=y_train,
            x_test=x_test_scaled,
            y_test=y_test,
            evaluator=evaluator,
            models_dir=models_dir,
            scaler=scaler,
        )
        results[key] = metrics
        trained_models[key] = model_instance

    evaluator.compare()

    figures_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.plot_confusion_matrices(str(figures_path))
    logger.info(f"Matrices de confusión (comparativa) guardadas en: {figures_path}")

    best_key = max(results, key=lambda k: results[k]["f1_score"])
    best_model_path = models_dir / f"model_{best_key}.pkl"
    model_path.write_bytes(best_model_path.read_bytes())
    logger.success(
        f"Mejor modelo por F1-score: '{best_key}' "
        f"(f1={results[best_key]['f1_score']:.4f}) -> copiado a {model_path}"
    )

    # --- Guardar métricas en JSON para DVC ---
    metrics_file = REPORTS_DIR / "metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    formatted_metrics = {}
    for m_name, m_vals in results.items():
        formatted_metrics[m_name] = {
            "accuracy": float(m_vals["accuracy"]),
            "precision": float(m_vals["precision"]),
            "recall": float(m_vals["recall"]),
            "f1_score": float(m_vals["f1_score"]),
        }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(formatted_metrics, f, indent=4)
    logger.info(f"Métricas para DVC guardadas en: {metrics_file}")

    # --- Guardar predicciones en CSV para gráficos DVC ---
    best_model = trained_models[best_key]
    preds = best_model.predict(x_test_scaled)
    plots_df = pd.DataFrame({"actual": y_test.values, "predicted": preds})
    plots_file = REPORTS_DIR / "plots.csv"
    plots_df.to_csv(plots_file, index=False)
    logger.info(f"Predicciones para gráficos DVC guardadas en: {plots_file}")

    logger.info(f"Revisa 'mlflow ui --backend-store-uri {tracking_uri}' para ver los runs.")


if __name__ == "__main__":
    app()
