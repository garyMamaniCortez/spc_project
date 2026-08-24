"""
train.py
--------
Carga las tablas ya procesadas (features/labels de train y test),
escala las columnas numéricas continuas,
entrena los modelos baseline (red neuronal MLP y XGBoost),
ejecuta tuning de hiperparámetros sobre train para ambos modelos,
evalúa cada modelo (baseline vs tuned) en el conjunto de test,
registra todo en MLflow (parámetros, métricas, artefactos y modelos)
y guarda los modelos en disco para que ``predict.py`` los use después.
"""

from __future__ import annotations

import hashlib
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
from spc_module.modeling.tuning import NeuralNetworkTuner, XGBoostTuner
from spc_module.preprocessing.scaling import StandardNumericalScaler

app = typer.Typer()

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "mlp": NeuralNetworkModel,
    "xgboost": XGBoostModel,
}


def _loggable_params(params: dict) -> dict[str, str]:
    """Convierte hiperparámetros a un formato seguro para ``mlflow.log_params``."""
    return {k: str(v) for k, v in params.items() if v is not None}

def _dataset_digest(path: Path) -> str:
    """Calcula un hash SHA-256 del dataset para identificar la versión utilizada."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _log_and_save_model(
    run_name: str,
    model_key: str,
    version_tag: str,
    model: BaseModel,
    x_test,
    y_test,
    evaluator: Evaluator,
    models_dir: Path,
    scaler: StandardNumericalScaler,
    dataset_name: str,
    dataset_digest: str,
    mlflow_dataset,
) -> dict:
    """Evalúa un modelo previamente entrenado, guarda sus métricas/artefactos y lo registra en MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_key", model_key)
        mlflow.set_tag("version", version_tag)
        mlflow.set_tag("dataset_name", dataset_name)
        mlflow.set_tag("dataset_sha256", dataset_digest)
        mlflow.log_params(_loggable_params(model.get_params()))

        mlflow.log_input(
            mlflow_dataset,
            context="training",
        )

        y_pred = model.predict(x_test)
        metrics = evaluator.evaluate(run_name, y_test, y_pred)
        evaluator.print_summary(run_name)

        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            safe_filename = run_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            cm_path = Path(tmp_dir) / f"confusion_matrix_{safe_filename}.png"
            evaluator.plot_single_confusion_matrix(run_name, str(cm_path))
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

        if model_key == "xgboost":
            mlflow.xgboost.log_model(model.model, name="model")
        else:
            mlflow.tensorflow.log_model(model.model, name="model")

        model_filename = f"model_{model_key}_{version_tag}.pkl"
        model_path = models_dir / model_filename
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        mlflow.log_artifact(str(model_path), artifact_path="bundle")
        logger.success(f"Modelo '{run_name}' guardado en: {model_path}")

        return metrics


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    figures_path: Path = REPORTS_DIR / "figures" / "confusion_matrices.png",
    models: str = typer.Option(
        "mlp,xgboost",
        help="Modelos a entrenar, separados por coma (mlp,xgboost).",
    ),
    do_tuning: bool = typer.Option(
        True,
        help="Si es True, ejecuta tuning de hiperparámetros además de los baseline.",
    ),
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    """Entrena modelos baseline y tuned, los evalúa en test y los registra en MLflow."""
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
    y_train = CSVDataLoader(labels_path).load().iloc[:, 0]
    x_test = CSVDataLoader(test_features_path).load()
    y_test = CSVDataLoader(test_labels_path).load().iloc[:, 0]

    raw_dataset_path = PROJ_ROOT / "data" / "raw" / "salary.csv"
    dataset_name = raw_dataset_path.name
    dataset_digest = _dataset_digest(raw_dataset_path)

    dataset_df = pd.read_csv(raw_dataset_path)
    mlflow_dataset = mlflow.data.from_pandas(
        dataset_df,
        source=str(raw_dataset_path),
        name=dataset_name,
    )

    logger.info(
        f"Dataset utilizado: {dataset_name} | SHA-256: {dataset_digest}"
    )

    # --- Escalado correcto: SOLO columnas numéricas continuas ---
    scaler = StandardNumericalScaler(columns=NUMERICAL_COLUMNS)
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    evaluator = Evaluator()
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    trained_models: dict[str, BaseModel] = {}
    saved_paths: dict[str, Path] = {}

    for key in model_keys:
        # 1. ENTRENAR Y LOGUEAR BASELINE
        baseline_name = f"{'XGBoost' if key == 'xgboost' else 'Red Neuronal (MLP)'} (Baseline)"
        logger.info(f"Entrenando baseline: {baseline_name}...")
        model_cls = MODEL_REGISTRY[key]
        model_kwargs = train_cfg.get(key, {})
        try:
            baseline_model = model_cls(**model_kwargs)
        except Exception:
            baseline_model = model_cls()

        baseline_model.fit(x_train_scaled, y_train)

        metrics_baseline = _log_and_save_model(
            run_name=baseline_name,
            model_key=key,
            version_tag="baseline",
            model=baseline_model,
            x_test=x_test_scaled,
            y_test=y_test,
            evaluator=evaluator,
            models_dir=models_dir,
            scaler=scaler,
            dataset_name=dataset_name,
            dataset_digest=dataset_digest,
            mlflow_dataset=mlflow_dataset,
        )
        results[baseline_name] = metrics_baseline
        trained_models[baseline_name] = baseline_model
        saved_paths[baseline_name] = models_dir / f"model_{key}_baseline.pkl"

        # 2. TUNING DE HIPERPARÁMETROS
        if do_tuning:
            tuned_name = f"{'XGBoost' if key == 'xgboost' else 'Red Neuronal (MLP)'} (Tuned)"
            logger.info(f"Iniciando tuning para: {tuned_name}...")
            if key == "xgboost":
                tuner = XGBoostTuner(n_iter=8, cv=3, random_state=42)
                tuned_model, best_params, best_val_score = tuner.tune(x_train_scaled, y_train)
            else:
                tuner = NeuralNetworkTuner(random_state=42)
                tuned_model, best_params, best_val_score = tuner.tune(x_train_scaled, y_train)

            logger.info(f"Evaluando modelo tuned ({tuned_name}) sobre el set de test...")
            metrics_tuned = _log_and_save_model(
                run_name=tuned_name,
                model_key=key,
                version_tag="tuned",
                model=tuned_model,
                x_test=x_test_scaled,
                y_test=y_test,
                evaluator=evaluator,
                models_dir=models_dir,
                scaler=scaler,
                dataset_name=dataset_name,
                dataset_digest=dataset_digest,
                mlflow_dataset=mlflow_dataset,
            )
            results[tuned_name] = metrics_tuned
            trained_models[tuned_name] = tuned_model
            saved_paths[tuned_name] = models_dir / f"model_{key}_tuned.pkl"

    evaluator.compare()

    figures_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.plot_confusion_matrices(str(figures_path))
    logger.info(f"Matrices de confusión (comparativa) guardadas en: {figures_path}")

    best_run_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model_source = saved_paths[best_run_name]
    model_path.write_bytes(best_model_source.read_bytes())
    logger.success(
        f"Mejor modelo global por F1-score: '{best_run_name}' "
        f"(f1={results[best_run_name]['f1_score']:.4f}) -> copiado a {model_path}"
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
    best_model = trained_models[best_run_name]
    preds = best_model.predict(x_test_scaled)
    plots_df = pd.DataFrame({"actual": y_test.values, "predicted": preds})
    plots_file = REPORTS_DIR / "plots.csv"
    plots_df.to_csv(plots_file, index=False)
    logger.info(f"Predicciones para gráficos DVC guardadas en: {plots_file}")

    logger.info(f"Revisa 'mlflow ui --backend-store-uri {tracking_uri}' para ver los runs.")


if __name__ == "__main__":
    app()
