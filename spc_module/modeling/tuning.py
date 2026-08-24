"""
tuning.py
---------
Módulo para el tuning de hiperparámetros de XGBoost y Red Neuronal.

Responsabilidades:
- XGBoostTuner: Realiza RandomizedSearchCV sobre X_train usando validación cruzada (cv=3)
  y scoring F1-score sin tocar el conjunto de test.
- NeuralNetworkTuner: Realiza una búsqueda modular acotada sobre un espacio pequeño
  de combinaciones para la Red Neuronal (MLP) utilizando validación interna
  (validation_split=0.2) y EarlyStopping sin tocar el conjunto de test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import itertools
import numpy as np
from loguru import logger
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

from spc_module.modeling.models import BaseModel, NeuralNetworkModel, XGBoostModel


class BaseTuner(ABC):
    """Clase base abstracta para los tuners de hiperparámetros.

    Establece el contrato e interfaz unificada que todo tuner del proyecto debe implementar.
    """

    @abstractmethod
    def tune(self, x_train, y_train) -> tuple[BaseModel, dict, float]:
        """Ejecuta la búsqueda y optimización de hiperparámetros sobre el conjunto de entrenamiento."""
        pass


class XGBoostTuner(BaseTuner):
    """Tuning de hiperparámetros para XGBoost usando RandomizedSearchCV."""

    def __init__(
        self,
        param_distributions: dict | None = None,
        n_iter: int = 8,
        cv: int = 3,
        scoring: str = "f1",
        random_state: int = 42,
    ):
        if param_distributions is None:
            param_distributions = {
                "n_estimators": [150, 300, 450],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1, 0.2],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "min_child_weight": [1, 3, 5],
                "scale_pos_weight": [1, 2, 3],
            }
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_: dict = {}
        self.best_score_: float = 0.0

    def tune(self, x_train, y_train) -> tuple[XGBoostModel, dict, float]:
        """Ejecuta RandomizedSearchCV sobre train para encontrar los mejores parámetros.

        Returns
        -------
        best_model: XGBoostModel
            Modelo XGBoost instanciado y entrenado con los mejores hiperparámetros.
        best_params: dict
            Diccionario de los mejores hiperparámetros encontrados.
        best_score: float
            Puntuación F1-score media obtenida en la validación cruzada.
        """
        logger.info(
            f"Iniciando tuning de XGBoost con RandomizedSearchCV (n_iter={self.n_iter}, cv={self.cv}, scoring='{self.scoring}')..."
        )
        base_xgb = XGBoostModel(random_state=self.random_state).model

        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(x_train, y_train)
        self.best_params_ = search.best_params_
        self.best_score_ = float(search.best_score_)

        logger.success(
            f"Tuning XGBoost finalizado. Best CV F1-score: {self.best_score_:.4f} | Best params: {self.best_params_}"
        )

        # Crear y entrenar una instancia de XGBoostModel con los mejores parámetros
        best_model = XGBoostModel(
            n_estimators=self.best_params_.get("n_estimators", 300),
            max_depth=self.best_params_.get("max_depth", 6),
            learning_rate=self.best_params_.get("learning_rate", 0.1),
            subsample=self.best_params_.get("subsample", 0.9),
            colsample_bytree=self.best_params_.get("colsample_bytree", 0.9),
            min_child_weight=self.best_params_.get("min_child_weight", 1),
            scale_pos_weight=self.best_params_.get("scale_pos_weight", 1.0),
            random_state=self.random_state,
        )
        best_model.fit(x_train, y_train)
        return best_model, self.best_params_, self.best_score_


class NeuralNetworkTuner(BaseTuner):
    """Tuning modular de hiperparámetros para la Red Neuronal (MLP)."""

    def __init__(
        self,
        param_grid: dict | None = None,
        random_state: int = 42,
    ):
        if param_grid is None:
            param_grid = {
                "hidden_layer_sizes": [
                    (512, 256),
                    (256, 128),
                    (256, 128, 64),
                    (128, 64),
                ],
                "learning_rate_init": [1e-4, 5e-4],
                "dropout_rate": [0.1, 0.2, 0.3, 0.4],
                "l2_reg": [1e-5, 1e-4],
                "pos_weight": [1.0, 1.5, 2.0],
                "batch_size": [32, 64],
            }
        self.param_grid = param_grid
        self.random_state = random_state
        self.best_params_: dict = {}
        self.best_score_: float = 0.0

    def tune(self, x_train, y_train) -> tuple[NeuralNetworkModel, dict, float]:
        """Ejecuta una búsqueda en grid acotada usando validation_split sobre train.

        Returns
        -------
        best_model: NeuralNetworkModel
            Modelo MLP instanciado y entrenado con los mejores hiperparámetros.
        best_params: dict
            Diccionario de los mejores hiperparámetros encontrados.
        best_score: float
            Puntuación F1-score estimada en validación interna.
        """
        # Generar lista de combinaciones
        keys, values = zip(*self.param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # Evaluamos hasta 8 configuraciones candidatas para explorar mejor el espacio
        if len(combinations) > 8:
            rng = np.random.default_rng(self.random_state)
            sampled_indices = rng.choice(len(combinations), size=8, replace=False)
            combinations = [combinations[i] for i in sampled_indices]

        logger.info(f"Iniciando tuning avanzado de Red Neuronal sobre {len(combinations)} configuraciones candidatas...")

        best_val_f1 = -1.0
        best_config = combinations[0]
        
        # Split interno para calcular F1 de validación sin tocar test
        val_size = int(len(x_train) * 0.2)
        x_tr = x_train.iloc[:-val_size] if hasattr(x_train, "iloc") else x_train[:-val_size]
        y_tr = y_train.iloc[:-val_size] if hasattr(y_train, "iloc") else y_train[:-val_size]
        x_val = x_train.iloc[-val_size:] if hasattr(x_train, "iloc") else x_train[-val_size:]
        y_val = y_train.iloc[-val_size:] if hasattr(y_train, "iloc") else y_train[-val_size:]

        for i, config in enumerate(combinations, start=1):
            logger.info(f"Evaluando candidato {i}/{len(combinations)} MLP: {config}")
            candidate = NeuralNetworkModel(
                hidden_layer_sizes=config["hidden_layer_sizes"],
                learning_rate_init=config["learning_rate_init"],
                dropout_rate=config["dropout_rate"],
                l2_reg=config.get("l2_reg", 1e-4),
                pos_weight=config.get("pos_weight", 1.0),
                batch_size=config["batch_size"],
                early_stopping=True,
                n_iter_no_change=10,
                max_iter=300,
                validation_split=0.2,
                random_state=self.random_state,
            )
            candidate.fit(x_tr, y_tr)
            y_val_pred = candidate.predict(x_val)
            val_f1 = f1_score(y_val, y_val_pred)

            logger.info(f"Candidato {i} Val F1-score: {val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config = config

        self.best_params_ = best_config
        self.best_score_ = float(best_val_f1)
        logger.success(
            f"Tuning Red Neuronal finalizado. Best Val F1-score: {self.best_score_:.4f} | Best params: {self.best_params_}"
        )

        # Entrenar modelo final con la mejor configuración sobre todo x_train
        best_model = NeuralNetworkModel(
            hidden_layer_sizes=best_config["hidden_layer_sizes"],
            learning_rate_init=best_config["learning_rate_init"],
            dropout_rate=best_config["dropout_rate"],
            l2_reg=best_config.get("l2_reg", 1e-4),
            pos_weight=best_config.get("pos_weight", 1.0),
            batch_size=best_config["batch_size"],
            early_stopping=True,
            n_iter_no_change=15,
            max_iter=500,
            validation_split=0.2,
            random_state=self.random_state,
        )
        best_model.fit(x_train, y_train)
        return best_model, self.best_params_, self.best_score_
