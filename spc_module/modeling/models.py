"""
models.py
----------
Define los modelos de clasificación usando POO.

- BaseModel: clase abstracta que define el contrato común (fit/predict).
  Se deja como clase abstracta (en vez de meter todo directo en cada
  modelo) para que el proyecto quede abierto a agregar otros modelos
  en el futuro sin romper el resto del código (Open/Closed Principle).
  ``train.py`` y ``predict.py`` dependen únicamente de este contrato,
  nunca de una implementación concreta (Dependency Inversion Principle).
- NeuralNetworkModel: red neuronal (Perceptrón Multicapa) para clasificar
  el salario.
- XGBoostModel: modelo de gradient boosting (XGBoost) para el mismo
  problema de clasificación binaria.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia logs de bajo nivel de TF

from typing import TYPE_CHECKING

import numpy as np
from xgboost import XGBClassifier

if TYPE_CHECKING:
    # Solo se importa para chequeo de tipos (no en tiempo de ejecución).
    # tensorflow es una dependencia pesada que solo necesita NeuralNetworkModel;
    # XGBoostModel (el modelo en producción) no debe requerirla para poder
    # des-picklearse en el contenedor liviano de inferencia (model_service).
    from tensorflow import keras


class BaseModel(ABC):
    """Contrato común que deben cumplir todos los modelos del proyecto."""

    name: str = "BaseModel"

    def __init__(self):
        self.model = None
        self.is_trained = False

    @abstractmethod
    def fit(self, X_train, y_train):
        """Entrena el modelo."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        """Genera predicciones sobre nuevos datos."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X_test):
        """Calcula probabilidades para cada clase [prob_0, prob_1]."""
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict:
        """Hiperparámetros del modelo (para logging, p. ej. MLflow)."""
        raise NotImplementedError

    def __repr__(self):
        estado = "entrenado" if self.is_trained else "sin entrenar"
        return f"<{self.name} ({estado})>"


class NeuralNetworkModel(BaseModel):
    """Red neuronal profunda (Keras/TensorFlow) para clasificar el salario
    (<=50K o >50K).

    Arquitectura, por cada tamaño en ``hidden_layer_sizes``
    (256 -> 128 -> 64 -> 32 por defecto):
        Dense -> BatchNormalization -> Activation(ReLU) -> Dropout(0.3)
    seguida de una capa de salida ``Dense(1, activation="sigmoid")`` para
    clasificación binaria.
    """

    name = "Red Neuronal (MLP)"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (256, 128, 64, 32),
        dropout_rate: float = 0.4,
        activation: str = "relu",
        l2_reg: float = 1e-4,
        learning_rate_init: float = 1e-4,
        max_iter: int = 500,
        batch_size: int = 32,
        early_stopping: bool = True,
        n_iter_no_change: int = 15,
        validation_split: float = 0.2,
        pos_weight: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.l2_reg = l2_reg
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_split = validation_split
        self.pos_weight = pos_weight
        self.random_state = random_state
        self.history_ = None

    def _build(self, input_dim: int) -> "keras.Model":
        from tensorflow import keras  # import perezoso: solo se necesita al construir/usar la red neuronal

        keras.utils.set_random_seed(self.random_state)
        regularizer = keras.regularizers.l2(self.l2_reg) if self.l2_reg else None
 
        network = keras.Sequential(name="mlp_dropout_batchnorm")
        network.add(keras.layers.Input(shape=(input_dim,)))
        for i, units in enumerate(self.hidden_layer_sizes, start=1):
            network.add(
                keras.layers.Dense(units, kernel_regularizer=regularizer, name=f"dense_{i}")
            )
            network.add(keras.layers.BatchNormalization(name=f"batch_norm_{i}"))
            network.add(keras.layers.Activation(self.activation, name=f"{self.activation}_{i}"))
            network.add(keras.layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))
        network.add(keras.layers.Dense(1, activation="sigmoid", name="output"))
 
        network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate_init),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return network

    def fit(self, X_train, y_train):
        x_array = np.asarray(X_train, dtype="float32")
        y_array = np.asarray(y_train, dtype="float32")
        self.model = self._build(input_dim=x_array.shape[1])
 
        callbacks = []
        if self.early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.n_iter_no_change,
                    restore_best_weights=True,
                )
            )
 
        class_weight = {0: 1.0, 1: float(self.pos_weight)} if self.pos_weight != 1.0 else None

        self.history_ = self.model.fit(
            x_array,
            y_array,
            epochs=self.max_iter,
            batch_size=self.batch_size,
            validation_split=self.validation_split if self.early_stopping else 0.0,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )
        self.is_trained = True
        return self

    def predict(self, X_test):
        x_array = np.asarray(X_test, dtype="float32")
        probabilities = self.model.predict(x_array, verbose=0).ravel()
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X_test):
        x_array = np.asarray(X_test, dtype="float32")
        prob_1 = self.model.predict(x_array, verbose=0).ravel()
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def get_params(self) -> dict:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "l2_reg": self.l2_reg,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "validation_split": self.validation_split,
            "pos_weight": self.pos_weight,
            "random_state": self.random_state,
        }
    
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        keras_model = state.pop("model")
        state.pop("history_", None)  # el historial de entrenamiento no se persiste
        state["_model_bytes"] = self._keras_model_to_bytes(keras_model)
        return state
 
    def __setstate__(self, state: dict) -> None:
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self.history_ = None
        self.model = self._keras_model_from_bytes(model_bytes)
 
    @staticmethod
    def _keras_model_to_bytes(keras_model: "keras.Model | None") -> bytes | None:
        if keras_model is None:
            return None
        fd, tmp_path = tempfile.mkstemp(suffix=".keras")
        os.close(fd)
        try:
            keras_model.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.remove(tmp_path)
 
    @staticmethod
    def _keras_model_from_bytes(model_bytes: bytes | None) -> "keras.Model | None":
        if model_bytes is None:
            return None
        from tensorflow import keras  # import perezoso: solo se necesita al restaurar la red neuronal

        fd, tmp_path = tempfile.mkstemp(suffix=".keras")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(model_bytes)
            return keras.models.load_model(tmp_path)
        finally:
            os.remove(tmp_path)
 
 
class XGBoostModel(BaseModel):
    """Gradient boosting (XGBoost) para clasificar el salario (<=50K o >50K).

    A diferencia de la red neuronal, los árboles de decisión no
    requieren que las variables estén escaladas; se entrena con las
    mismas features (escaladas o no) para mantener el pipeline simple
    y comparable, ya que escalar columnas numéricas continuas no
    perjudica a XGBoost.
    """

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_params(self) -> dict:
        return self.model.get_params()

