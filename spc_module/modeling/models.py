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

from abc import ABC, abstractmethod

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


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

    def get_params(self) -> dict:
        """Hiperparámetros del estimador subyacente (para logging, p. ej. MLflow)."""
        return self.model.get_params()

    def __repr__(self):
        estado = "entrenado" if self.is_trained else "sin entrenar"
        return f"<{self.name} ({estado})>"


class NeuralNetworkModel(BaseModel):
    """Red neuronal (Perceptrón Multicapa) para clasificar el salario
    (<=50K o >50K).

    Arquitectura: 4 capas ocultas (256 -> 128 -> 64 -> 32 neuronas) con
    activación ReLU, optimizador Adam, regularización L2 (``alpha``) y
    early stopping para evitar sobreajuste.

    Nota: requiere datos escalados (StandardScaler) para converger bien,
    ya que las neuronas son sensibles a la magnitud de las variables.
    """

    name = "Red Neuronal (MLP)"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (256, 128, 64, 32),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = True,
        n_iter_no_change: int = 15,
        random_state: int = 42,
    ):
        super().__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            early_stopping=early_stopping,
        )

    def fit(self, X_train_scaled, y_train):
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        return self

    def predict(self, X_test_scaled):
        return self.model.predict(X_test_scaled)


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
        random_state: int = 42,
    ):
        super().__init__()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
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
