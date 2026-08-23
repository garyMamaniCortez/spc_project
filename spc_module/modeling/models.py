"""
models.py
----------
Define los modelos de clasificación usando POO.

- BaseModel: clase abstracta que define el contrato común (fit/predict).
  Se deja como clase abstracta (en vez de meter todo directo en
  NeuralNetworkModel) para que el proyecto quede abierto a agregar
  otros modelos en el futuro sin romper el resto del código.
- NeuralNetworkModel: red neuronal (Perceptrón Multicapa) para clasificar
  el salario.
"""

from abc import ABC, abstractmethod

from sklearn.neural_network import MLPClassifier


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

    def __repr__(self):
        estado = "entrenado" if self.is_trained else "sin entrenar"
        return f"<{self.name} ({estado})>"


class NeuralNetworkModel(BaseModel):
    """Red neuronal (Perceptrón Multicapa) para clasificar el salario
    (<=50K o >50K).

    Nota: requiere datos escalados (StandardScaler) para converger bien,
    ya que las neuronas son sensibles a la magnitud de las variables.
    """

    name = "Red Neuronal (MLP)"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        max_iter: int = 300,
        random_state: int = 42,
    ):
        super().__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
        )

    def fit(self, X_train_scaled, y_train):
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        return self

    def predict(self, X_test_scaled):
        return self.model.predict(X_test_scaled)
