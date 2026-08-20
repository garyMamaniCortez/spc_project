"""
evaluator.py
-------------
Responsable de medir el desempeño de los modelos y compararlos entre sí.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class Evaluator:
    """Calcula métricas de clasificación y genera comparaciones visuales."""

    def __init__(self):
        self.results: dict[str, dict] = {}

    def evaluate(self, model_name: str, y_true, y_pred) -> dict:
        """Calcula las métricas principales para un modelo y las guarda."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "report": classification_report(y_true, y_pred),
        }
        self.results[model_name] = metrics
        return metrics

    def print_summary(self, model_name: str) -> None:
        """Imprime un resumen legible de las métricas de un modelo."""
        m = self.results[model_name]
        print(f"\n--- {model_name} ---")
        print(f"Accuracy : {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")
        print(f"F1-score : {m['f1_score']:.4f}")
        print("\nMatriz de confusión:")
        print(m["confusion_matrix"])

    def compare(self) -> None:
        """Imprime una tabla resumen de todos los modelos evaluados.
        Con un solo modelo funciona igual, solo que la tabla tendrá una fila."""
        print("\n=== Resumen de métricas ===")
        header = f"{'Modelo':<25}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}"
        print(header)
        print("-" * len(header))
        for name, m in self.results.items():
            print(
                f"{name:<25}{m['accuracy']:<10.4f}{m['precision']:<10.4f}"
                f"{m['recall']:<10.4f}{m['f1_score']:<10.4f}"
            )

    def plot_confusion_matrices(self, output_path: str) -> None:
        """Genera y guarda una imagen con las matrices de confusión de
        todos los modelos evaluados, una al lado de la otra."""
        n = len(self.results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, m) in zip(axes, self.results.items()):
            cm = m["confusion_matrix"]
            ax.imshow(cm, cmap="Blues")
            ax.set_title(name)
            ax.set_xlabel("Predicho")
            ax.set_ylabel("Real")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["<=50K", ">50K"])
            ax.set_yticklabels(["<=50K", ">50K"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
