"""Plotting utilities for the EDA.

Defines an abstract :class:`BaseVisualizer` (Open/Closed Principle:
new backends such as Plotly could be added by implementing this
interface, without touching :class:`~spc_module.eda.report.EDAReport`)
and a concrete Matplotlib/Seaborn implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BaseVisualizer(ABC):
    """Abstract interface for a component able to render and save EDA plots."""

    @abstractmethod
    def plot_numerical_distributions(
        self, dataframe: pd.DataFrame, columns: list[str], filename: str
    ) -> Path:
        """Plot histograms with KDE for the given numerical columns."""
        raise NotImplementedError

    @abstractmethod
    def plot_boxplots_by_target(
        self, dataframe: pd.DataFrame, columns: list[str], target: str, filename: str
    ) -> Path:
        """Plot one boxplot per numerical column, grouped by ``target``."""
        raise NotImplementedError

    @abstractmethod
    def plot_categorical_counts(
        self, dataframe: pd.DataFrame, columns: list[str], filename: str
    ) -> Path:
        """Plot bar charts of value counts for the given categorical columns."""
        raise NotImplementedError

    @abstractmethod
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, filename: str) -> Path:
        """Plot a heatmap of a correlation matrix."""
        raise NotImplementedError

    @abstractmethod
    def plot_target_balance(self, dataframe: pd.DataFrame, target: str, filename: str) -> Path:
        """Plot the class balance of the target variable."""
        raise NotImplementedError


class MatplotlibVisualizer(BaseVisualizer):
    """Render EDA plots with Matplotlib/Seaborn and save them to disk.

    Parameters
    ----------
    output_dir:
        Directory where generated figures (PNG) will be saved. It is
        created automatically if it does not exist.
    style:
        Seaborn style theme applied to every figure.
    dpi:
        Resolution used when saving figures.
    """

    def __init__(self, output_dir: Path | str, style: str = "whitegrid", dpi: int = 150) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        sns.set_style(style)

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        output_path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Figura guardada en: {output_path}")
        return output_path

    def plot_numerical_distributions(
        self, dataframe: pd.DataFrame, columns: list[str], filename: str
    ) -> Path:
        n_cols = 3
        n_rows = -(-len(columns) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for ax, column in zip(axes, columns):
            sns.histplot(dataframe[column], kde=True, ax=ax, color="#4C72B0")
            ax.set_title(f"Distribución de {column}")
        for ax in axes[len(columns) :]:
            ax.axis("off")
        return self._save(fig, filename)

    def plot_boxplots_by_target(
        self, dataframe: pd.DataFrame, columns: list[str], target: str, filename: str
    ) -> Path:
        n_cols = 3
        n_rows = -(-len(columns) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for ax, column in zip(axes, columns):
            sns.boxplot(
                data=dataframe,
                x=target,
                y=column,
                hue=target,
                ax=ax,
                palette="Set2",
                legend=False,
            )
            ax.set_title(f"{column} por {target}")
        for ax in axes[len(columns) :]:
            ax.axis("off")
        return self._save(fig, filename)

    def plot_categorical_counts(
        self, dataframe: pd.DataFrame, columns: list[str], filename: str
    ) -> Path:
        n_cols = 2
        n_rows = -(-len(columns) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        for ax, column in zip(axes, columns):
            order = dataframe[column].value_counts().index
            sns.countplot(data=dataframe, y=column, order=order, ax=ax, color="#55A868")
            ax.set_title(f"Frecuencia de {column}")
        for ax in axes[len(columns) :]:
            ax.axis("off")
        return self._save(fig, filename)

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlación (variables numéricas)")
        return self._save(fig, filename)

    def plot_target_balance(self, dataframe: pd.DataFrame, target: str, filename: str) -> Path:
        fig, ax = plt.subplots(figsize=(6, 5))
        counts = dataframe[target].value_counts()
        sns.barplot(
            x=counts.index,
            y=counts.values,
            hue=counts.index,
            ax=ax,
            palette="pastel",
            legend=False,
        )
        for i, value in enumerate(counts.values):
            ax.text(i, value, f"{value} ({value / counts.sum():.1%})", ha="center", va="bottom")
        ax.set_title(f"Balance de clases: {target}")
        ax.set_ylabel("Frecuencia")
        return self._save(fig, filename)
