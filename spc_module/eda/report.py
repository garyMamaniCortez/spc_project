"""High-level EDA orchestrator.

``EDAReport`` coordinates :class:`DataLoader`, :class:`DataQualityChecker`,
:class:`DataProfiler` and :class:`BaseVisualizer` to produce a full
exploratory analysis. Following the Dependency Inversion Principle,
it depends only on abstractions (``DataLoader``, ``BaseVisualizer``)
that are injected through the constructor, so the concrete CSV source
or plotting backend can be swapped without modifying this class.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import pandas as pd

from spc_module.eda.loader import DataLoader
from spc_module.eda.profiler import DataProfiler
from spc_module.eda.quality import DataQualityChecker
from spc_module.eda.visualizer import BaseVisualizer


class EDAReport:
    """Run a complete exploratory data analysis and persist its artifacts.

    Parameters
    ----------
    loader:
        Any object implementing :class:`DataLoader`.
    visualizer:
        Any object implementing :class:`BaseVisualizer`.
    target:
        Name of the target/label column, used for target-aware plots
        (boxplots by class, class balance).
    report_dir:
        Directory where the Markdown summary report will be written.
    """

    def __init__(
        self,
        loader: DataLoader,
        visualizer: BaseVisualizer,
        target: str = "salary",
        report_dir: Path | str = "reports",
    ) -> None:
        self._loader = loader
        self._visualizer = visualizer
        self._target = target
        self._report_dir = Path(report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)

        self.dataframe: pd.DataFrame | None = None
        self.quality: DataQualityChecker | None = None
        self.profiler: DataProfiler | None = None
        self._figures: dict[str, Path] = {}

    def run(self) -> pd.DataFrame:
        """Execute the full EDA pipeline and return the loaded dataset."""
        self.dataframe = self._loader.load()
        self.quality = DataQualityChecker(self.dataframe)
        self.profiler = DataProfiler(self.dataframe)

        self._generate_figures()
        self._write_markdown_report()

        logger.success("Análisis EDA completado.")
        return self.dataframe

    def _generate_figures(self) -> None:
        numeric_cols = self.profiler.numerical_columns
        categorical_cols = [c for c in self.profiler.categorical_columns if c != self._target]

        self._figures["distributions"] = self._visualizer.plot_numerical_distributions(
            self.dataframe, numeric_cols, "01_numerical_distributions.png"
        )
        self._figures["boxplots"] = self._visualizer.plot_boxplots_by_target(
            self.dataframe, numeric_cols, self._target, "02_boxplots_by_target.png"
        )
        self._figures["categorical"] = self._visualizer.plot_categorical_counts(
            self.dataframe, categorical_cols, "03_categorical_counts.png"
        )
        self._figures["correlation"] = self._visualizer.plot_correlation_heatmap(
            self.profiler.correlation_matrix(), "04_correlation_heatmap.png"
        )
        self._figures["target_balance"] = self._visualizer.plot_target_balance(
            self.dataframe, self._target, "05_target_balance.png"
        )

    def _write_markdown_report(self) -> None:
        summary = self.quality.summary()
        missing = self.quality.missing_values_report()
        outliers = self.profiler.outlier_summary()

        lines = [
            "# Reporte EDA — salary.csv (Adult Census Income)",
            "",
            "## 1. Resumen general",
            f"- Filas: **{summary['n_rows']}**",
            f"- Columnas: **{summary['n_columns']}**",
            f"- Filas duplicadas: **{summary['n_duplicate_rows']}**",
            (
                f"- Columnas con valores faltantes: "
                f"**{', '.join(summary['columns_with_missing']) or 'ninguna'}**"
            ),
            "",
            "## 2. Valores faltantes",
            missing.to_markdown() if not missing.empty else "No se encontraron valores nulos.",
            "",
            "## 3. Outliers (regla IQR)",
            outliers.to_markdown(),
            "",
            "## 4. Estadísticas numéricas",
            self.profiler.numerical_summary().to_markdown(),
            "",
            "## 5. Figuras generadas",
        ]
        lines += [f"- `{name}`: {path.name}" for name, path in self._figures.items()]

        report_path = self._report_dir / "eda_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.success(f"Reporte Markdown guardado en: {report_path}")
