"""CLI entry point to run the full EDA of salary.csv.

Usage
-----
    python -m spc_module.eda.runner
    python -m spc_module.eda.runner --input-path data/raw/salary.csv --target salary
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import typer

from spc_module.config import FIGURES_DIR, RAW_DATA_DIR, REPORTS_DIR
from spc_module.eda.loader import CSVDataLoader
from spc_module.eda.report import EDAReport
from spc_module.eda.visualizer import MatplotlibVisualizer

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "salary.csv",
    figures_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORTS_DIR,
    target: str = "salary",
) -> None:
    """Run the exploratory data analysis and save figures + Markdown report."""
    logger.info("Iniciando análisis exploratorio de datos (EDA)...")

    loader = CSVDataLoader(file_path=input_path)
    visualizer = MatplotlibVisualizer(output_dir=figures_dir)
    eda = EDAReport(loader=loader, visualizer=visualizer, target=target, report_dir=report_dir)
    eda.run()

    logger.success("EDA finalizado. Revisa 'reports/eda_report.md' y 'reports/figures/'.")


if __name__ == "__main__":
    app()
