"""Load the raw salary dataset and produce a clean, encoding-ready version.

Responsibility boundary (cookiecutter-data-science convention): this
module only *cleans* the raw data (drops non-predictive/redundant
columns, imputes missing categories, removes duplicates) and writes
the result to ``data/interim``. Encoding and train/test splitting
belong to ``features.py``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import typer

from spc_module.config import (
    CATEGORICAL_COLUMNS_TO_IMPUTE,
    COLUMNS_TO_DROP,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from spc_module.eda.loader import CSVDataLoader
from spc_module.preprocessing.cleaning import (
    CleaningPipeline,
    DropColumnsStep,
    DuplicateRowsDropper,
    ModeImputer,
)

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "salary.csv",
    output_path: Path = INTERIM_DATA_DIR / "salary_clean.csv",
) -> None:
    """Clean the raw ``salary.csv`` and persist it to ``data/interim``."""
    logger.info("Procesando dataset crudo (salary.csv)...")

    loader = CSVDataLoader(file_path=input_path)
    pipeline = CleaningPipeline(
        steps=[
            DropColumnsStep(columns=COLUMNS_TO_DROP),
            ModeImputer(columns=CATEGORICAL_COLUMNS_TO_IMPUTE),
            DuplicateRowsDropper(),
        ]
    )

    dataframe = loader.load()
    clean_dataframe = pipeline.run(dataframe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_dataframe.to_csv(output_path, index=False)

    logger.success(
        f"Dataset limpio guardado en: {output_path} "
        f"({clean_dataframe.shape[0]} filas, {clean_dataframe.shape[1]} columnas)."
    )


if __name__ == "__main__":
    app()
