"""Build the mineable table: one-hot encoding + binary target + train/test split.

Reads the already-clean dataset produced by ``dataset.py``
(``data/interim/salary_clean.csv``), binarizes the ``salary`` target,
one-hot encodes the remaining categorical variables and splits the
result into train/test sets, persisting everything under
``data/processed`` ready to be consumed by
``spc_module.modeling.train`` / ``spc_module.modeling.predict``.
"""

from __future__ import annotations

from pathlib import Path
import yaml

from loguru import logger
import typer

from spc_module.config import (
    INTERIM_DATA_DIR,
    POSITIVE_LABEL,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    TARGET_COLUMN,
)
from spc_module.eda.loader import CSVDataLoader
from spc_module.preprocessing.builder import MineableTableBuilder
from spc_module.preprocessing.cleaning import CleaningPipeline
from spc_module.preprocessing.encoding import BinaryTargetEncoder, OneHotCategoricalEncoder
from spc_module.preprocessing.splitting import DatasetSplitter

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "salary_clean.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    test_features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    test_labels_path: Path = PROCESSED_DATA_DIR / "test_labels.csv",
    target: str = TARGET_COLUMN,
    positive_label: str = POSITIVE_LABEL,
    drop_first: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Generate the mineable table and its train/test split."""
    logger.info("Generando la tabla minable (one-hot encoding + target binario)...")

    params_file = PROJ_ROOT / "params.yaml"
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        feat_cfg = cfg.get("features", {})
        if "test_size" in feat_cfg:
            test_size = float(feat_cfg["test_size"])
        if "random_state" in feat_cfg:
            random_state = int(feat_cfg["random_state"])
        if "drop_first" in feat_cfg:
            drop_first = bool(feat_cfg["drop_first"])

    # El archivo interim ya fue limpiado por dataset.py: no se vuelve a
    # tratar "?" como NaN aquí (na_values vacío) y la CleaningPipeline
    # se pasa vacía porque no hay nada más que limpiar.
    loader = CSVDataLoader(file_path=input_path, na_values=())

    builder = MineableTableBuilder(
        loader=loader,
        cleaning_pipeline=CleaningPipeline(steps=[]),
        target_encoder=BinaryTargetEncoder(column=target, positive_label=positive_label),
        categorical_encoder=OneHotCategoricalEncoder(exclude=[target], drop_first=drop_first),
        splitter=DatasetSplitter(test_size=test_size, random_state=random_state),
        target_column=target,
    )
    result = builder.build()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    result.x_train.to_csv(features_path, index=False)
    result.y_train.to_csv(labels_path, index=False)
    result.x_test.to_csv(test_features_path, index=False)
    result.y_test.to_csv(test_labels_path, index=False)

    logger.success(
        f"Tabla minable guardada en '{PROCESSED_DATA_DIR}': "
        f"{features_path.name}, {labels_path.name}, "
        f"{test_features_path.name}, {test_labels_path.name}"
    )


if __name__ == "__main__":
    app()
