"""Smoke tests for the raw dataset and the full mineable-table pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from spc_module.config import RAW_DATA_DIR, TARGET_COLUMN
from spc_module.eda.loader import CSVDataLoader
from spc_module.preprocessing.builder import MineableTableBuilder
from spc_module.preprocessing.cleaning import (
    CleaningPipeline,
    DropColumnsStep,
    DuplicateRowsDropper,
    ModeImputer,
)
from spc_module.preprocessing.encoding import BinaryTargetEncoder, OneHotCategoricalEncoder
from spc_module.preprocessing.splitting import DatasetSplitter


@pytest.fixture
def raw_dataset_path():
    path = RAW_DATA_DIR / "salary.csv"
    if not path.exists():
        pytest.skip(f"Dataset crudo no encontrado en {path}")
    return path


def test_raw_dataset_loads_with_expected_schema(raw_dataset_path):
    loader = CSVDataLoader(file_path=raw_dataset_path)
    dataframe = loader.load()

    assert not dataframe.empty
    assert TARGET_COLUMN in dataframe.columns
    assert set(dataframe[TARGET_COLUMN].unique()) == {"<=50K", ">50K"}


def test_mineable_table_pipeline_end_to_end(raw_dataset_path):
    loader = CSVDataLoader(file_path=raw_dataset_path)
    builder = MineableTableBuilder(
        loader=loader,
        cleaning_pipeline=CleaningPipeline(
            steps=[
                DropColumnsStep(columns=["fnlwgt", "education"]),
                ModeImputer(columns=["workclass", "occupation", "native-country"]),
                DuplicateRowsDropper(),
            ]
        ),
        target_encoder=BinaryTargetEncoder(column=TARGET_COLUMN, positive_label=">50K"),
        categorical_encoder=OneHotCategoricalEncoder(exclude=[TARGET_COLUMN]),
        splitter=DatasetSplitter(test_size=0.2, random_state=42),
        target_column=TARGET_COLUMN,
    )

    result = builder.build()

    # La tabla minable no debe contener columnas de texto ni valores nulos.
    assert result.x_train.select_dtypes(include="object").empty
    assert result.x_train.isna().sum().sum() == 0
    assert result.x_test.isna().sum().sum() == 0

    # No debe filtrarse la variable objetivo hacia las features.
    assert TARGET_COLUMN not in result.x_train.columns

    # Las proporciones de train/test deben respetar test_size=0.2.
    total = len(result.x_train) + len(result.x_test)
    assert result.x_test.shape[0] / total == pytest.approx(0.2, abs=0.01)
