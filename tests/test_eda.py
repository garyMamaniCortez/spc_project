"""Unit tests for the spc_module.eda package."""

from pathlib import Path

import pandas as pd
import pytest

from spc_module.eda.loader import CSVDataLoader
from spc_module.eda.profiler import DataProfiler
from spc_module.eda.quality import DataQualityChecker


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a small CSV file mimicking salary.csv's quirks."""
    content = (
        "age, workclass, salary\n"
        "39, State-gov, <=50K\n"
        "50, ?, >50K\n"
        "39, State-gov, <=50K\n"
    )
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_csv_data_loader_strips_whitespace_and_parses_missing(sample_csv: Path) -> None:
    loader = CSVDataLoader(file_path=sample_csv)
    df = loader.load()

    assert list(df.columns) == ["age", "workclass", "salary"]
    assert df.loc[0, "workclass"] == "State-gov"
    assert df["workclass"].isna().sum() == 1


def test_csv_data_loader_raises_for_missing_file(tmp_path: Path) -> None:
    loader = CSVDataLoader(file_path=tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_data_quality_checker_detects_missing_and_duplicates(sample_csv: Path) -> None:
    df = CSVDataLoader(file_path=sample_csv).load()
    checker = DataQualityChecker(df)

    assert checker.duplicate_rows_count() == 1
    missing = checker.missing_values_report()
    assert "workclass" in missing.index
    assert missing.loc["workclass", "missing_count"] == 1


def test_data_profiler_numerical_summary_contains_expected_columns() -> None:
    df = pd.DataFrame({"age": [20, 30, 40, 50], "salary_num": [1, 2, 3, 100]})
    profiler = DataProfiler(df)

    summary = profiler.numerical_summary()
    assert "mean" in summary.columns
    assert "skew" in summary.columns
    assert set(profiler.numerical_columns) == {"age", "salary_num"}


def test_data_profiler_outliers_iqr_flags_extreme_value() -> None:
    df = pd.DataFrame({"value": [10, 11, 12, 13, 14, 1000]})
    profiler = DataProfiler(df)

    outliers = profiler.outliers_iqr("value")
    assert 1000 in outliers["value"].values
