"""Unit tests for the spc_module.preprocessing package."""

from __future__ import annotations

import pandas as pd
import pytest

from spc_module.preprocessing.cleaning import (
    CleaningPipeline,
    DropColumnsStep,
    DuplicateRowsDropper,
    ModeImputer,
)
from spc_module.preprocessing.encoding import BinaryTargetEncoder, OneHotCategoricalEncoder
from spc_module.preprocessing.splitting import DatasetSplitter


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 40, 35, 50, 28, 60],
            "fnlwgt": [1, 2, 3, 4, 5, 6],
            "education": ["Bachelors", "HS-grad", "Masters", "HS-grad", "Bachelors", "Doctorate"],
            "workclass": ["Private", None, "Private", "Self-emp", "Private", "Private"],
            "sex": ["Male", "Female", "Female", "Male", "Male", "Female"],
            "salary": ["<=50K", ">50K", "<=50K", ">50K", "<=50K", ">50K"],
        }
    )


class TestDropColumnsStep:
    def test_drops_existing_columns(self, sample_dataframe):
        result = DropColumnsStep(columns=["fnlwgt", "education"]).apply(sample_dataframe)
        assert "fnlwgt" not in result.columns
        assert "education" not in result.columns
        assert "age" in result.columns

    def test_ignores_missing_columns(self, sample_dataframe):
        result = DropColumnsStep(columns=["not_a_column"]).apply(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)


class TestModeImputer:
    def test_imputes_missing_with_mode(self, sample_dataframe):
        result = ModeImputer(columns=["workclass"]).apply(sample_dataframe)
        assert result["workclass"].isna().sum() == 0
        assert result.loc[1, "workclass"] == "Private"

    def test_does_not_mutate_input(self, sample_dataframe):
        ModeImputer(columns=["workclass"]).apply(sample_dataframe)
        assert sample_dataframe["workclass"].isna().sum() == 1


class TestDuplicateRowsDropper:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = DuplicateRowsDropper().apply(df)
        assert len(result) == 2


class TestCleaningPipeline:
    def test_runs_all_steps_in_order(self, sample_dataframe):
        pipeline = CleaningPipeline(
            steps=[
                DropColumnsStep(columns=["fnlwgt", "education"]),
                ModeImputer(columns=["workclass"]),
            ]
        )
        result = pipeline.run(sample_dataframe)
        assert "fnlwgt" not in result.columns
        assert result["workclass"].isna().sum() == 0

    def test_empty_pipeline_is_a_noop(self, sample_dataframe):
        result = CleaningPipeline(steps=[]).run(sample_dataframe)
        pd.testing.assert_frame_equal(result, sample_dataframe)


class TestBinaryTargetEncoder:
    def test_encodes_positive_and_negative_labels(self, sample_dataframe):
        encoder = BinaryTargetEncoder(column="salary", positive_label=">50K")
        result = encoder.encode(sample_dataframe)
        assert set(result["salary"].unique()) <= {0, 1}
        assert result["salary"].tolist() == [0, 1, 0, 1, 0, 1]

    def test_can_write_to_a_different_output_column(self, sample_dataframe):
        encoder = BinaryTargetEncoder(
            column="salary", positive_label=">50K", output_column="high_income"
        )
        result = encoder.encode(sample_dataframe)
        assert "salary" not in result.columns
        assert "high_income" in result.columns


class TestOneHotCategoricalEncoder:
    def test_encodes_all_object_columns_except_excluded(self, sample_dataframe):
        encoder = OneHotCategoricalEncoder(exclude=["salary"], drop_first=False)
        result = encoder.encode(sample_dataframe)
        assert not any(result[c].dtype == "object" for c in result.columns if c != "salary")
        assert "sex_Male" in result.columns
        assert "sex_Female" in result.columns
        assert "salary" in result.columns  # excluded column untouched

    def test_drop_first_removes_one_dummy_per_variable(self, sample_dataframe):
        encoder_full = OneHotCategoricalEncoder(exclude=["salary"], drop_first=False)
        encoder_dropped = OneHotCategoricalEncoder(exclude=["salary"], drop_first=True)
        full = encoder_full.encode(sample_dataframe)
        dropped = encoder_dropped.encode(sample_dataframe)
        assert full.shape[1] > dropped.shape[1]

    def test_no_missing_values_after_encoding(self, sample_dataframe):
        sample_dataframe.loc[1, "workclass"] = None
        result = OneHotCategoricalEncoder(exclude=["salary"]).encode(sample_dataframe)
        # NaN in a categorical column becomes an all-zero dummy row, not a NaN.
        assert result.isna().sum().sum() == 0


class TestDatasetSplitter:
    def test_split_sizes_and_stratification(self):
        features = pd.DataFrame({"x": range(100)})
        target = pd.Series([0] * 80 + [1] * 20)
        splitter = DatasetSplitter(test_size=0.2, random_state=42, stratify=True)
        x_train, x_test, y_train, y_test = splitter.split(features, target)

        assert len(x_train) == 80
        assert len(x_test) == 20
        assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)
