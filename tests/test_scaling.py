"""Unit tests for spc_module.preprocessing.scaling."""

from __future__ import annotations

import pandas as pd
import pytest

from spc_module.preprocessing.scaling import StandardNumericalScaler


@pytest.fixture
def encoded_dataframe() -> pd.DataFrame:
    """Mimics a table already one-hot encoded: numeric + dummy columns."""
    return pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "hours-per-week": [10, 20, 30, 40, 50, 60],
            "sex_Male": [1, 0, 1, 0, 1, 0],
            "workclass_Private": [0, 1, 1, 0, 1, 1],
        }
    )


class TestStandardNumericalScaler:
    def test_scales_only_the_given_numeric_columns(self, encoded_dataframe):
        scaler = StandardNumericalScaler(columns=["age", "hours-per-week"])
        result = scaler.fit_transform(encoded_dataframe)

        assert result["age"].mean() == pytest.approx(0.0, abs=1e-8)
        assert result["hours-per-week"].mean() == pytest.approx(0.0, abs=1e-8)

    def test_dummy_columns_are_left_untouched(self, encoded_dataframe):
        scaler = StandardNumericalScaler(columns=["age", "hours-per-week"])
        result = scaler.fit_transform(encoded_dataframe)

        pd.testing.assert_series_equal(
            result["sex_Male"], encoded_dataframe["sex_Male"]
        )
        pd.testing.assert_series_equal(
            result["workclass_Private"], encoded_dataframe["workclass_Private"]
        )

    def test_transform_reuses_train_fitted_statistics(self, encoded_dataframe):
        scaler = StandardNumericalScaler(columns=["age", "hours-per-week"])
        scaler.fit_transform(encoded_dataframe)

        test_df = encoded_dataframe.copy()
        test_df["age"] = [25, 35, 45, 55, 65, 75]
        result = scaler.transform(test_df)

        # Same fitted mean/std as train, so the test mean should NOT be ~0.
        assert result["age"].mean() != pytest.approx(0.0, abs=1e-8)

    def test_transform_before_fit_raises(self, encoded_dataframe):
        scaler = StandardNumericalScaler(columns=["age"])
        with pytest.raises(RuntimeError):
            scaler.transform(encoded_dataframe)

    def test_ignores_columns_not_present(self, encoded_dataframe):
        scaler = StandardNumericalScaler(columns=["age", "not_a_column"])
        result = scaler.fit_transform(encoded_dataframe)
        assert "not_a_column" not in result.columns
