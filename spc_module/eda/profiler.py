"""Descriptive statistical profiling of numerical and categorical data."""

from __future__ import annotations

import pandas as pd


class DataProfiler:
    """Compute descriptive statistics for numerical and categorical columns.

    Parameters
    ----------
    dataframe:
        Dataset to profile. The object is never mutated.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._df = dataframe

    @property
    def numerical_columns(self) -> list[str]:
        """Names of the numerical columns in the dataset."""
        return self._df.select_dtypes(include="number").columns.tolist()

    @property
    def categorical_columns(self) -> list[str]:
        """Names of the categorical (object) columns in the dataset."""
        return self._df.select_dtypes(include="object").columns.tolist()

    def numerical_summary(self) -> pd.DataFrame:
        """Return descriptive statistics (mean, std, quartiles, skew...)."""
        summary = self._df[self.numerical_columns].describe().T
        summary["skew"] = self._df[self.numerical_columns].skew()
        summary["kurtosis"] = self._df[self.numerical_columns].kurtosis()
        return summary.round(2)

    def categorical_summary(self, top_n: int = 5) -> dict[str, pd.Series]:
        """Return the top-N most frequent categories per categorical column.

        Parameters
        ----------
        top_n:
            Number of most frequent categories to keep per column.
        """
        return {
            column: self._df[column].value_counts(dropna=True).head(top_n)
            for column in self.categorical_columns
        }

    def outliers_iqr(self, column: str) -> pd.DataFrame:
        """Return rows considered outliers in ``column`` using the IQR rule.

        Parameters
        ----------
        column:
            Name of a numerical column.
        """
        q1, q3 = self._df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (self._df[column] < lower_bound) | (self._df[column] > upper_bound)
        return self._df.loc[mask, [column]]

    def outlier_summary(self) -> pd.DataFrame:
        """Return the count and percentage of IQR outliers per numeric column."""
        rows = []
        for column in self.numerical_columns:
            n_outliers = len(self.outliers_iqr(column))
            rows.append(
                {
                    "column": column,
                    "n_outliers": n_outliers,
                    "pct_outliers": round(n_outliers / len(self._df) * 100, 2),
                }
            )
        return pd.DataFrame(rows).set_index("column").sort_values("n_outliers", ascending=False)

    def correlation_matrix(self, method: str = "pearson") -> pd.DataFrame:
        """Return the correlation matrix of numerical columns.

        Parameters
        ----------
        method:
            Correlation method: ``"pearson"``, ``"spearman"`` or ``"kendall"``.
        """
        return self._df[self.numerical_columns].corr(method=method).round(2)
