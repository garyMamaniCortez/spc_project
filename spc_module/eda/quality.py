"""Data quality checks: missing values, duplicates and cardinality.

Kept isolated from :mod:`spc_module.eda.profiler` (Single
Responsibility Principle): this module only answers "is the data
clean?", while the profiler answers "what does the data look like?".
"""

from __future__ import annotations

import pandas as pd


class DataQualityChecker:
    """Compute data-quality indicators for a given ``DataFrame``.

    Parameters
    ----------
    dataframe:
        Dataset to be analyzed. The object is never mutated.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self._df = dataframe

    def missing_values_report(self) -> pd.DataFrame:
        """Return count and percentage of missing values per column."""
        total = self._df.isna().sum()
        percentage = (total / len(self._df) * 100).round(2)
        report = pd.DataFrame({"missing_count": total, "missing_pct": percentage})
        return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)

    def duplicate_rows_count(self) -> int:
        """Return the number of fully duplicated rows."""
        return int(self._df.duplicated().sum())

    def cardinality_report(self) -> pd.DataFrame:
        """Return the number of unique values per categorical column."""
        cat_columns = self._df.select_dtypes(include="object").columns
        cardinality = {col: self._df[col].nunique(dropna=True) for col in cat_columns}
        return pd.Series(cardinality, name="unique_values").sort_values(ascending=False).to_frame()

    def constant_columns(self) -> list[str]:
        """Return columns that have a single unique value (zero variance)."""
        return [col for col in self._df.columns if self._df[col].nunique(dropna=True) <= 1]

    def summary(self) -> dict[str, object]:
        """Return a compact dictionary summarizing data-quality issues."""
        return {
            "n_rows": len(self._df),
            "n_columns": self._df.shape[1],
            "n_duplicate_rows": self.duplicate_rows_count(),
            "columns_with_missing": self.missing_values_report().index.tolist(),
            "constant_columns": self.constant_columns(),
        }
