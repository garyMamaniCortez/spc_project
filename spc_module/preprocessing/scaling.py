"""Numerical scaling, kept strictly separate from categorical encoding.

Bug fixed here: scaling must run on the original continuous numeric
columns only (``age``, ``education-num``, ``capital-gain``,
``capital-loss``, ``hours-per-week``). Applying ``StandardScaler`` to
the *whole* feature matrix **after** one-hot encoding also rescales
the 0/1 dummy columns, which:

- destroys their interpretation as "category present / absent",
- is unnecessary (dummies are already on the same 0/1 scale), and
- silently breaks column-level assumptions other steps may rely on.

:class:`StandardNumericalScaler` below only ever touches the columns
it is explicitly given, leaving every dummy column untouched -- no
matter whether scaling happens before or after one-hot encoding in
the pipeline that calls it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler


class NumericalScaler(ABC):
    """Abstract interface for scalers restricted to a fixed set of columns."""

    @abstractmethod
    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler on ``dataframe`` and return the scaled copy."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply an already-fitted scaler and return the scaled copy."""
        raise NotImplementedError


class StandardNumericalScaler(NumericalScaler):
    """Standardize (zero mean, unit variance) a fixed set of numeric columns.

    Every other column in the ``DataFrame`` (including one-hot dummy
    columns) is returned unchanged.

    Parameters
    ----------
    columns:
        Names of the continuous numeric columns to standardize. Only
        the subset of these that is actually present in the
        ``DataFrame`` passed to :meth:`fit_transform` is used; the
        resolved subset is remembered so :meth:`transform` (e.g. on a
        held-out test set) scales exactly the same columns.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self._scaler = StandardScaler()
        self._fitted_columns: list[str] = []

    def fit_transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        self._fitted_columns = [c for c in self.columns if c in dataframe.columns]
        logger.info(
            f"Escalando (StandardScaler) SOLO las columnas numéricas continuas: "
            f"{self._fitted_columns}. Las columnas dummy de one-hot no se tocan."
        )
        dataframe[self._fitted_columns] = self._scaler.fit_transform(
            dataframe[self._fitted_columns]
        )
        return dataframe

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted_columns:
            raise RuntimeError("El escalador no ha sido ajustado; llama a fit_transform primero.")
        dataframe = dataframe.copy()
        dataframe[self._fitted_columns] = self._scaler.transform(dataframe[self._fitted_columns])
        return dataframe
