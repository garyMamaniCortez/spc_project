"""Cleaning steps for the raw salary dataset.

Each cleaning operation is implemented as an independent
:class:`CleaningStep` (Strategy pattern). ``CleaningPipeline`` composes
an ordered list of steps and applies them sequentially, so new
cleaning logic (e.g. an outlier capper) can be added later without
touching existing steps or the pipeline itself (Open/Closed
Principle). Each step also has a single, well-defined responsibility
(Single Responsibility Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger
import pandas as pd


class CleaningStep(ABC):
    """Abstract single-responsibility cleaning operation."""

    @abstractmethod
    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a new ``DataFrame`` with this cleaning step applied."""
        raise NotImplementedError


class DropColumnsStep(CleaningStep):
    """Remove columns that are not useful for modeling.

    Used here to drop ``fnlwgt`` (a census sampling weight, not a
    predictive feature) and ``education`` (redundant with the already
    numeric/ordinal ``education-num``).

    Parameters
    ----------
    columns:
        Names of the columns to drop. Columns absent from the
        dataframe are ignored instead of raising an error, so the
        step stays safe to reuse across slightly different schemas.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in self.columns if c in dataframe.columns]
        if existing:
            logger.info(f"Eliminando columnas no predictivas/redundantes: {existing}")
        return dataframe.drop(columns=existing)


class ModeImputer(CleaningStep):
    """Impute missing values of the given columns with their mode.

    Intended for the categorical columns that encode missing data as
    ``"?"`` in the raw file (``workclass``, ``occupation``,
    ``native-country``), which :class:`~spc_module.eda.loader.CSVDataLoader`
    already converts to ``NaN``.

    Parameters
    ----------
    columns:
        Names of the (typically categorical) columns to impute.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        for column in self.columns:
            if column not in dataframe.columns:
                continue
            n_missing = int(dataframe[column].isna().sum())
            if n_missing == 0:
                continue
            mode_value = dataframe[column].mode(dropna=True).iloc[0]
            dataframe[column] = dataframe[column].fillna(mode_value)
            logger.info(
                f"Imputados {n_missing} valores faltantes en '{column}' "
                f"con la moda: '{mode_value}'."
            )
        return dataframe


class DuplicateRowsDropper(CleaningStep):
    """Remove fully duplicated rows from the dataset."""

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        n_before = len(dataframe)
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
        n_removed = n_before - len(dataframe)
        if n_removed:
            logger.info(f"Eliminadas {n_removed} filas duplicadas.")
        return dataframe


class CleaningPipeline:
    """Apply an ordered sequence of :class:`CleaningStep` objects.

    Parameters
    ----------
    steps:
        Ordered list of cleaning steps to execute. Pass an empty list
        when the input is already clean (e.g. when reading data that
        ``dataset.py`` already processed).
    """

    def __init__(self, steps: list[CleaningStep]) -> None:
        self.steps = steps

    def run(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply every step in order and return the cleaned ``DataFrame``."""
        for step in self.steps:
            dataframe = step.apply(dataframe)
        return dataframe
