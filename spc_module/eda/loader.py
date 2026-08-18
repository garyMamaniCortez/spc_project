"""Data loading utilities.

Defines an abstract :class:`DataLoader` interface
and a concrete :class:`CSVDataLoader` implementation
(this module is only responsible for reading raw data
into memory, not for cleaning or profiling it).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
import pandas as pd


class DataLoader(ABC):
    """Abstract interface for objects able to load a dataset.

    Any new data source (CSV, Parquet, SQL, API, ...) can implement
    this interface without requiring changes to the classes that
    consume it (Open/Closed Principle).
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return the raw dataset as a ``pandas.DataFrame``."""
        raise NotImplementedError


class CSVDataLoader(DataLoader):
    """Load a tabular dataset stored as a CSV file.

    Parameters
    ----------
    file_path:
        Path to the CSV file to be loaded.
    na_values:
        Extra tokens that must be interpreted as missing values. The
        ``salary.csv`` (Adult Census Income) dataset encodes missing
        categorical values as ``" ?"``, so this defaults to that token.
    strip_whitespace:
        When ``True`` (default), leading/trailing whitespace is
        stripped from string columns and column names, since the raw
        file has a leading space after every comma.
    """

    def __init__(
        self,
        file_path: Path | str,
        na_values: tuple[str, ...] = (" ?", "?"),
        strip_whitespace: bool = True,
    ) -> None:
        self.file_path = Path(file_path)
        self.na_values = list(na_values)
        self.strip_whitespace = strip_whitespace

    def load(self) -> pd.DataFrame:
        """Read the CSV file and return a cleaned ``DataFrame``.

        Returns
        -------
        pandas.DataFrame
            The raw dataset with normalized column names and
            whitespace-stripped string values.

        Raises
        ------
        FileNotFoundError
            If ``self.file_path`` does not exist.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.file_path}")

        logger.info(f"Cargando dataset desde: {self.file_path}")
        dataframe = pd.read_csv(self.file_path, na_values=self.na_values, skipinitialspace=True)
        dataframe.columns = [col.strip() for col in dataframe.columns]

        if self.strip_whitespace:
            dataframe = self._strip_string_columns(dataframe)

        logger.success(
            f"Dataset cargado correctamente: {dataframe.shape[0]} filas, "
            f"{dataframe.shape[1]} columnas."
        )
        return dataframe

    @staticmethod
    def _strip_string_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Strip surrounding whitespace from every object/string column."""
        str_columns = dataframe.select_dtypes(include="object").columns
        for column in str_columns:
            dataframe[column] = dataframe[column].str.strip()
        return dataframe
