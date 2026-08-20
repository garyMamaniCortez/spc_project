"""Encoders used to turn the cleaned dataset into a model-ready table.

Two independent responsibilities are kept apart (Single Responsibility
Principle): :class:`CategoricalEncoder` transforms the explanatory
categorical variables, while :class:`TargetEncoder` transforms the
label column. Both are defined as abstractions (Dependency Inversion
Principle) so alternative encoding strategies (ordinal encoding,
target/mean encoding, multiclass labels, ...) can be plugged into
:class:`~spc_module.preprocessing.builder.MineableTableBuilder` later
without modifying it (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loguru import logger
import pandas as pd


class CategoricalEncoder(ABC):
    """Abstract interface for encoders of explanatory categorical columns."""

    @abstractmethod
    def encode(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a new ``DataFrame`` with categorical columns encoded."""
        raise NotImplementedError


class OneHotCategoricalEncoder(CategoricalEncoder):
    """One-hot encode every categorical (string) column except the target.

    Parameters
    ----------
    exclude:
        Columns that must never be one-hot encoded (typically the
        target column, already handled by a :class:`TargetEncoder`).
    drop_first:
        When ``True`` (default), drops the first category of every
        variable to avoid the dummy-variable trap (perfect
        multicollinearity), which matters for linear/logistic models.
        Tree-based models are insensitive to this choice, so it is
        exposed as a parameter (and as a CLI flag in ``features.py``)
        instead of being hard-coded.
    dtype:
        Dtype used for the generated dummy columns.
    """

    def __init__(
        self,
        exclude: list[str] | None = None,
        drop_first: bool = True,
        dtype: type = int,
    ) -> None:
        self.exclude = exclude or []
        self.drop_first = drop_first
        self.dtype = dtype
        self.categorical_columns_: list[str] = []
        self.encoded_columns_: list[str] = []

    def encode(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self.categorical_columns_ = [
            col
            for col in dataframe.select_dtypes(include=["object", "string"]).columns
            if col not in self.exclude
        ]
        logger.info(
            f"Aplicando one-hot encoding a {len(self.categorical_columns_)} "
            f"columnas categóricas: {self.categorical_columns_} "
            f"(drop_first={self.drop_first})."
        )
        encoded = pd.get_dummies(
            dataframe,
            columns=self.categorical_columns_,
            drop_first=self.drop_first,
            dtype=self.dtype,
        )
        self.encoded_columns_ = [c for c in encoded.columns if c not in dataframe.columns]
        logger.success(f"Generadas {len(self.encoded_columns_)} columnas dummy.")
        return encoded


class TargetEncoder(ABC):
    """Abstract interface for encoders of the label/target column."""

    @abstractmethod
    def encode(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a new ``DataFrame`` with the target column encoded."""
        raise NotImplementedError


class BinaryTargetEncoder(TargetEncoder):
    """Encode a binary text target as ``0``/``1``.

    Used to turn ``salary`` (``"<=50K"`` / ``">50K"``) into a numeric
    label suitable for classification algorithms.

    Parameters
    ----------
    column:
        Name of the target column (e.g. ``"salary"``).
    positive_label:
        Value that must be mapped to ``1``. Every other value is
        mapped to ``0``.
    output_column:
        Name of the resulting numeric column. When ``None`` (default)
        the original column is overwritten in place.
    """

    def __init__(
        self,
        column: str = "salary",
        positive_label: str = ">50K",
        output_column: str | None = None,
    ) -> None:
        self.column = column
        self.positive_label = positive_label
        self.output_column = output_column or column

    def encode(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.copy()
        normalized = dataframe[self.column].astype(str).str.strip()
        dataframe[self.output_column] = (normalized == self.positive_label).astype(int)
        if self.output_column != self.column:
            dataframe = dataframe.drop(columns=[self.column])

        distribution = dataframe[self.output_column].value_counts(normalize=True).round(3)
        logger.info(
            f"Variable objetivo '{self.column}' binarizada -> '{self.output_column}' "
            f"(1 = '{self.positive_label}'). Distribución: {distribution.to_dict()}"
        )
        return dataframe
