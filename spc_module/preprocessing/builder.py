"""High-level orchestrator that assembles the mineable table.

``MineableTableBuilder`` follows the same orchestration pattern as
:class:`spc_module.eda.report.EDAReport`: it depends only on
abstractions (``DataLoader``, ``CategoricalEncoder``, ``TargetEncoder``)
injected through its constructor (Dependency Inversion Principle), so
any collaborator (data source, encoding strategy, split strategy) can
be swapped or mocked in tests without changing this class at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
import pandas as pd

from spc_module.eda.loader import DataLoader
from spc_module.preprocessing.cleaning import CleaningPipeline
from spc_module.preprocessing.encoding import CategoricalEncoder, TargetEncoder
from spc_module.preprocessing.splitting import DatasetSplitter


@dataclass
class MineableTableResult:
    """Container for the artifacts produced by :class:`MineableTableBuilder`."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class MineableTableBuilder:
    """Build the model-ready ("mineable") table from a raw/interim dataset.

    Parameters
    ----------
    loader:
        Any object implementing :class:`~spc_module.eda.loader.DataLoader`.
    cleaning_pipeline:
        Sequence of cleaning steps applied before encoding. Pass an
        empty :class:`~spc_module.preprocessing.cleaning.CleaningPipeline`
        when the input was already cleaned upstream (e.g. by
        ``dataset.py``).
    target_encoder:
        Encoder used to binarize the label column.
    categorical_encoder:
        Encoder used to one-hot encode explanatory categorical columns.
    splitter:
        Train/test splitter applied to the fully encoded table.
    target_column:
        Name of the (already-encoded) target column, popped from the
        feature matrix before splitting.
    """

    def __init__(
        self,
        loader: DataLoader,
        cleaning_pipeline: CleaningPipeline,
        target_encoder: TargetEncoder,
        categorical_encoder: CategoricalEncoder,
        splitter: DatasetSplitter,
        target_column: str = "salary",
    ) -> None:
        self._loader = loader
        self._cleaning_pipeline = cleaning_pipeline
        self._target_encoder = target_encoder
        self._categorical_encoder = categorical_encoder
        self._splitter = splitter
        self._target_column = target_column

    def build(self) -> MineableTableResult:
        """Run the full pipeline and return the train/test features and labels."""
        raw = self._loader.load()
        clean = self._cleaning_pipeline.run(raw)
        target_encoded = self._target_encoder.encode(clean)
        fully_encoded = self._categorical_encoder.encode(target_encoded)

        target = fully_encoded.pop(self._target_column)
        x_train, x_test, y_train, y_test = self._splitter.split(fully_encoded, target)

        logger.success(
            f"Tabla minable construida: {fully_encoded.shape[1]} features, "
            f"{len(target)} filas totales."
        )
        return MineableTableResult(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
