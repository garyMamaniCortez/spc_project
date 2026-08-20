"""Train/test splitting of the mineable table.

Kept as its own single-responsibility class so the split strategy
(e.g. simple vs. stratified, or a future k-fold variant) can evolve
independently from cleaning and encoding.
"""

from __future__ import annotations

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Split a feature matrix and a target vector into train/test sets.

    Parameters
    ----------
    test_size:
        Proportion of the dataset reserved for the test split.
    random_state:
        Seed used to make the split reproducible.
    stratify:
        When ``True`` (default), the split preserves the class
        proportions of the target variable. This matters here because
        the ``salary`` target is imbalanced (~24% earn ``>50K``).
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return ``(X_train, X_test, y_train, y_test)``."""
        strat = target if self.stratify else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat,
        )
        logger.info(
            f"Split train/test -> train: {x_train.shape[0]} filas, "
            f"test: {x_test.shape[0]} filas (test_size={self.test_size}, "
            f"stratify={self.stratify})."
        )
        return x_train, x_test, y_train, y_test
