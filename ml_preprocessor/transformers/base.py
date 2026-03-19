"""
Base class for all transformers in the ML Preprocessor framework.
Optionally wraps into a sklearn-compatible estimator.
"""

from __future__ import annotations
import pandas as pd
from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    """
    Abstract base for all preprocessing transformers.
    Exposes fit / transform / fit_transform interface.
    """

    name: str = "base"

    def __init__(self, columns: list[str] | None = None, **kwargs):
        self.columns = columns  # None = apply to all eligible columns
        self._is_fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """Learn parameters from training data."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformation to data."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling transform()."
            )

    def _resolve_columns(self, df: pd.DataFrame, dtype_filter=None) -> list[str]:
        """Return the list of columns to operate on."""
        if self.columns:
            missing = [c for c in self.columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in DataFrame: {missing}")
            return self.columns
        if dtype_filter:
            return df.select_dtypes(include=dtype_filter).columns.tolist()
        return df.columns.tolist()

    def to_sklearn(self):
        """
        Wrap this transformer as a sklearn-compatible object.
        Requires scikit-learn to be installed.
        """
        try:
            from sklearn.base import BaseEstimator, TransformerMixin
        except ImportError:
            raise ImportError(
                "scikit-learn is not installed. "
                "Run: pip install scikit-learn"
            )

        outer = self

        class SklearnWrapper(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                outer.fit(pd.DataFrame(X))
                return self

            def transform(self, X):
                return outer.transform(pd.DataFrame(X)).values

            def __repr__(self):
                return f"SklearnWrapper({outer.__class__.__name__})"

        return SklearnWrapper()
