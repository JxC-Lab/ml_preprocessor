"""
MissingValueHandler — handles NaN imputation strategies.

Supported strategies:
  - mean         : replace with column mean (numeric)
  - median       : replace with column median (numeric)
  - most_frequent: replace with most common value (any dtype)
  - constant     : replace with a fixed value (fill_value param)
  - drop_rows    : drop rows that contain NaN in the specified columns
  - drop_cols    : drop columns that exceed a NaN threshold
"""

from __future__ import annotations
import pandas as pd
from .base import BaseTransformer


class MissingValueHandler(BaseTransformer):
    name = "missing"

    SUPPORTED_STRATEGIES = {"mean", "median", "most_frequent", "constant", "drop_rows", "drop_cols"}

    def __init__(
        self,
        columns: list[str] | None = None,
        strategy: str = "mean",
        fill_value=None,
        drop_threshold: float = 0.5,  # for drop_cols: max ratio of NaNs allowed
    ):
        super().__init__(columns)
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {self.SUPPORTED_STRATEGIES}"
            )
        self.strategy = strategy
        self.fill_value = fill_value
        self.drop_threshold = drop_threshold
        self._fill_map: dict = {}  # column -> fill value learned at fit time

    def fit(self, df: pd.DataFrame) -> "MissingValueHandler":
        cols = self._resolve_columns(df)

        if self.strategy == "mean":
            self._fill_map = {c: df[c].mean() for c in cols if pd.api.types.is_numeric_dtype(df[c])}

        elif self.strategy == "median":
            self._fill_map = {c: df[c].median() for c in cols if pd.api.types.is_numeric_dtype(df[c])}

        elif self.strategy == "most_frequent":
            self._fill_map = {c: df[c].mode().iloc[0] for c in cols if not df[c].mode().empty}

        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value must be set when strategy='constant'.")
            self._fill_map = {c: self.fill_value for c in cols}

        elif self.strategy == "drop_cols":
            nan_ratios = df[cols].isnull().mean()
            self._cols_to_drop = nan_ratios[nan_ratios > self.drop_threshold].index.tolist()

        # drop_rows needs no fitting
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        cols = self._resolve_columns(df)

        if self.strategy in {"mean", "median", "most_frequent", "constant"}:
            df.fillna({c: v for c, v in self._fill_map.items() if c in df.columns}, inplace=True)

        elif self.strategy == "drop_rows":
            df.dropna(subset=cols, inplace=True)
            df.reset_index(drop=True, inplace=True)

        elif self.strategy == "drop_cols":
            df.drop(columns=[c for c in self._cols_to_drop if c in df.columns], inplace=True)

        return df
