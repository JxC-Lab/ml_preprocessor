"""
CategoricalEncoder — encodes categorical variables.

Supported methods:
  - onehot   : One-Hot Encoding (pd.get_dummies), drops first by default
  - label    : Label Encoding (maps categories to integers)
  - ordinal  : Ordinal Encoding with a user-defined order
  - frequency: Replace category with its frequency in training data
  - target   : Target Encoding (requires y during fit)
"""

from __future__ import annotations
import pandas as pd
from .base import BaseTransformer


class CategoricalEncoder(BaseTransformer):
    name = "encoding"

    SUPPORTED_METHODS = {"onehot", "label", "ordinal", "frequency", "target"}

    def __init__(
        self,
        columns: list[str] | None = None,
        method: str = "onehot",
        drop_first: bool = True,         # for onehot
        ordinal_order: dict | None = None,  # {col: [cat1, cat2, ...]}
        handle_unknown: str = "ignore",  # 'ignore' or 'error'
    ):
        super().__init__(columns)
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.drop_first = drop_first
        self.ordinal_order = ordinal_order or {}
        self.handle_unknown = handle_unknown
        self._label_maps: dict[str, dict] = {}
        self._frequency_maps: dict[str, dict] = {}
        self._target_maps: dict[str, dict] = {}

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalEncoder":
        cols = self._resolve_columns(df, dtype_filter=["object", "category"])

        if self.method == "label":
            for c in cols:
                categories = df[c].dropna().unique()
                self._label_maps[c] = {cat: i for i, cat in enumerate(sorted(categories, key=str))}

        elif self.method == "ordinal":
            for c in cols:
                order = self.ordinal_order.get(c)
                if order is None:
                    raise ValueError(
                        f"ordinal_order must be provided for column '{c}' when method='ordinal'."
                    )
                self._label_maps[c] = {cat: i for i, cat in enumerate(order)}

        elif self.method == "frequency":
            for c in cols:
                freq = df[c].value_counts(normalize=True)
                self._frequency_maps[c] = freq.to_dict()

        elif self.method == "target":
            if y is None:
                raise ValueError("y (target Series) must be provided for target encoding.")
            tmp = df[cols].copy()
            tmp["__target__"] = y.values
            for c in cols:
                self._target_maps[c] = tmp.groupby(c)["__target__"].mean().to_dict()

        self._cols_to_encode = cols
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()
        cols = self._cols_to_encode

        if self.method == "onehot":
            df = pd.get_dummies(df, columns=cols, drop_first=self.drop_first, dtype=int)

        elif self.method in {"label", "ordinal"}:
            for c in cols:
                if c not in df.columns:
                    continue
                mapping = self._label_maps[c]
                if self.handle_unknown == "ignore":
                    df[c] = df[c].map(mapping)
                else:
                    unknown = set(df[c].dropna().unique()) - set(mapping.keys())
                    if unknown:
                        raise ValueError(f"Unknown categories in column '{c}': {unknown}")
                    df[c] = df[c].map(mapping)

        elif self.method == "frequency":
            for c in cols:
                if c not in df.columns:
                    continue
                df[c] = df[c].map(self._frequency_maps[c])

        elif self.method == "target":
            for c in cols:
                if c not in df.columns:
                    continue
                df[c] = df[c].map(self._target_maps[c])

        return df
