"""
FeatureScaler — scales/normalizes numerical features.

Supported methods:
  - standard  : StandardScaler  (zero mean, unit variance)
  - minmax    : MinMaxScaler    (scales to [0, 1] or custom range)
  - robust    : RobustScaler    (median + IQR, robust to outliers)
  - maxabs    : MaxAbsScaler    (scales to [-1, 1])
  - log       : log1p transform (for skewed positive distributions)
  - quantile  : Quantile transform (maps to uniform or normal distribution)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseTransformer


class FeatureScaler(BaseTransformer):
    name = "scaling"

    SUPPORTED_METHODS = {"standard", "minmax", "robust", "maxabs", "log", "quantile"}

    def __init__(
        self,
        columns: list[str] | None = None,
        method: str = "standard",
        feature_range: tuple[float, float] = (0, 1),  # for minmax
        quantile_output: str = "uniform",              # 'uniform' or 'normal'
    ):
        super().__init__(columns)
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method
        self.feature_range = feature_range
        self.quantile_output = quantile_output
        self._params: dict[str, dict] = {}  # learned stats per column

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        cols = self._resolve_columns(df, dtype_filter="number")

        for c in cols:
            series = df[c].dropna()

            if self.method == "standard":
                self._params[c] = {"mean": series.mean(), "std": series.std()}

            elif self.method == "minmax":
                lo, hi = self.feature_range
                self._params[c] = {
                    "min": series.min(), "max": series.max(),
                    "lo": lo, "hi": hi,
                }

            elif self.method == "robust":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                self._params[c] = {"median": series.median(), "iqr": q3 - q1}

            elif self.method == "maxabs":
                self._params[c] = {"max_abs": series.abs().max()}

            elif self.method == "quantile":
                # store sorted reference distribution
                self._params[c] = {
                    "sorted_vals": np.sort(series.values),
                    "output": self.quantile_output,
                }

            # log needs no fitting

        self._cols_to_scale = cols
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()

        for c in self._cols_to_scale:
            if c not in df.columns:
                continue
            p = self._params.get(c, {})

            if self.method == "standard":
                std = p["std"] if p["std"] != 0 else 1
                df[c] = (df[c] - p["mean"]) / std

            elif self.method == "minmax":
                data_range = p["max"] - p["min"] or 1
                df[c] = (df[c] - p["min"]) / data_range
                df[c] = df[c] * (p["hi"] - p["lo"]) + p["lo"]

            elif self.method == "robust":
                iqr = p["iqr"] if p["iqr"] != 0 else 1
                df[c] = (df[c] - p["median"]) / iqr

            elif self.method == "maxabs":
                max_abs = p["max_abs"] if p["max_abs"] != 0 else 1
                df[c] = df[c] / max_abs

            elif self.method == "log":
                if (df[c] < 0).any():
                    raise ValueError(
                        f"Column '{c}' contains negative values — log transform not applicable."
                    )
                df[c] = np.log1p(df[c])

            elif self.method == "quantile":
                sorted_vals = p["sorted_vals"]
                ranks = pd.Series(df[c]).rank(method="average") / len(sorted_vals)
                if p["output"] == "uniform":
                    df[c] = ranks
                else:  # normal
                    from scipy.stats import norm
                    df[c] = norm.ppf(ranks.clip(1e-6, 1 - 1e-6))

        return df
