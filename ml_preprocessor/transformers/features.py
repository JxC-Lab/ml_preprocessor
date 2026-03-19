"""
FeatureEngineer — creates new features from existing ones.

Capabilities:
  - interactions : polynomial/product features between column pairs
  - binning      : discretize continuous variables into bins
  - dates        : extract components from datetime columns
  - aggregations : group-level statistics (mean, std, count, etc.)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from .base import BaseTransformer


class FeatureEngineer(BaseTransformer):
    name = "features"

    def __init__(
        self,
        columns: list[str] | None = None,
        interactions: list[list[str]] | None = None,
        binning: list[dict] | None = None,
        dates: list[dict] | None = None,
        aggregations: list[dict] | None = None,
    ):
        """
        Parameters
        ----------
        interactions : list of [col_a, col_b] pairs — creates col_a * col_b
        binning : list of dicts, each with keys:
                  - column (str)
                  - bins   (list of numeric cut-points or int for equal-width)
                  - labels (list of str, optional)
                  - drop_original (bool, default False)
        dates : list of dicts, each with keys:
                - column (str)
                - extract (list): any of ['year','month','day','dayofweek',
                                  'hour','minute','is_weekend','quarter']
                - drop_original (bool, default False)
        aggregations : list of dicts, each with keys:
                  - group_by (str | list[str])
                  - agg_col  (str)
                  - func     (str | list): e.g. 'mean', ['mean','std']
                  - prefix   (str, optional)
        """
        super().__init__(columns)
        self.interactions = interactions or []
        self.binning = binning or []
        self.dates = dates or []
        self.aggregations = aggregations or []
        self._agg_frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # fit — only aggregations need to learn from training data
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        self._agg_frames = []
        for agg_cfg in self.aggregations:
            group_by = agg_cfg["group_by"]
            agg_col  = agg_cfg["agg_col"]
            func     = agg_cfg["func"]
            prefix   = agg_cfg.get("prefix", f"{agg_col}")

            grouped = df.groupby(group_by)[agg_col].agg(func)
            if isinstance(func, list):
                grouped.columns = [f"{prefix}_{f}" for f in func]
            else:
                grouped = grouped.rename(f"{prefix}_{func}")

            self._agg_frames.append((group_by, grouped.reset_index()))

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        df = df.copy()

        # 1. Interaction features
        for pair in self.interactions:
            if len(pair) != 2:
                raise ValueError(f"Each interaction must be a pair [col_a, col_b], got: {pair}")
            col_a, col_b = pair
            if col_a not in df.columns or col_b not in df.columns:
                continue
            df[f"{col_a}_x_{col_b}"] = df[col_a] * df[col_b]

        # 2. Binning
        for bin_cfg in self.binning:
            col    = bin_cfg["column"]
            bins   = bin_cfg["bins"]
            labels = bin_cfg.get("labels", None)
            drop   = bin_cfg.get("drop_original", False)

            if col not in df.columns:
                continue

            new_col = f"{col}_bin"
            df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
            if drop:
                df.drop(columns=[col], inplace=True)

        # 3. Date feature extraction
        for date_cfg in self.dates:
            col     = date_cfg["column"]
            extract = date_cfg.get("extract", ["year", "month", "day"])
            drop    = date_cfg.get("drop_original", False)

            if col not in df.columns:
                continue

            dt = pd.to_datetime(df[col], errors="coerce")

            extractor_map = {
                "year":       lambda s: s.dt.year,
                "month":      lambda s: s.dt.month,
                "day":        lambda s: s.dt.day,
                "dayofweek":  lambda s: s.dt.dayofweek,
                "hour":       lambda s: s.dt.hour,
                "minute":     lambda s: s.dt.minute,
                "quarter":    lambda s: s.dt.quarter,
                "is_weekend": lambda s: s.dt.dayofweek.isin([5, 6]).astype(int),
            }

            for component in extract:
                if component not in extractor_map:
                    raise ValueError(
                        f"Unknown date component '{component}'. "
                        f"Choose from: {list(extractor_map.keys())}"
                    )
                df[f"{col}_{component}"] = extractor_map[component](dt)

            if drop:
                df.drop(columns=[col], inplace=True)

        # 4. Aggregations (merge pre-fitted stats)
        for group_by, agg_df in self._agg_frames:
            df = df.merge(agg_df, on=group_by, how="left")

        return df
