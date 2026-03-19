"""
PreprocessingPipeline — orchestrates a sequence of transformers.

Usage (Python API):
    pipeline = PreprocessingPipeline([
        MissingValueHandler(columns=["age"], strategy="median"),
        CategoricalEncoder(columns=["city"], method="onehot"),
        FeatureScaler(method="standard"),
    ])
    df_clean = pipeline.fit_transform(df_train)
    df_test_clean = pipeline.transform(df_test)

Usage (from config):
    pipeline = PreprocessingPipeline.from_config("config.yaml")
    df_clean = pipeline.fit_transform(df)
"""

from __future__ import annotations
import time
import pandas as pd
from pathlib import Path

from .transformers.base import BaseTransformer
from .utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:

    def __init__(self, steps: list[BaseTransformer]):
        if not steps:
            raise ValueError("Pipeline must contain at least one transformer.")
        self.steps = steps
        self._fitted = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: str | Path) -> "PreprocessingPipeline":
        """Build a pipeline directly from a YAML or JSON config file."""
        from .config import load_config, build_pipeline_from_config
        cfg = load_config(config_path)
        steps = build_pipeline_from_config(cfg)
        logger.info(f"Pipeline loaded from config: {config_path} ({len(steps)} steps)")
        return cls(steps)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        logger.info(f"Fitting pipeline on DataFrame shape={df.shape}")
        for i, step in enumerate(self.steps):
            t0 = time.perf_counter()
            step.fit(df)
            # apply transform in-place so subsequent steps see updated data
            df = step.transform(df)
            elapsed = time.perf_counter() - t0
            logger.info(f"  [{i+1}/{len(self.steps)}] {step.__class__.__name__} fitted in {elapsed:.3f}s")
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform().")
        logger.info(f"Transforming DataFrame shape={df.shape}")
        for i, step in enumerate(self.steps):
            t0 = time.perf_counter()
            df = step.transform(df)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"  [{i+1}/{len(self.steps)}] {step.__class__.__name__} done in {elapsed:.3f}s"
                f" → shape={df.shape}"
            )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Fit+transform pipeline on DataFrame shape={df.shape}")
        for i, step in enumerate(self.steps):
            t0 = time.perf_counter()
            df = step.fit_transform(df)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"  [{i+1}/{len(self.steps)}] {step.__class__.__name__} fit_transform done"
                f" in {elapsed:.3f}s → shape={df.shape}"
            )
        self._fitted = True
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        """Serialize the fitted pipeline to disk (pickle)."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingPipeline":
        """Load a previously saved pipeline."""
        import pickle
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Pipeline loaded from {path}")
        return pipeline

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self):
        steps_repr = "\n  ".join(
            f"{i+1}. {s.__class__.__name__}" for i, s in enumerate(self.steps)
        )
        return f"PreprocessingPipeline(\n  {steps_repr}\n)"

    def summary(self) -> str:
        """Return a human-readable pipeline summary."""
        lines = [f"PreprocessingPipeline — {len(self.steps)} step(s)", ""]
        for i, step in enumerate(self.steps):
            params = {k: v for k, v in step.__dict__.items() if not k.startswith("_")}
            lines.append(f"  Step {i+1}: {step.__class__.__name__}")
            for k, v in params.items():
                lines.append(f"           {k}: {v}")
        return "\n".join(lines)
