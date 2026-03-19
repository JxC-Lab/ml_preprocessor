"""
Config loader — parses YAML or JSON pipeline configuration files
and instantiates the appropriate transformers.
"""

from __future__ import annotations
import json
from pathlib import Path
import yaml

from .transformers import TRANSFORMER_REGISTRY
from .transformers.base import BaseTransformer


def load_config(path: str | Path) -> dict:
    """Load a YAML or JSON config file and return the raw dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            cfg = yaml.safe_load(f)
        elif path.suffix == ".json":
            cfg = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    return cfg


def build_pipeline_from_config(cfg: dict) -> list[BaseTransformer]:
    """
    Given a parsed config dict, instantiate transformers in order.

    Expected config structure:
    ```yaml
    pipeline:
      - step: missing
        columns: [age, salary]
        strategy: mean
      - step: encoding
        columns: [city]
        method: onehot
    ```
    """
    steps_cfg = cfg.get("pipeline", [])
    if not steps_cfg:
        raise ValueError("Config must contain a non-empty 'pipeline' key.")

    transformers = []
    for i, step_cfg in enumerate(steps_cfg):
        step_cfg = dict(step_cfg)  # copy to avoid mutating original
        step_name = step_cfg.pop("step", None)

        if step_name is None:
            raise ValueError(f"Step #{i+1} is missing the 'step' key.")

        cls = TRANSFORMER_REGISTRY.get(step_name)
        if cls is None:
            available = list(TRANSFORMER_REGISTRY.keys())
            raise ValueError(
                f"Unknown step '{step_name}' at position #{i+1}. "
                f"Available: {available}"
            )

        try:
            transformer = cls(**step_cfg)
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for step '{step_name}' at position #{i+1}: {e}"
            )

        transformers.append(transformer)

    return transformers
