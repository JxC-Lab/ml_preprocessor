from .missing import MissingValueHandler
from .encoding import CategoricalEncoder
from .scaling import FeatureScaler
from .features import FeatureEngineer

TRANSFORMER_REGISTRY = {
    MissingValueHandler.name: MissingValueHandler,
    CategoricalEncoder.name: CategoricalEncoder,
    FeatureScaler.name: FeatureScaler,
    FeatureEngineer.name: FeatureEngineer,
}

__all__ = [
    "MissingValueHandler",
    "CategoricalEncoder",
    "FeatureScaler",
    "FeatureEngineer",
    "TRANSFORMER_REGISTRY",
]
