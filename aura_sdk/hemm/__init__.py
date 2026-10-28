"""
aura_sdk.hemm
──────────────
Language: Python (wraps Rust core)
HEMM entropy feature extraction.

    from aura_sdk.hemm import HEMMExtractor, EntropyFeatures, normalise_joints
"""
from .extractor import HEMMExtractor, EntropyFeatures, normalise_joints

__all__ = ["HEMMExtractor", "EntropyFeatures", "normalise_joints"]
