"""
aura_sdk.training
──────────────────
Language: Python
Why: BNN training uses PyTorch/NumPy ecosystem.
     Runs once offline. Exports weights to JSON for Rust.

    from aura_sdk.training.train import (
        generate_synthetic_dataset,
        train_and_export,
        loso_cross_validate,
        EntropyRecord,
    )
"""
from .train import (
    EntropyRecord,
    generate_synthetic_dataset,
    train_and_export,
    loso_cross_validate,
    PurePythonBNN,
)

__all__ = [
    "EntropyRecord",
    "generate_synthetic_dataset",
    "train_and_export",
    "loso_cross_validate",
    "PurePythonBNN",
]
