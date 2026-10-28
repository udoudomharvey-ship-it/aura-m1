"""
aura_sdk.ncl
────────────
Language: Python (wraps Rust core when available)
NCL dual-rail primitives, simulator, and circuit building blocks.

    from aura_sdk.ncl import RailState, DualRail, TH22, TH12, NCLSimulator
    from aura_sdk.ncl.circuits import QualityStateEncoder, ThresholdDetector
"""
from .gates import (
    RailState,
    DualRail,
    NCLGate,
    TH22, TH12, TH13, TH23, TH33, THmn,
    NCLInverter,
    NCLRegister,
    CompletionDetector,
)
from .simulator import (
    NCLSimulator,
    SimResult,
    SimEvent,
    SimStats,
)

__all__ = [
    "RailState", "DualRail", "NCLGate",
    "TH22", "TH12", "TH13", "TH23", "TH33", "THmn",
    "NCLInverter", "NCLRegister", "CompletionDetector",
    "NCLSimulator", "SimResult", "SimEvent", "SimStats",
]
