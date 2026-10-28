"""
aura_sdk — Aura-M1 Sub-Threshold Kinetic Micro-Controller SDK
═══════════════════════════════════════════════════════════════

GreyMatter Labs Ltd.  ·  "Where intelligence begins."
Harvey Frederick Udoudom  ·  University of Uyo  ·  2025

Hybrid Architecture
────────────────────
  Python layer  : Training, research, API interface, sensor drivers
  Rust core     : Entropy computation, BNN inference, harvesting models
  Binding layer : PyO3 — identical API regardless of backend

Quick start
────────────
    # NCL gate
    from aura_sdk.ncl import TH22, DualRail, RailState
    a = DualRail("a", RailState.DATA1)
    b = DualRail("b", RailState.DATA1)
    print(TH22()(a, b))           # DATA1

    # Compile NCL expression
    from aura_sdk.compiler import compile
    cr = compile("AND(OR(a, b), NOT(c))")
    cr.print_truth_table()

    # BNN inference
    from aura_sdk.inference import AuraBNN
    bnn    = AuraBNN()
    result = bnn.infer(H_sp=0.42, H_te=0.61, H_in=0.28)
    print(result.q_hci, result.idle_power_uW)   # 0.xxxx  0.0

    # Energy harvesting
    from aura_sdk.harvesting import ThermalHarvester, HarvestingSession
    h     = ThermalHarvester(delta_T_C=3.0)
    stats = HarvestingSession(h, duration_s=10.0).run()
    print(stats["idle_power_W"])                 # 0.0

    # Full pipeline
    from aura_sdk.pipeline import make_demo_pipeline
    make_demo_pipeline(n_frames=120).run()
"""

__version__  = "0.1.0"
__author__   = "Harvey Frederick Udoudom"
__company__  = "GreyMatter Labs Ltd."
__tagline__  = "Where intelligence begins."

# Detect Rust backend availability
try:
    import aura_core_rs as _rust
    RUST_BACKEND = True
    RUST_VERSION = getattr(_rust, "__version__", "unknown")
except ImportError:
    RUST_BACKEND = False
    RUST_VERSION = None

# ── Public API imports ─────────────────────────────────────────────────────

from .ncl import (
    RailState, DualRail, NCLGate,
    TH22, TH12, TH13, TH23, TH33, THmn,
    NCLInverter, NCLRegister, CompletionDetector,
    NCLSimulator, SimResult,
)
from .ncl.circuits import QualityStateEncoder, ThresholdDetector

from .compiler import compile, CompileResult

from .inference import AuraBNN, InferenceResult, PowerBudget

from .harvesting import (
    EnergyHarvester,
    RFHarvester, VibrationHarvester, ThermalHarvester,
    LightHarvester, FlowHarvester, HybridHarvester,
    HarvestingSession,
    BNN_INFERENCE_ENERGY_NJ,
)

from .hemm import HEMMExtractor, EntropyFeatures, normalise_joints

from .sensors import (
    SkeletonSource, SkeletonFrame,
    SyntheticSource, FileSource, MediaPipeSource,
)

from .output import (
    TelemetryRecord, OutputSink,
    CSVLogger, ConsoleOutput,
    WebSocketServer, MQTTPublisher, MultiSink,
)

from .pipeline import (
    AuraPipeline, PipelineConfig, SessionStats,
    make_demo_pipeline, make_wearable_pipeline, make_industrial_pipeline,
)

from .calibration import CalibrationProfile, UserCalibrator

__all__ = [
    # NCL
    "RailState", "DualRail", "NCLGate",
    "TH22", "TH12", "TH13", "TH23", "TH33", "THmn",
    "NCLInverter", "NCLRegister", "CompletionDetector",
    "NCLSimulator", "SimResult",
    "QualityStateEncoder", "ThresholdDetector",
    # Compiler
    "compile", "CompileResult",
    # Inference
    "AuraBNN", "InferenceResult", "PowerBudget",
    # Harvesting
    "EnergyHarvester",
    "RFHarvester", "VibrationHarvester", "ThermalHarvester",
    "LightHarvester", "FlowHarvester", "HybridHarvester",
    "HarvestingSession", "BNN_INFERENCE_ENERGY_NJ",
    # HEMM
    "HEMMExtractor", "EntropyFeatures", "normalise_joints",
    # Sensors
    "SkeletonSource", "SkeletonFrame",
    "SyntheticSource", "FileSource", "MediaPipeSource",
    # Output
    "TelemetryRecord", "OutputSink",
    "CSVLogger", "ConsoleOutput",
    "WebSocketServer", "MQTTPublisher", "MultiSink",
    # Pipeline
    "AuraPipeline", "PipelineConfig", "SessionStats",
    "make_demo_pipeline", "make_wearable_pipeline", "make_industrial_pipeline",
    # Calibration
    "CalibrationProfile", "UserCalibrator",
    # Meta
    "RUST_BACKEND", "RUST_VERSION",
]


def info() -> str:
    """Return SDK info string."""
    backend = f"Rust ({RUST_VERSION})" if RUST_BACKEND else "Python (Rust not compiled)"
    return (
        f"aura_sdk v{__version__}\n"
        f"  Company  : {__company__}\n"
        f"  Author   : {__author__}\n"
        f"  Backend  : {backend}\n"
        f"  Tagline  : '{__tagline__}'\n"
        f"  Idle pwr : 0.00 W  ← SIMULATED ZERO (NCL NULL state — no hardware measurement)"
    )
