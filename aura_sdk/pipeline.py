"""
aura_sdk/pipeline.py
─────────────────────
Language: Python (orchestration layer)
Why: Pipeline wiring, threading, and session management are
     expressed most clearly in Python. All heavy computation
     routes to Rust via the hemm and inference modules.

The AuraPipeline class is the primary integration point:

  [SkeletonSource]
       ↓  normalise joints
  [HEMMExtractor]   ← Rust backend (aura_core_rs) if available
       ↓  [H_sp, H_te, H_in]
  [HarvestingModel] ← energy gate
       ↓  if energy available
  [AuraBNN]         ← Rust backend if available
       ↓  Q_HCI, state
  [OutputSink(s)]   ← CSV, WebSocket, MQTT, console

Design principle:
  Python owns the wiring.
  Rust owns the math.
  The API boundary is clean at every layer.
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List

from .sensors   import SkeletonSource, SkeletonFrame, SyntheticSource
from .hemm      import HEMMExtractor, EntropyFeatures
from .inference import AuraBNN
from .harvesting import (
    EnergyHarvester, ThermalHarvester, RFHarvester,
    VibrationHarvester, HybridHarvester,
    BNN_INFERENCE_ENERGY_UJ,
)
from .ncl.circuits  import QualityStateEncoder, ThresholdDetector
from .output        import TelemetryRecord, OutputSink, ConsoleOutput
from .recommend     import RecommendationEngine, RecommendationEvent


# ── Pipeline configuration ──────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Configuration for the Aura-M1 Python pipeline.
    Mirror of aura_core::pipeline::PipelineConfig in Rust.
    """
    alert_threshold:  float = 0.42
    frame_dt_ms:      float = 33.3    # 30 fps
    vel_max:          float = 0.5     # m/frame, for entropy bin width
    max_frames:       Optional[int]   = None   # None = run until source exhausted
    verbose:          bool  = False


# ── Session statistics ──────────────────────────────────────────────────────

@dataclass
class SessionStats:
    total_frames:    int   = 0
    fired_frames:    int   = 0
    skipped_frames:  int   = 0
    fire_rate_pct:   float = 0.0
    mean_q:          float = 0.0
    total_energy_nj: float = 0.0
    idle_power_w:    float = 0.0
    alert_count:     int   = 0
    duration_s:      float = 0.0

    def report(self) -> str:
        return (
            f"── Session Statistics ─────────────────────\n"
            f"  Total frames     : {self.total_frames}\n"
            f"  Fired inferences : {self.fired_frames}  "
            f"({self.fire_rate_pct:.1f}%)\n"
            f"  Skipped (energy) : {self.skipped_frames}\n"
            f"  Mean Q_HCI       : {self.mean_q:.4f}\n"
            f"  Total energy     : {self.total_energy_nj:.2f} nJ\n"
            f"  Alert count      : {self.alert_count}\n"
            f"  Duration         : {self.duration_s:.2f} s\n"
            f"  Idle power       : {self.idle_power_w} W  ← SIMULATED ZERO (no hardware measurement)\n"
            f"────────────────────────────────────────────"
        )


# ── Main pipeline class ─────────────────────────────────────────────────────

class AuraPipeline:
    """
    Aura-M1 real-time inference pipeline.

    Connects a skeleton source to the entropy extractor,
    energy harvester, BNN inference engine, and output sinks.

    Usage — basic
    ─────────────
        pipeline = AuraPipeline()
        pipeline.run()   # runs synthetic session, prints to console

    Usage — full configuration
    ───────────────────────────
        pipeline = AuraPipeline(
            source     = MediaPipeSource(camera_idx=0),
            harvester  = ThermalHarvester(delta_T_C=3.0),
            sinks      = [CSVLogger("sessions/"), WebSocketServer(8765)],
            config     = PipelineConfig(alert_threshold=0.40),
        )
        pipeline.run()

    Usage — manual frame-by-frame
    ──────────────────────────────
        pipeline.open()
        for frame in source.frames():
            tel = pipeline.process_frame(frame)
            print(tel.q_hci, tel.state)
        pipeline.close()
    """

    def __init__(
        self,
        source:      Optional[SkeletonSource]      = None,
        harvester:   Optional[EnergyHarvester]     = None,
        sinks:       Optional[List[OutputSink]]    = None,
        config:      Optional[PipelineConfig]      = None,
        recommender: Optional[RecommendationEngine] = None,
    ):
        # NOTE: these were previously `source or SyntheticSource(...)` /
        # `sinks or [ConsoleOutput(...)]`, which is a real, confirmed bug:
        # `FileSource` defines `__len__` (returns len(self._frames)), and an
        # unopened FileSource has zero frames until .open() populates it —
        # so `bool(unopened_file_source)` is False, and `or` silently
        # discarded the caller's real FileSource in favour of 120 frames of
        # fake SyntheticSource data, with no error or warning. The same
        # pattern silently replaced an intentionally empty `sinks=[]`
        # (e.g. "run with no output sinks") with the default ConsoleOutput,
        # since `[] or default` is also always falsy in Python. Both are
        # fixed by testing identity against None instead of truthiness.
        self.source    = source    if source    is not None else SyntheticSource(n_frames=120)
        self.harvester = harvester if harvester is not None else ThermalHarvester(delta_T_C=3.0)
        self.sinks     = sinks     if sinks     is not None else [ConsoleOutput(every_n=10)]
        self.config    = config    if config    is not None else PipelineConfig()

        # Internal components
        self._extractor = HEMMExtractor(vel_max=self.config.vel_max)
        self._bnn       = AuraBNN()
        self._encoder   = QualityStateEncoder()
        self._alert     = ThresholdDetector(self.config.alert_threshold)
        # Optional recommend/act layer — sits on top of detect/measure/alert.
        # None by default: existing behaviour is unchanged unless a caller
        # opts in by passing `recommender=`.
        self._recommender = recommender
        self.last_recommendation: Optional[RecommendationEvent] = None

        # Session state
        self._prev_frame: Optional[SkeletonFrame] = None
        self._frame_idx   = 0
        self._q_sum       = 0.0
        self._fired       = 0
        self._alert_count = 0
        self._start_time  = 0.0
        self._open_flag   = False

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open source and all output sinks."""
        self.source.open()
        for sink in self.sinks:
            sink.open()
        self._start_time = time.time()
        self._open_flag  = True
        if self.config.verbose:
            print(f"  AuraPipeline opened")
            print(f"  Source    : {self.source}")
            print(f"  Harvester : {self.harvester}")
            print(f"  Backend   : {self._extractor.backend}")

    def close(self) -> None:
        """Close source and all output sinks."""
        self.source.close()
        for sink in self.sinks:
            sink.close()
        self._open_flag = False

    def reset_session(self) -> None:
        """Reset session accumulators without closing."""
        self._prev_frame  = None
        self._frame_idx   = 0
        self._q_sum       = 0.0
        self._fired       = 0
        self._alert_count = 0
        self._start_time  = time.time()
        self._bnn.reset_session()
        self._encoder.reset()
        self._alert.reset()
        if self._recommender is not None:
            self._recommender.reset()
        self.last_recommendation = None

    # ── Frame processing ───────────────────────────────────────────────────

    def process_frame(self, frame: SkeletonFrame) -> TelemetryRecord:
        """
        Process one skeleton frame through the full pipeline.

        This is the hot path — called 30× per second.
        Python handles routing; Rust handles entropy + BNN.
        """
        self._frame_idx += 1

        # Stage 1: Harvest energy for this frame period
        power_uw      = self.harvester.update(self.config.frame_dt_ms)
        accumulated_uj = self.harvester.accumulated_uJ

        # Stage 2: Skip if insufficient energy
        if not self.harvester.can_fire():
            self.last_recommendation = None
            record = TelemetryRecord(
                frame_idx      = self._frame_idx,
                timestamp_ms   = frame.timestamp_ms,
                fired          = False,
                q_hci          = None,
                state          = None,
                state_label    = None,
                alert          = False,
                power_uw       = power_uw,
                accumulated_uj = accumulated_uj,
                energy_nj      = 0.0,
                idle_power_w   = 0.0,
            )
            self._write_to_sinks(record)
            self._prev_frame = frame
            return record

        # Stage 3: Consume energy pulse
        self.harvester.consume_pulse()

        # Stage 4: Compute entropy features (Rust if available)
        features: EntropyFeatures
        if self._prev_frame is not None:
            features = self._extractor.compute(
                self._prev_frame.joints,
                frame.joints,
            )
        else:
            # First frame — no previous frame for velocity computation
            self._prev_frame = frame
            features = EntropyFeatures(h_sp=0.3, h_te=0.3, h_in=0.3)

        # Stage 5: BNN inference (Rust if available)
        result = self._bnn.infer(
            H_sp=features.h_sp,
            H_te=features.h_te,
            H_in=features.h_in,
        )

        # Stage 6: Encode quality state onto NCL dual-rail
        self._encoder.encode(result.q_hci)

        # Stage 7: Alert threshold check
        alert = self._alert.update(result.q_hci)

        # Stage 7b: Recommend/act layer (opt-in) — only ever sees the
        # already-computed `alert`, so it can't disagree with detection,
        # only decide what to do once that alert has been sustained.
        self.last_recommendation = None
        if self._recommender is not None:
            self.last_recommendation = self._recommender.update(
                alert=alert, state=result.state, q_hci=result.q_hci,
                features=features, frame_idx=self._frame_idx,
            )

        # Stage 8: Accumulate session stats
        self._q_sum       += result.q_hci
        self._fired       += 1
        if alert: self._alert_count += 1

        # Build telemetry record
        record = TelemetryRecord(
            frame_idx      = self._frame_idx,
            timestamp_ms   = frame.timestamp_ms,
            fired          = True,
            q_hci          = result.q_hci,
            state          = result.state,
            state_label    = result.state_label,
            alert          = alert,
            power_uw       = power_uw,
            accumulated_uj = self.harvester.accumulated_uJ,
            energy_nj      = result.energy_nJ,
            idle_power_w   = 0.0,
        )

        self._prev_frame = frame
        self._write_to_sinks(record)
        return record

    def _write_to_sinks(self, record: TelemetryRecord) -> None:
        for sink in self.sinks:
            try:
                sink.write(record)
            except Exception:
                pass

    # ── Run loop ───────────────────────────────────────────────────────────

    def run(self) -> SessionStats:
        """
        Run the full pipeline until source is exhausted or max_frames hit.
        Returns session statistics.
        """
        self.open()
        try:
            frame_count = 0
            for frame in self.source.frames():
                self.process_frame(frame)
                frame_count += 1
                if (self.config.max_frames is not None
                        and frame_count >= self.config.max_frames):
                    break
        finally:
            self.close()

        return self.session_stats()

    # ── Statistics ─────────────────────────────────────────────────────────

    def session_stats(self) -> SessionStats:
        total   = self._frame_idx
        fired   = self._fired
        elapsed = time.time() - self._start_time

        return SessionStats(
            total_frames    = total,
            fired_frames    = fired,
            skipped_frames  = total - fired,
            fire_rate_pct   = (fired / total * 100.0) if total > 0 else 0.0,
            mean_q          = (self._q_sum / fired) if fired > 0 else 0.0,
            total_energy_nj = self._bnn.total_energy_nJ,
            idle_power_w    = 0.0,
            alert_count     = self._alert_count,
            duration_s      = elapsed,
        )


# ── Convenience factory functions ───────────────────────────────────────────

def make_wearable_pipeline(
    delta_t_celsius: float = 3.0,
    alert_threshold: float = 0.40,
    sinks: Optional[List[OutputSink]] = None,
) -> AuraPipeline:
    """
    Pre-configured pipeline for wearable / medical implant deployment.
    Uses body heat as primary energy source.
    """
    return AuraPipeline(
        source    = SyntheticSource(120),
        harvester = HybridHarvester([
            ThermalHarvester(delta_T_C=delta_t_celsius),
            VibrationHarvester(acceleration_g=0.3),
        ]),
        sinks  = sinks if sinks is not None else [ConsoleOutput(every_n=15)],
        config = PipelineConfig(
            alert_threshold=alert_threshold,
            verbose=True,
        ),
    )


def make_industrial_pipeline(
    acceleration_g: float = 2.0,
    frequency_hz:   float = 60.0,
    sinks: Optional[List[OutputSink]] = None,
) -> AuraPipeline:
    """
    Pre-configured pipeline for industrial IoT deployment.
    Uses machine vibration as energy source.
    """
    return AuraPipeline(
        source    = SyntheticSource(120),
        harvester = VibrationHarvester(
            acceleration_g=acceleration_g,
        ),
        sinks  = sinks if sinks is not None else [ConsoleOutput(every_n=10)],
        config = PipelineConfig(
            alert_threshold=0.45,
            verbose=True,
        ),
    )


def make_demo_pipeline(n_frames: int = 120) -> AuraPipeline:
    """
    Minimal demo pipeline — no camera, no hardware needed.
    Runs a synthetic session and prints to console.
    """
    return AuraPipeline(
        source    = SyntheticSource(n_frames=n_frames),
        harvester = VibrationHarvester(acceleration_g=2.0),
        sinks     = [ConsoleOutput(every_n=8)],
        config    = PipelineConfig(verbose=True),
    )
