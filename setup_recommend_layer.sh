#!/usr/bin/env bash
# Run this from the ROOT of your aura-m1 project in the VS Code terminal:
#   bash setup_recommend_layer.sh
#
# It creates aura_sdk/recommend/ (new), adds the new test file, adds
# demo_recommend.py, and overwrites aura_sdk/pipeline.py with the version
# that has the optional `recommender=` param wired in.
#
# SAFETY: if you've made local edits to pipeline.py since the last export,
# check `git diff aura_sdk/pipeline.py` (or your own backup) before running
# this — it does a straight overwrite of that one file.

set -euo pipefail

echo "Creating aura_sdk/recommend/ ..."
mkdir -p aura_sdk/recommend

cat > aura_sdk/recommend/__init__.py << 'AURA_EOF_INIT'
"""
aura_sdk.recommend
────────────────────
Language: Python
The recommend/act layer — sits above detect/measure/alert.

    from aura_sdk.recommend import RecommendationEngine, RecommendationEvent, ProblemClass
"""
from .rules import (
    RecommendationEngine,
    RecommendationEvent,
    ProblemClass,
    ACTION_LIBRARY,
    WHY_TEMPLATES,
)

__all__ = [
    "RecommendationEngine", "RecommendationEvent", "ProblemClass",
    "ACTION_LIBRARY", "WHY_TEMPLATES",
]
AURA_EOF_INIT

cat > aura_sdk/recommend/rules.py << 'AURA_EOF_RULES'
"""
aura_sdk/recommend/rules.py
──────────────────────────────
Language: Python
Why: Sits above detection, not inside it. It consumes the same per-frame
     telemetry that already flows through the pipeline (Q_HCI, quality
     state, and — averaged over a short window — the underlying HEMM
     entropy features) and turns a *sustained* alert into a concrete,
     plain-worded recommendation, instead of leaving the loop at "alert".

Design principle
──────────────────
  Detection   answers "is something wrong right now?"        (ThresholdDetector)
  Measurement answers "how wrong, in which dimension?"        (H_sp / H_te / H_in)
  This module answers "what would actually help, and how do
                        I phrase that for someone who isn't
                        reading entropy values?"

Deliberately deterministic and explainable — no learning here. This is
step 1 of the 4-part recommend/act layer:
    1. Rule-based recommendation table          ← this file
    2. Plain-language explanation layer
    3. Contextual bandit for personalisation
    4. Escalation / human-in-the-loop path

Nagging is a real failure mode for a system like this, so two forms of
hysteresis are built in on purpose:
  - sustain_frames  — a problem must persist, not just spike one frame
  - cooldown_frames — once a recommendation fires, wait before firing again
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Optional, List, Dict

from ..hemm.extractor import EntropyFeatures


# ── Problem taxonomy ─────────────────────────────────────────────────────────

class ProblemClass(Enum):
    """Which dimension of the HEMM feature vector is driving the sustained
    alert — determined from the *averaged* recent H_sp/H_te/H_in, not a
    single noisy frame."""
    SPATIAL_RESTLESSNESS  = "spatial_restlessness"    # H_sp dominant
    TEMPORAL_ERRATICNESS  = "temporal_erraticness"    # H_te dominant
    COGNITIVE_OVERLOAD    = "cognitive_overload"       # H_in dominant
    SUSTAINED_LOW_QUALITY = "sustained_low_quality"    # no single dominant term


# ── Action + explanation library ─────────────────────────────────────────────
# Short and concrete on purpose — this is what a person under load actually
# has to read and act on, not a menu.

ACTION_LIBRARY: Dict[ProblemClass, List[str]] = {
    ProblemClass.SPATIAL_RESTLESSNESS: [
        "Take a 60-second stretch break",
        "Adjust your seating or posture",
    ],
    ProblemClass.TEMPORAL_ERRATICNESS: [
        "Pause for three slow breaths before continuing",
        "Step away for a short walk if you can",
    ],
    ProblemClass.COGNITIVE_OVERLOAD: [
        "Break the current task into smaller steps",
        "Postpone non-urgent decisions for a few minutes",
    ],
    ProblemClass.SUSTAINED_LOW_QUALITY: [
        "Take a short break — even 2-3 minutes helps",
        "Switch to a lower-focus task for a while",
    ],
}

WHY_TEMPLATES: Dict[ProblemClass, str] = {
    ProblemClass.SPATIAL_RESTLESSNESS:  "You've been shifting position a lot.",
    ProblemClass.TEMPORAL_ERRATICNESS:  "Your movements have been unusually irregular.",
    ProblemClass.COGNITIVE_OVERLOAD:    "There's a lot of coordination strain right now.",
    ProblemClass.SUSTAINED_LOW_QUALITY: "Your interaction quality has been low for a while.",
}


# ── Recommendation event ──────────────────────────────────────────────────────

@dataclass
class RecommendationEvent:
    """One recommendation, ready for a UI layer or the plain-language
    layer (step 2) to render."""
    frame_idx:         int
    problem_class:     ProblemClass
    state:             int              # 1 (very high workload) .. 5 (very low workload)
    q_hci:             float
    why:               str              # ~1 plain-language sentence
    suggested_actions: List[str]        # 1-3 concrete, low-friction actions
    rationale:         str              # technical detail, for logs/debugging
    sustained_frames:  int              # how many consecutive alert frames triggered this

    def to_dict(self) -> dict:
        return {
            "frame_idx":         self.frame_idx,
            "problem_class":     self.problem_class.value,
            "state":             self.state,
            "q_hci":             round(self.q_hci, 4),
            "why":               self.why,
            "suggested_actions": self.suggested_actions,
            "rationale":         self.rationale,
            "sustained_frames":  self.sustained_frames,
        }


# ── Recommendation engine ─────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Turns a stream of (alert, state, Q_HCI, entropy features) into
    RecommendationEvents — only once a problem has been sustained, and
    not more often than a cooldown allows.

    Deliberately takes the already-computed `alert` boolean from your
    existing ThresholdDetector rather than re-deriving it, so this stays
    a layer on top of detection rather than a second, possibly-divergent
    definition of "alert".

    Usage
    ─────
        engine = RecommendationEngine()
        for record, features in stream:               # per fired frame
            rec = engine.update(
                alert=record.alert, state=record.state,
                q_hci=record.q_hci, features=features,
                frame_idx=record.frame_idx,
            )
            if rec:
                notify_user(rec)
    """

    # A feature term counts as "dominant" once it clears this share of the
    # H_sp + H_te + H_in total — i.e. it's driving the degradation more
    # than an even three-way split would suggest.
    DOMINANCE_SHARE = 0.45

    def __init__(
        self,
        sustain_frames:  int = 5,     # matches HEMM_WINDOW_SIZE — ignore single-frame noise
        cooldown_frames: int = 90,    # ~3s at 30fps — advise, don't nag
    ):
        assert sustain_frames  >= 1
        assert cooldown_frames >= 0
        self.sustain_frames  = sustain_frames
        self.cooldown_frames = cooldown_frames

        self._streak = 0
        # Start "off cooldown" so the very first sustained problem can fire.
        self._frames_since_last_rec = cooldown_frames
        self._recent_features: deque[EntropyFeatures] = deque(maxlen=sustain_frames)

    # ── Internal ──────────────────────────────────────────────────────────

    def _classify(self) -> ProblemClass:
        n = len(self._recent_features)
        if n == 0:
            return ProblemClass.SUSTAINED_LOW_QUALITY

        h_sp = sum(f.h_sp for f in self._recent_features) / n
        h_te = sum(f.h_te for f in self._recent_features) / n
        h_in = sum(f.h_in for f in self._recent_features) / n
        total = h_sp + h_te + h_in
        if total <= 1e-9:
            return ProblemClass.SUSTAINED_LOW_QUALITY

        shares = {
            ProblemClass.SPATIAL_RESTLESSNESS: h_sp / total,
            ProblemClass.TEMPORAL_ERRATICNESS: h_te / total,
            ProblemClass.COGNITIVE_OVERLOAD:   h_in / total,
        }
        top_class, top_share = max(shares.items(), key=lambda kv: kv[1])
        return top_class if top_share >= self.DOMINANCE_SHARE else ProblemClass.SUSTAINED_LOW_QUALITY

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        alert:     bool,
        state:     Optional[int],
        q_hci:     Optional[float],
        features:  Optional[EntropyFeatures],
        frame_idx: int,
    ) -> Optional[RecommendationEvent]:
        """Call once per fired frame. Returns a RecommendationEvent only
        when the alert has been sustained for `sustain_frames` and the
        cooldown has elapsed since the last one — otherwise None."""
        self._frames_since_last_rec += 1

        if features is not None:
            self._recent_features.append(features)

        if not alert or state is None or q_hci is None:
            # A clear frame breaks the streak — a brief dip shouldn't
            # carry over into an unrelated later episode.
            self._streak = 0
            return None

        self._streak += 1
        if self._streak < self.sustain_frames:
            return None
        if self._frames_since_last_rec < self.cooldown_frames:
            return None

        problem   = self._classify()
        why       = WHY_TEMPLATES[problem]
        actions   = ACTION_LIBRARY[problem]
        rationale = (
            f"state={state} q_hci={q_hci:.4f} "
            f"sustained={self._streak}f problem={problem.value}"
        )

        self._frames_since_last_rec = 0

        return RecommendationEvent(
            frame_idx         = frame_idx,
            problem_class     = problem,
            state             = state,
            q_hci             = q_hci,
            why               = why,
            suggested_actions = actions,
            rationale         = rationale,
            sustained_frames  = self._streak,
        )

    def reset(self) -> None:
        self._streak = 0
        self._frames_since_last_rec = self.cooldown_frames
        self._recent_features.clear()

    def __repr__(self):
        return (f"RecommendationEngine(streak={self._streak}, "
                f"since_last={self._frames_since_last_rec})")
AURA_EOF_RULES

echo "Creating aura_sdk_tests/test_recommend_rules.py ..."
cat > aura_sdk_tests/test_recommend_rules.py << 'AURA_EOF_TESTS'
"""
aura_sdk_tests/test_recommend_rules.py
─────────────────────────────────────────
Language: Python (unittest)
Why: Exercises the two pieces of hysteresis the recommendation layer's
"advise, don't nag" behaviour depends on (sustain + cooldown), and the
dominant-entropy-term classification that decides which suggestion is
shown. Deterministic and independent of the Rust backend.

Run: python -m unittest aura_sdk_tests.test_recommend_rules -v
"""
from __future__ import annotations
import unittest

from aura_sdk.recommend import RecommendationEngine, ProblemClass
from aura_sdk.hemm import EntropyFeatures


def feed(engine, n, alert=True, state=1, q_hci=0.2, features=None, start_frame=1):
    """Feed `n` identical frames through the engine, returning the last result."""
    result = None
    for i in range(n):
        result = engine.update(
            alert=alert, state=state, q_hci=q_hci,
            features=features, frame_idx=start_frame + i,
        )
    return result


class TestSustainedTrigger(unittest.TestCase):

    def test_no_recommendation_before_sustain_threshold(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        result = feed(engine, 4)
        self.assertIsNone(result)

    def test_recommendation_fires_on_reaching_sustain_threshold(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        result = feed(engine, 5)
        self.assertIsNotNone(result)
        self.assertEqual(result.sustained_frames, 5)

    def test_single_clear_frame_resets_the_streak(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        feed(engine, 4)
        # One clear frame in the middle should break the streak.
        cleared = engine.update(alert=False, state=4, q_hci=0.7,
                                 features=None, frame_idx=100)
        self.assertIsNone(cleared)
        # Needs the full 5 again, not just 1 more.
        result = feed(engine, 4, start_frame=200)
        self.assertIsNone(result)
        result = feed(engine, 1, start_frame=300)
        self.assertIsNotNone(result)

    def test_no_alert_never_fires(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        result = feed(engine, 50, alert=False, state=5, q_hci=0.9)
        self.assertIsNone(result)


class TestCooldown(unittest.TestCase):

    def test_second_recommendation_suppressed_within_cooldown(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=10)
        first = feed(engine, 3)
        self.assertIsNotNone(first)
        # Keep alerting immediately after — still within cooldown.
        second = feed(engine, 3, start_frame=100)
        self.assertIsNone(second)

    def test_recommendation_fires_again_after_cooldown_elapses(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=5)
        first = feed(engine, 3)
        self.assertIsNotNone(first)
        # 5 more alerting frames: cooldown (5) elapses, streak keeps climbing.
        second = feed(engine, 5, start_frame=100)
        self.assertIsNotNone(second)


class TestClassification(unittest.TestCase):

    def test_spatial_dominance_detected(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        feats = EntropyFeatures(h_sp=0.9, h_te=0.1, h_in=0.1)
        result = feed(engine, 5, features=feats)
        self.assertEqual(result.problem_class, ProblemClass.SPATIAL_RESTLESSNESS)

    def test_temporal_dominance_detected(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        feats = EntropyFeatures(h_sp=0.1, h_te=0.9, h_in=0.1)
        result = feed(engine, 5, features=feats)
        self.assertEqual(result.problem_class, ProblemClass.TEMPORAL_ERRATICNESS)

    def test_cognitive_overload_dominance_detected(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        feats = EntropyFeatures(h_sp=0.1, h_te=0.1, h_in=0.9)
        result = feed(engine, 5, features=feats)
        self.assertEqual(result.problem_class, ProblemClass.COGNITIVE_OVERLOAD)

    def test_even_split_falls_back_to_sustained_low_quality(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        feats = EntropyFeatures(h_sp=0.33, h_te=0.33, h_in=0.34)
        result = feed(engine, 5, features=feats)
        self.assertEqual(result.problem_class, ProblemClass.SUSTAINED_LOW_QUALITY)

    def test_no_features_falls_back_to_sustained_low_quality(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        result = feed(engine, 5, features=None)
        self.assertEqual(result.problem_class, ProblemClass.SUSTAINED_LOW_QUALITY)

    def test_every_problem_class_has_actions_and_a_why(self):
        from aura_sdk.recommend import ACTION_LIBRARY, WHY_TEMPLATES
        for pc in ProblemClass:
            self.assertIn(pc, ACTION_LIBRARY)
            self.assertGreater(len(ACTION_LIBRARY[pc]), 0)
            self.assertIn(pc, WHY_TEMPLATES)


class TestRecommendationEventShape(unittest.TestCase):

    def test_to_dict_is_json_safe_types(self):
        engine = RecommendationEngine(sustain_frames=1, cooldown_frames=0)
        feats = EntropyFeatures(h_sp=0.9, h_te=0.05, h_in=0.05)
        result = engine.update(alert=True, state=1, q_hci=0.1234,
                                features=feats, frame_idx=7)
        d = result.to_dict()
        self.assertEqual(d["problem_class"], "spatial_restlessness")
        self.assertIsInstance(d["suggested_actions"], list)
        self.assertIsInstance(d["why"], str)
        self.assertEqual(d["frame_idx"], 7)


class TestReset(unittest.TestCase):

    def test_reset_clears_streak_and_history(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        feed(engine, 4)
        engine.reset()
        # Needs the full 5 again after reset, not just 1 more.
        result = feed(engine, 1, start_frame=500)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
AURA_EOF_TESTS

echo "Creating demo_recommend.py ..."
cat > demo_recommend.py << 'AURA_EOF_DEMO'
"""
Demo: recommend/act layer (step 1, rule-based) running on top of the
existing detect/measure/alert pipeline, using the Hybrid RF+Thermal
harvester config.
"""
from aura_sdk.pipeline import AuraPipeline, PipelineConfig
from aura_sdk.sensors import SyntheticSource
from aura_sdk.harvesting import RFHarvester, ThermalHarvester, HybridHarvester
from aura_sdk.recommend import RecommendationEngine

pipeline = AuraPipeline(
    source=SyntheticSource(n_frames=200, seed=42),
    harvester=HybridHarvester(sources=[RFHarvester(), ThermalHarvester()]),
    sinks=[],
    config=PipelineConfig(verbose=False),
    recommender=RecommendationEngine(sustain_frames=5, cooldown_frames=60),
)

pipeline.open()
events = []
for frame in pipeline.source.frames():
    tel = pipeline.process_frame(frame)
    if pipeline.last_recommendation:
        events.append(pipeline.last_recommendation)
pipeline.close()

print(f"{len(events)} recommendation(s) fired over 200 frames:\n")
for e in events:
    print(f"  [frame {e.frame_idx:4d}] {e.problem_class.value:22s} "
          f"Q_HCI={e.q_hci:.3f}  \"{e.why}\"")
    for a in e.suggested_actions:
        print(f"              → {a}")
    print()
AURA_EOF_DEMO

echo "Overwriting aura_sdk/pipeline.py with the recommender-wired version ..."
cat > aura_sdk/pipeline.py << 'AURA_EOF_PIPELINE'
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
AURA_EOF_PIPELINE

echo ""
echo "Done. Running the test suite now..."
python3 -m unittest discover -s aura_sdk_tests -v
