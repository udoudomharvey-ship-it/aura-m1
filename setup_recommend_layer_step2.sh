#!/usr/bin/env bash
# Run from the ROOT of your aura-m1 project, AFTER step 1's script.
#   bash setup_recommend_layer_step2.sh
#
# Updates aura_sdk/recommend/rules.py and __init__.py (adds episode_count
# tracking), adds aura_sdk/recommend/explain.py (the plain-language layer),
# adds aura_sdk_tests/test_recommend_explain.py, and refreshes
# demo_recommend.py to render through it.

set -euo pipefail

echo "Updating aura_sdk/recommend/rules.py ..."
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
    episode_count:     int              # how many recommendations have fired this session (incl. this one)

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
            "episode_count":     self.episode_count,
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
        # How many recommendations have fired this session — lets the
        # plain-language layer (step 2) escalate tone on repeat episodes
        # instead of repeating the exact same gentle phrasing every time.
        self.episode_count = 0

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
        self.episode_count += 1

        return RecommendationEvent(
            frame_idx         = frame_idx,
            problem_class     = problem,
            state             = state,
            q_hci             = q_hci,
            why               = why,
            suggested_actions = actions,
            rationale         = rationale,
            sustained_frames  = self._streak,
            episode_count     = self.episode_count,
        )

    def reset(self) -> None:
        self._streak = 0
        self._frames_since_last_rec = self.cooldown_frames
        self._recent_features.clear()
        self.episode_count = 0

    def __repr__(self):
        return (f"RecommendationEngine(streak={self._streak}, "
                f"since_last={self._frames_since_last_rec})")
AURA_EOF_RULES

echo "Updating aura_sdk/recommend/__init__.py ..."
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
from .explain import (
    explain,
    Notification,
    Severity,
)

__all__ = [
    "RecommendationEngine", "RecommendationEvent", "ProblemClass",
    "ACTION_LIBRARY", "WHY_TEMPLATES",
    "explain", "Notification", "Severity",
]
AURA_EOF_INIT

echo "Creating aura_sdk/recommend/explain.py ..."
cat > aura_sdk/recommend/explain.py << 'AURA_EOF_EXPLAIN'
"""
aura_sdk/recommend/explain.py
──────────────────────────────
Language: Python
Why: Step 1 (rules.py) decides WHAT the problem is and WHAT to suggest.
     This module decides HOW to say it to a person — a duration-aware,
     escalation-aware sentence instead of a static template, packaged
     as a small `Notification` a UI layer can render directly.

No network calls, no external model — fully deterministic and testable,
same as step 1. This is step 2 of the 4-part recommend/act layer:
    1. Rule-based recommendation table
    2. Plain-language explanation layer      ← this file
    3. Contextual bandit for personalisation
    4. Escalation / human-in-the-loop path

Two things make this "cognizant" rather than a template lookup:
  - duration:  "for about 15 seconds" is computed from sustained_frames,
               not hardcoded, so it stays accurate as sustain_frames
               config changes.
  - repetition: episode_count lets the tone shift from a gentle first
               nudge to a more direct one on repeat episodes in the
               same session, without ever becoming alarmist.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List

from .rules import RecommendationEvent, ProblemClass


# ── Severity ──────────────────────────────────────────────────────────────
# Derived from workload state, not from Q_HCI directly, so it reads
# naturally: state 1 = "Very High Workload" -> HIGH severity notification.

class Severity(Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"


def _severity_for_state(state: int) -> Severity:
    if state <= 1:
        return Severity.HIGH
    if state == 2:
        return Severity.MODERATE
    return Severity.LOW


# Human-readable label per problem class, used in the title — short,
# plain, no jargon (never "H_sp dominant", always the everyday name).
_LABELS = {
    ProblemClass.SPATIAL_RESTLESSNESS:  "Restlessness",
    ProblemClass.TEMPORAL_ERRATICNESS:  "Irregular movement",
    ProblemClass.COGNITIVE_OVERLOAD:    "Cognitive overload",
    ProblemClass.SUSTAINED_LOW_QUALITY: "Low interaction quality",
}

# Escalation phrasing appended when this is a repeat episode this
# session. Deliberately supportive, never scolding.
_ESCALATION = {
    1: "",  # first time this session — no escalation needed
    2: " This is the second time this has come up — worth taking seriously.",
    3: " This keeps recurring — a longer break might help more than a short one.",
}
_ESCALATION_DEFAULT = (
    " This has come up several times now — it may be worth stepping away "
    "for longer, or checking in with someone."
)


# ── Notification ──────────────────────────────────────────────────────────

@dataclass
class Notification:
    """A fully-formed, ready-to-render message for a UI layer — the
    companion app, the frontend orb, a push notification, whatever
    consumes this next. No entropy values, no jargon."""
    title:             str
    body:              str
    severity:          Severity
    suggested_actions: List[str]
    tag:               str     # ProblemClass value — for de-duplicating/grouping in a UI

    def to_dict(self) -> dict:
        return {
            "title":             self.title,
            "body":              self.body,
            "severity":          self.severity.value,
            "suggested_actions": self.suggested_actions,
            "tag":               self.tag,
        }


# ── Explain ───────────────────────────────────────────────────────────────

def _duration_phrase(duration_s: float) -> str:
    if duration_s < 1.0:
        return "the last few seconds"
    if duration_s < 60.0:
        return f"about {duration_s:.0f} seconds"
    return f"about {duration_s / 60.0:.1f} minutes"


def explain(event: RecommendationEvent, frame_dt_ms: float = 33.3) -> Notification:
    """
    Turn a RecommendationEvent into a plain-language Notification.

    frame_dt_ms should match the pipeline's PipelineConfig.frame_dt_ms —
    it's only used to turn `sustained_frames` into a human duration.
    """
    severity   = _severity_for_state(event.state)
    label      = _LABELS[event.problem_class]
    duration_s = (event.sustained_frames * frame_dt_ms) / 1000.0

    title = f"{label} detected" if severity != Severity.LOW else "Quick check-in"

    escalation = _ESCALATION.get(event.episode_count, _ESCALATION_DEFAULT)
    body = f"{event.why} This has been going on for {_duration_phrase(duration_s)}.{escalation}"

    return Notification(
        title             = title,
        body              = body,
        severity          = severity,
        suggested_actions = event.suggested_actions,
        tag               = event.problem_class.value,
    )
AURA_EOF_EXPLAIN

echo "Creating aura_sdk_tests/test_recommend_explain.py ..."
cat > aura_sdk_tests/test_recommend_explain.py << 'AURA_EOF_TESTS2'
"""
aura_sdk_tests/test_recommend_explain.py
───────────────────────────────────────────
Language: Python (unittest)
Why: Exercises the duration phrasing and episode-count escalation
that make the plain-language layer "cognizant" rather than a static
template — plus the severity mapping used to prioritise a UI.

Run: python -m unittest aura_sdk_tests.test_recommend_explain -v
"""
from __future__ import annotations
import unittest

from aura_sdk.recommend import (
    RecommendationEngine, RecommendationEvent, ProblemClass,
    explain, Severity,
)


def make_event(state=1, sustained_frames=5, episode_count=1,
               problem_class=ProblemClass.COGNITIVE_OVERLOAD):
    return RecommendationEvent(
        frame_idx=1, problem_class=problem_class, state=state,
        q_hci=0.1, why="There's a lot of coordination strain right now.",
        suggested_actions=["Break the current task into smaller steps"],
        rationale="test", sustained_frames=sustained_frames,
        episode_count=episode_count,
    )


class TestSeverityMapping(unittest.TestCase):

    def test_state_one_is_high_severity(self):
        n = explain(make_event(state=1))
        self.assertEqual(n.severity, Severity.HIGH)

    def test_state_two_is_moderate_severity(self):
        n = explain(make_event(state=2))
        self.assertEqual(n.severity, Severity.MODERATE)

    def test_state_three_or_lower_workload_is_low_severity(self):
        n = explain(make_event(state=3))
        self.assertEqual(n.severity, Severity.LOW)


class TestDurationPhrasing(unittest.TestCase):

    def test_sub_second_streak_does_not_say_zero_seconds(self):
        # 5 frames at 33.3ms ≈ 0.17s — this used to render as
        # "about 0 seconds", which is exactly the bug this guards against.
        n = explain(make_event(sustained_frames=5), frame_dt_ms=33.3)
        self.assertNotIn("0 seconds", n.body)

    def test_multi_second_streak_shows_seconds(self):
        # 60 frames at 33.3ms ≈ 2.0s
        n = explain(make_event(sustained_frames=60), frame_dt_ms=33.3)
        self.assertIn("2 seconds", n.body)

    def test_long_streak_shows_minutes_not_seconds(self):
        # 3000 frames at 33.3ms = 99.9s ≈ 1.7 minutes
        n = explain(make_event(sustained_frames=3000), frame_dt_ms=33.3)
        self.assertIn("minute", n.body)

    def test_longer_streak_produces_longer_duration_text(self):
        short = explain(make_event(sustained_frames=5),  frame_dt_ms=33.3)
        long_ = explain(make_event(sustained_frames=150), frame_dt_ms=33.3)
        self.assertNotEqual(short.body, long_.body)


class TestEscalation(unittest.TestCase):

    def test_first_episode_has_no_escalation_language(self):
        n = explain(make_event(episode_count=1))
        self.assertNotIn("recurring", n.body)
        self.assertNotIn("second time", n.body)

    def test_second_episode_mentions_repetition(self):
        n = explain(make_event(episode_count=2))
        self.assertIn("second time", n.body)

    def test_many_episodes_falls_back_to_default_escalation(self):
        n = explain(make_event(episode_count=10))
        self.assertIn("several times", n.body)


class TestNotificationShape(unittest.TestCase):

    def test_no_entropy_jargon_in_title_or_body(self):
        n = explain(make_event())
        for jargon in ("H_sp", "H_te", "H_in", "Q_HCI", "entropy"):
            self.assertNotIn(jargon, n.title)
            self.assertNotIn(jargon, n.body)

    def test_to_dict_is_json_safe(self):
        n = explain(make_event())
        d = n.to_dict()
        self.assertIsInstance(d["title"], str)
        self.assertIsInstance(d["body"], str)
        self.assertEqual(d["severity"], "high")
        self.assertEqual(d["tag"], "cognitive_overload")

    def test_actions_pass_through_unchanged(self):
        event = make_event()
        n = explain(event)
        self.assertEqual(n.suggested_actions, event.suggested_actions)


class TestEndToEndWithEngine(unittest.TestCase):

    def test_explain_consumes_real_engine_output(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        result = None
        for i in range(3):
            result = engine.update(alert=True, state=1, q_hci=0.1,
                                    features=None, frame_idx=i)
        self.assertIsNotNone(result)
        n = explain(result)
        self.assertEqual(n.severity, Severity.HIGH)
        self.assertTrue(len(n.body) > 0)


if __name__ == "__main__":
    unittest.main()
AURA_EOF_TESTS2

echo "Refreshing demo_recommend.py ..."
cat > demo_recommend.py << 'AURA_EOF_DEMO2'
"""
Demo: recommend/act layer running on top of the existing
detect/measure/alert pipeline, using the Hybrid RF+Thermal harvester
config — step 1 (rule-based table) feeding step 2 (plain-language layer).
"""
from aura_sdk.pipeline import AuraPipeline, PipelineConfig
from aura_sdk.sensors import SyntheticSource
from aura_sdk.harvesting import RFHarvester, ThermalHarvester, HybridHarvester
from aura_sdk.recommend import RecommendationEngine, explain

pipeline = AuraPipeline(
    source=SyntheticSource(n_frames=200, seed=42),
    harvester=HybridHarvester(sources=[RFHarvester(), ThermalHarvester()]),
    sinks=[],
    config=PipelineConfig(verbose=False),
    recommender=RecommendationEngine(sustain_frames=5, cooldown_frames=60),
)

pipeline.open()
notifications = []
for frame in pipeline.source.frames():
    tel = pipeline.process_frame(frame)
    if pipeline.last_recommendation:
        notifications.append(explain(pipeline.last_recommendation,
                                      frame_dt_ms=pipeline.config.frame_dt_ms))
pipeline.close()

print(f"{len(notifications)} notification(s) over 200 frames:\n")
for n in notifications:
    print(f"  [{n.severity.value.upper():8s}] {n.title}")
    print(f"      {n.body}")
    for a in n.suggested_actions:
        print(f"      → {a}")
    print()
AURA_EOF_DEMO2

echo ""
echo "Done. Running the full test suite now..."
python3 -m unittest discover -s aura_sdk_tests -v
