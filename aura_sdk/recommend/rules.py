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
