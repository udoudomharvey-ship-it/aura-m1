"""
aura_sdk/recommend/personalize.py
────────────────────────────────────
Language: Python
Why: A bandit is only as good as its reward loop. This module closes
     it: fire a recommendation with the bandit's current best-guess
     action leading the list, remember Q_HCI at that moment, then —
     a fixed number of frames later — check whether Q_HCI actually
     recovered and feed that back to the bandit as the reward. No
     labeling and no UI interaction are required for this to work,
     though explicit accept/dismiss feedback could supplement it later.

Wraps RecommendationEngine (step 1) rather than replacing it — sustain
and cooldown behaviour are unchanged; only the ORDER of
`suggested_actions` becomes learned per person, per problem class.

This is step 3 of the 4-part recommend/act layer:
    1. Rule-based recommendation table
    2. Plain-language explanation layer
    3. Contextual bandit for personalisation      ← this file
    4. Escalation / human-in-the-loop path
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Callable

from .rules import RecommendationEngine, RecommendationEvent, ProblemClass
from .bandit import ActionBandit
from ..hemm.extractor import EntropyFeatures


@dataclass
class _PendingOutcome:
    problem_class: ProblemClass
    action:        str
    q_hci_at_fire: float
    eval_frame:    int     # frame_idx at which to score this recommendation


class PersonalizedRecommender:
    """
    Wraps a RecommendationEngine with a bandit that learns which action
    helps most, per problem class, for this specific person — using
    Q_HCI recovery within `eval_window_frames` as the reward.

    Usage
    ─────
        recommender = PersonalizedRecommender()
        for frame in stream:
            event = recommender.update(alert=..., state=..., q_hci=...,
                                        features=..., frame_idx=...)
            if event:
                notify_user(event)   # suggested_actions[0] is the
                                      # bandit's current best guess

    Saving/restoring learned preferences across sessions:
        saved = recommender.bandit.to_dict()
        ...
        recommender = PersonalizedRecommender(bandit=ActionBandit.from_dict(saved))
    """

    def __init__(
        self,
        base_engine:        Optional[RecommendationEngine] = None,
        bandit:              Optional[ActionBandit]          = None,
        eval_window_frames:  int   = 150,   # ~5s at 30fps — time allowed to judge recovery
        recovery_margin:     float = 0.05,  # min Q_HCI gain counted as "this helped"
        on_outcome: Optional[Callable[[ProblemClass, str, bool, int], None]] = None,
    ):
        assert eval_window_frames >= 1
        self.engine = base_engine if base_engine is not None else RecommendationEngine()
        self.bandit = bandit      if bandit      is not None else ActionBandit()
        self.eval_window_frames = eval_window_frames
        self.recovery_margin    = recovery_margin
        # Optional observer, called (problem_class, action, reward, frame_idx)
        # whenever an outcome resolves — lets step 4 (escalation) track
        # unresolved streaks without duplicating this reward computation.
        self._on_outcome = on_outcome

        self._pending: List[_PendingOutcome] = []

    def update(
        self,
        alert:     bool,
        state:     Optional[int],
        q_hci:     Optional[float],
        features:  Optional[EntropyFeatures],
        frame_idx: int,
    ) -> Optional[RecommendationEvent]:
        """Same signature as RecommendationEngine.update — drop-in
        replacement wherever step 1's engine was used directly."""
        # Resolve any due outcomes BEFORE firing anything new, using this
        # frame's Q_HCI as the "after" reading — so a brand-new
        # recommendation never scores itself on the frame it's created.
        if q_hci is not None:
            self._resolve_pending(frame_idx, q_hci)

        event = self.engine.update(alert=alert, state=state, q_hci=q_hci,
                                    features=features, frame_idx=frame_idx)
        if event is None:
            return None

        # Let the bandit choose which of the (already-vetted, step-1)
        # actions to lead with, then keep the rest as fallback options.
        chosen = self.bandit.choose(event.problem_class)
        event.suggested_actions = [chosen] + [
            a for a in event.suggested_actions if a != chosen
        ]

        self._pending.append(_PendingOutcome(
            problem_class = event.problem_class,
            action        = chosen,
            q_hci_at_fire = event.q_hci,
            eval_frame    = frame_idx + self.eval_window_frames,
        ))
        return event

    def _resolve_pending(self, frame_idx: int, q_hci_now: float) -> None:
        still_pending = []
        for p in self._pending:
            if frame_idx < p.eval_frame:
                still_pending.append(p)
                continue
            reward = (q_hci_now - p.q_hci_at_fire) >= self.recovery_margin
            self.bandit.update(p.problem_class, p.action, reward)
            if self._on_outcome is not None:
                self._on_outcome(p.problem_class, p.action, reward, frame_idx)
        self._pending = still_pending

    def reset(self) -> None:
        """Resets session-local state (sustain streak, cooldown, pending
        evaluations) but deliberately does NOT reset the bandit — learned
        preferences are meant to persist across sessions, that's the
        entire point of personalization."""
        self.engine.reset()
        self._pending.clear()

    def __repr__(self):
        return (f"PersonalizedRecommender(pending={len(self._pending)}, "
                f"{self.bandit!r})")
