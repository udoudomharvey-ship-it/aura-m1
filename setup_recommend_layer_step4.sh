#!/usr/bin/env bash
# Run from the ROOT of your aura-m1 project, AFTER steps 1, 2, and 3.
#   bash setup_recommend_layer_step4.sh
#
# Updates aura_sdk/recommend/personalize.py (adds an on_outcome hook, additive
# and backward-compatible) and __init__.py, adds aura_sdk/recommend/escalate.py
# (escalation/human-in-the-loop), adds aura_sdk_tests/test_recommend_escalate.py,
# and adds demo_escalation.py (a deterministic scenario, since random synthetic
# data won't reliably trigger escalation).

set -euo pipefail

echo "Updating aura_sdk/recommend/personalize.py ..."
cat > aura_sdk/recommend/personalize.py << 'AURA_EOF_PERSONALIZE'
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
AURA_EOF_PERSONALIZE

echo "Creating aura_sdk/recommend/escalate.py ..."
cat > aura_sdk/recommend/escalate.py << 'AURA_EOF_ESCALATE'
"""
aura_sdk/recommend/escalate.py
──────────────────────────────────
Language: Python
Why: Steps 1-3 all end at "notify the person." That's the right amount
     of intervention for a single, resolvable episode — but if
     recommendations keep firing and Q_HCI never actually recovers,
     continuing to just nudge the same person is exactly the nagging
     failure mode this whole layer exists to avoid. This is the
     fallback for that case: escalate to a human, with consent
     configured explicitly rather than assumed.

Escalation triggers on UNRESOLVED outcomes, not on raw episode count —
it reuses the exact Q_HCI-recovery signal step 3 already computes (via
PersonalizedRecommender's on_outcome hook). A person who gets five
recommendations that each individually helped never escalates; someone
whose Q_HCI doesn't recover across several attempts in a row does —
that distinction is the actual point of escalating.

Consent is structural, not a suggestion: EscalationConfig defaults to
disabled, and enabling it without a configured contact raises rather
than silently doing nothing.

This is step 4 of the 4-part recommend/act layer:
    1. Rule-based recommendation table
    2. Plain-language explanation layer
    3. Contextual bandit for personalisation
    4. Escalation / human-in-the-loop path        ← this file
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .rules import RecommendationEngine, RecommendationEvent, ProblemClass
from .bandit import ActionBandit
from .personalize import PersonalizedRecommender
from ..hemm.extractor import EntropyFeatures


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class EscalationConfig:
    """Escalation is OFF by default. It never turns itself on, and
    turning it on without a contact configured is a programming error,
    not a silent no-op — see validate()."""
    enabled:              bool          = False
    contact_name:         Optional[str] = None
    contact_method:       Optional[str] = None   # e.g. "sms:+234...", "email:a@b.com"
    unresolved_threshold: int           = 3       # consecutive unresolved outcomes to trigger
    cooldown_frames:      int           = 9000    # ~5 min at 30fps — don't re-escalate immediately

    def validate(self) -> None:
        if self.enabled and not (self.contact_name and self.contact_method):
            raise ValueError(
                "EscalationConfig.enabled=True requires both contact_name "
                "and contact_method — escalation must never fire without "
                "an explicit, configured place to send it."
            )


@dataclass
class EscalationEvent:
    """One escalation, ready to hand to whatever actually sends the
    message (SMS gateway, email, push to a caregiver app, ...)."""
    frame_idx:         int
    unresolved_count:  int
    contact_name:      str
    contact_method:    str
    message:           str

    def to_dict(self) -> dict:
        return {
            "frame_idx":        self.frame_idx,
            "unresolved_count": self.unresolved_count,
            "contact_name":     self.contact_name,
            "contact_method":   self.contact_method,
            "message":          self.message,
        }


# ── Manager ──────────────────────────────────────────────────────────────

class EscalationManager:
    """
    Tracks a running streak of consecutive UNRESOLVED recommendation
    outcomes and raises exactly one EscalationEvent once that streak
    reaches `unresolved_threshold` — never more often than
    `cooldown_frames` allows, and never at all unless `config.enabled`.

    Usage
    ─────
        manager = EscalationManager(EscalationConfig(
            enabled=True, contact_name="Amaka",
            contact_method="sms:+234-555-0100",
        ))
        manager.record_outcome(resolved=False)   # call per resolved outcome
        event = manager.maybe_escalate(frame_idx)
    """

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config if config is not None else EscalationConfig()
        self.config.validate()
        self._unresolved_streak = 0
        self._frames_since_last_escalation = self.config.cooldown_frames

    def record_outcome(self, resolved: bool) -> None:
        """Call once per resolved recommendation outcome — `resolved`
        should be the same reward boolean the bandit received (True =
        Q_HCI recovered, False = it didn't)."""
        if resolved:
            self._unresolved_streak = 0
        else:
            self._unresolved_streak += 1

    def maybe_escalate(self, frame_idx: int) -> Optional[EscalationEvent]:
        self._frames_since_last_escalation += 1

        if not self.config.enabled:
            return None
        if self._unresolved_streak < self.config.unresolved_threshold:
            return None
        if self._frames_since_last_escalation < self.config.cooldown_frames:
            return None

        count = self._unresolved_streak
        self._frames_since_last_escalation = 0

        message = (
            f"Automated check-in: recommended breaks haven't seemed to "
            f"help {count} times in a row. It might be worth checking in."
        )

        return EscalationEvent(
            frame_idx        = frame_idx,
            unresolved_count = count,
            contact_name     = self.config.contact_name,
            contact_method   = self.config.contact_method,
            message          = message,
        )

    def reset(self) -> None:
        self._unresolved_streak = 0
        self._frames_since_last_escalation = self.config.cooldown_frames


# ── Full 4-step recommender ─────────────────────────────────────────────────

class EscalatingRecommender:
    """
    All four steps composed into one object: rule-based detection
    (step 1), plain-language rendering available via the module-level
    `explain()` function (step 2), personalized action ordering
    (step 3), and human escalation when recommendations keep failing
    to help (step 4).

    Usage
    ─────
        recommender = EscalatingRecommender(
            escalation_config=EscalationConfig(
                enabled=True, contact_name="Amaka",
                contact_method="sms:+234-555-0100",
            ),
        )
        for frame in stream:
            event = recommender.update(alert=..., state=..., q_hci=...,
                                        features=..., frame_idx=...)
            if event:
                notify_user(explain(event))
            if recommender.last_escalation:
                notify_contact(recommender.last_escalation)
    """

    def __init__(
        self,
        base_engine:        Optional[RecommendationEngine] = None,
        bandit:              Optional[ActionBandit]          = None,
        escalation_config:   Optional[EscalationConfig]      = None,
        eval_window_frames:  int   = 150,
        recovery_margin:     float = 0.05,
    ):
        self.escalation = EscalationManager(escalation_config)
        self._personalized = PersonalizedRecommender(
            base_engine        = base_engine,
            bandit             = bandit,
            eval_window_frames = eval_window_frames,
            recovery_margin    = recovery_margin,
            on_outcome         = self._on_outcome,
        )
        self.last_escalation: Optional[EscalationEvent] = None

    @property
    def bandit(self) -> ActionBandit:
        return self._personalized.bandit

    def _on_outcome(self, problem_class: ProblemClass, action: str,
                     reward: bool, frame_idx: int) -> None:
        self.escalation.record_outcome(resolved=reward)

    def update(
        self,
        alert:     bool,
        state:     Optional[int],
        q_hci:     Optional[float],
        features:  Optional[EntropyFeatures],
        frame_idx: int,
    ) -> Optional[RecommendationEvent]:
        event = self._personalized.update(alert=alert, state=state, q_hci=q_hci,
                                           features=features, frame_idx=frame_idx)
        self.last_escalation = self.escalation.maybe_escalate(frame_idx)
        return event

    def reset(self) -> None:
        self._personalized.reset()
        self.escalation.reset()
        self.last_escalation = None
AURA_EOF_ESCALATE

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
from .bandit import ActionBandit
from .personalize import PersonalizedRecommender
from .escalate import EscalationConfig, EscalationEvent, EscalationManager, EscalatingRecommender

__all__ = [
    "RecommendationEngine", "RecommendationEvent", "ProblemClass",
    "ACTION_LIBRARY", "WHY_TEMPLATES",
    "explain", "Notification", "Severity",
    "ActionBandit", "PersonalizedRecommender",
    "EscalationConfig", "EscalationEvent", "EscalationManager", "EscalatingRecommender",
]
AURA_EOF_INIT

echo "Creating aura_sdk_tests/test_recommend_escalate.py ..."
cat > aura_sdk_tests/test_recommend_escalate.py << 'AURA_EOF_TESC'
"""
aura_sdk_tests/test_recommend_escalate.py
──────────────────────────────────────────────
Language: Python (unittest)
Why: Escalation is the highest-stakes piece of this whole layer — it
reaches a second person. These tests focus hardest on the two
guarantees that matter most: it is OFF unless explicitly configured
with a contact, and it fires on genuinely unresolved outcomes (not
raw episode count), with both a trigger threshold and a cooldown so
it can't spam the contact either.

Run: python -m unittest aura_sdk_tests.test_recommend_escalate -v
"""
from __future__ import annotations
import unittest

from aura_sdk.recommend import (
    RecommendationEngine, ProblemClass, ACTION_LIBRARY,
    EscalationConfig, EscalationManager, EscalatingRecommender,
)


# ── EscalationConfig / EscalationManager ─────────────────────────────────

class TestEscalationConfigConsent(unittest.TestCase):

    def test_disabled_by_default(self):
        config = EscalationConfig()
        self.assertFalse(config.enabled)
        config.validate()  # should not raise — disabled needs no contact

    def test_enabling_without_contact_raises(self):
        config = EscalationConfig(enabled=True)
        with self.assertRaises(ValueError):
            config.validate()

    def test_enabling_with_only_name_raises(self):
        config = EscalationConfig(enabled=True, contact_name="Amaka")
        with self.assertRaises(ValueError):
            config.validate()

    def test_enabling_with_full_contact_is_valid(self):
        config = EscalationConfig(enabled=True, contact_name="Amaka",
                                   contact_method="sms:+234-555-0100")
        config.validate()  # should not raise


class TestEscalationManagerTrigger(unittest.TestCase):

    def _enabled_manager(self, threshold=3, cooldown=0):
        config = EscalationConfig(
            enabled=True, contact_name="Amaka", contact_method="sms:+234-555-0100",
            unresolved_threshold=threshold, cooldown_frames=cooldown,
        )
        return EscalationManager(config)

    def test_never_escalates_while_disabled(self):
        manager = EscalationManager(EscalationConfig())  # disabled
        for _ in range(50):
            manager.record_outcome(resolved=False)
        self.assertIsNone(manager.maybe_escalate(frame_idx=1000))

    def test_escalates_after_threshold_unresolved_outcomes(self):
        manager = self._enabled_manager(threshold=3, cooldown=0)
        for _ in range(2):
            manager.record_outcome(resolved=False)
        self.assertIsNone(manager.maybe_escalate(frame_idx=1))
        manager.record_outcome(resolved=False)
        event = manager.maybe_escalate(frame_idx=2)
        self.assertIsNotNone(event)
        self.assertEqual(event.unresolved_count, 3)

    def test_resolved_outcome_resets_the_streak(self):
        manager = self._enabled_manager(threshold=3, cooldown=0)
        manager.record_outcome(resolved=False)
        manager.record_outcome(resolved=False)
        manager.record_outcome(resolved=True)   # breaks the streak
        manager.record_outcome(resolved=False)
        self.assertIsNone(manager.maybe_escalate(frame_idx=1))

    def test_cooldown_suppresses_immediate_reescalation(self):
        manager = self._enabled_manager(threshold=2, cooldown=100)
        manager.record_outcome(resolved=False)
        manager.record_outcome(resolved=False)
        first = manager.maybe_escalate(frame_idx=1)
        self.assertIsNotNone(first)

        manager.record_outcome(resolved=False)
        manager.record_outcome(resolved=False)
        second = manager.maybe_escalate(frame_idx=2)
        self.assertIsNone(second)  # still within cooldown

    def test_escalates_again_after_cooldown_elapses(self):
        manager = self._enabled_manager(threshold=2, cooldown=5)
        manager.record_outcome(resolved=False)
        manager.record_outcome(resolved=False)
        first = manager.maybe_escalate(frame_idx=1)
        self.assertIsNotNone(first)

        manager.record_outcome(resolved=False)
        manager.record_outcome(resolved=False)
        # maybe_escalate is called once per pipeline frame in real use,
        # so the cooldown is measured in calls, not in frame_idx deltas —
        # advance it call-by-call rather than jumping frame_idx.
        for i in range(4):
            self.assertIsNone(manager.maybe_escalate(frame_idx=2 + i))
        second = manager.maybe_escalate(frame_idx=6)
        self.assertIsNotNone(second)

    def test_event_to_dict_is_json_safe(self):
        manager = self._enabled_manager(threshold=1, cooldown=0)
        manager.record_outcome(resolved=False)
        event = manager.maybe_escalate(frame_idx=7)
        d = event.to_dict()
        self.assertEqual(d["frame_idx"], 7)
        self.assertEqual(d["contact_name"], "Amaka")
        self.assertIn("check-in", d["message"])


# ── EscalatingRecommender (full stack) ────────────────────────────────────

def feed(recommender, n, alert=True, state=1, q_hci=0.1, start_frame=1):
    result = None
    for i in range(n):
        result = recommender.update(alert=alert, state=state, q_hci=q_hci,
                                     features=None, frame_idx=start_frame + i)
    return result


class TestEscalatingRecommenderDisabledByDefault(unittest.TestCase):

    def test_no_escalation_without_config(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        recommender = EscalatingRecommender(base_engine=engine, eval_window_frames=5)

        frame = 1
        for _ in range(10):
            feed(recommender, 3, q_hci=0.10, start_frame=frame)
            frame += 3
            # Feed a low q_hci reading past the eval window — never recovers.
            recommender.update(alert=True, state=1, q_hci=0.10,
                                features=None, frame_idx=frame + 5)
            frame += 6
            self.assertIsNone(recommender.last_escalation)


class TestEscalatingRecommenderTriggersOnUnresolvedStreak(unittest.TestCase):

    def _make(self, threshold=2, cooldown=0):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        config = EscalationConfig(
            enabled=True, contact_name="Amaka", contact_method="sms:+234-555-0100",
            unresolved_threshold=threshold, cooldown_frames=cooldown,
        )
        return EscalatingRecommender(base_engine=engine, escalation_config=config,
                                      eval_window_frames=5, recovery_margin=0.05)

    def test_escalates_when_qhci_never_recovers(self):
        recommender = self._make(threshold=2, cooldown=0)
        frame = 1
        escalated = False
        for _ in range(6):
            feed(recommender, 3, q_hci=0.10, start_frame=frame)
            frame += 3
            # Never recovers — well within the 0.05 margin, no improvement.
            recommender.update(alert=True, state=1, q_hci=0.10,
                                features=None, frame_idx=frame + 5)
            frame += 6
            if recommender.last_escalation is not None:
                escalated = True
                break
        self.assertTrue(escalated)

    def test_no_escalation_when_qhci_recovers_each_time(self):
        recommender = self._make(threshold=2, cooldown=0)
        frame = 1
        for _ in range(6):
            feed(recommender, 3, q_hci=0.10, start_frame=frame)
            frame += 3
            # Clear recovery each time — never builds an unresolved streak.
            recommender.update(alert=False, state=5, q_hci=0.90,
                                features=None, frame_idx=frame + 5)
            frame += 6
            self.assertIsNone(recommender.last_escalation)


class TestEscalatingRecommenderReset(unittest.TestCase):

    def test_reset_clears_escalation_state_but_keeps_bandit(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        config = EscalationConfig(
            enabled=True, contact_name="Amaka", contact_method="sms:+234-555-0100",
            unresolved_threshold=1, cooldown_frames=0,
        )
        recommender = EscalatingRecommender(base_engine=engine, escalation_config=config,
                                             eval_window_frames=5)
        bandit_before = recommender.bandit

        feed(recommender, 3, q_hci=0.10, start_frame=1)
        recommender.update(alert=True, state=1, q_hci=0.10, features=None, frame_idx=9)
        self.assertIsNotNone(recommender.last_escalation)

        recommender.reset()
        self.assertIsNone(recommender.last_escalation)
        self.assertIs(recommender.bandit, bandit_before)


if __name__ == "__main__":
    unittest.main()
AURA_EOF_TESC

echo "Creating demo_escalation.py ..."
cat > demo_escalation.py << 'AURA_EOF_DEMOESC'
"""
Demo: the full 4-step recommend/act layer, driven directly (not via
AuraPipeline) so the scenario is deterministic and legible: Q_HCI stays
persistently low and never recovers, showing recommendations fire at a
realistic pace (respecting cooldown) and escalation eventually trigger.
Random synthetic sessions won't reliably demonstrate this — this demo
exists specifically to make step 4 visible.
"""
from aura_sdk.recommend import (
    RecommendationEngine, explain, EscalationConfig, EscalatingRecommender,
)

recommender = EscalatingRecommender(
    base_engine=RecommendationEngine(sustain_frames=5, cooldown_frames=30),
    escalation_config=EscalationConfig(
        enabled=True,
        contact_name="Amaka",
        contact_method="sms:+234-555-0100",
        unresolved_threshold=3,
        cooldown_frames=200,
    ),
    eval_window_frames=15,
    recovery_margin=0.05,
)

N_FRAMES = 400
for frame in range(1, N_FRAMES + 1):
    # Q_HCI stays low and never recovers for the whole session — the
    # scenario this demo is built to show.
    event = recommender.update(alert=True, state=1, q_hci=0.10,
                                features=None, frame_idx=frame)
    if event:
        n = explain(event, frame_dt_ms=33.3)
        print(f"[frame {frame:4d}] Recommendation: {n.title} "
              f"— leading action: \"{n.suggested_actions[0]}\"")
    if recommender.last_escalation:
        e = recommender.last_escalation
        print(f"[frame {frame:4d}] 🚨 ESCALATION -> {e.contact_name} "
              f"({e.contact_method})")
        print(f"              \"{e.message}\"")
AURA_EOF_DEMOESC

echo ""
echo "Done. Running the full test suite now..."
python3 -m unittest discover -s aura_sdk_tests -v
