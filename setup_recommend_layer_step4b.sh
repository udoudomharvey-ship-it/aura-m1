#!/usr/bin/env bash
# Run from the ROOT of your aura-m1 project, AFTER the original step 4
# script has already been applied.
#   bash setup_recommend_layer_step4b.sh
#
# Replaces the single-contact escalate.py with a TIERED version
# (multiple contacts, each with its own trigger threshold — e.g.
# partner at 3 unresolved, clinician at 8), adds JSON save/load for
# EscalationConfig, updates __init__.py, rewrites the test file for
# the new API, and refreshes demo_escalation.py to show both tiers
# firing in order.

set -euo pipefail

echo "Updating aura_sdk/recommend/escalate.py (tiered contacts) ..."
cat > aura_sdk/recommend/escalate.py << 'AURA_EOF_ESCALATE2'
"""
aura_sdk/recommend/escalate.py
──────────────────────────────────
Language: Python
Why: Steps 1-3 all end at "notify the person." That's the right amount
     of intervention for a single, resolvable episode — but if
     recommendations keep firing and Q_HCI never actually recovers,
     continuing to just nudge the same person is exactly the nagging
     failure mode this whole layer exists to avoid. This is the
     fallback for that case: escalate to a human — potentially several,
     in order — with consent configured explicitly rather than assumed.

Escalation triggers on UNRESOLVED outcomes, not on raw episode count —
it reuses the exact Q_HCI-recovery signal step 3 already computes (via
PersonalizedRecommender's on_outcome hook). A person who gets five
recommendations that each individually helped never escalates; someone
whose Q_HCI doesn't recover across several attempts in a row does —
that distinction is the actual point of escalating.

Contacts are TIERED: each has its own unresolved-count threshold, so a
typical setup is "partner at 3 unresolved, clinician at 8" — a longer
struggle escalates further up the chain rather than re-notifying the
same person every time. Once the streak resolves (Q_HCI recovers),
the whole chain resets — the next episode starts at tier 1 again.

Consent is structural, not a suggestion: EscalationConfig defaults to
disabled and to zero contacts, and enabling it with no contacts raises
rather than silently doing nothing. Contacts can be saved/loaded as
JSON so a companion app's setup flow owns collecting and storing them —
this module never invents or guesses a contact.

This is step 4 of the 4-part recommend/act layer:
    1. Rule-based recommendation table
    2. Plain-language explanation layer
    3. Contextual bandit for personalisation
    4. Escalation / human-in-the-loop path        ← this file
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional, List, Set

from .rules import RecommendationEngine, RecommendationEvent, ProblemClass
from .bandit import ActionBandit
from .personalize import PersonalizedRecommender
from ..hemm.extractor import EntropyFeatures


# ── Contacts ─────────────────────────────────────────────────────────────

@dataclass
class EscalationContact:
    """One entry in the escalation chain. `threshold` is the consecutive
    unresolved-outcome count at which THIS contact gets notified —
    contacts should be given ascending thresholds (nearer/lighter-touch
    contact first, e.g. a partner at 3, a clinician at 8)."""
    name:      str
    method:    str    # e.g. "sms:+234-555-0100", "email:a@b.com"
    threshold: int

    def to_dict(self) -> dict:
        return {"name": self.name, "method": self.method, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, data: dict) -> "EscalationContact":
        return cls(name=data["name"], method=data["method"], threshold=data["threshold"])


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class EscalationConfig:
    """Escalation is OFF by default and requires at least one contact.
    Enabling it with an empty contact list is a programming error, not
    a silent no-op — see validate(). This is meant to be filled in by a
    companion app's setup flow (see to_dict/from_dict/save/load) rather
    than hand-authored per session."""
    enabled:          bool                       = False
    contacts:         List[EscalationContact]    = field(default_factory=list)
    cooldown_frames:  int                        = 9000   # ~5 min at 30fps between any two escalations

    def validate(self) -> None:
        if self.enabled and not self.contacts:
            raise ValueError(
                "EscalationConfig.enabled=True requires at least one "
                "contact — escalation must never fire without an "
                "explicit, configured place to send it."
            )
        thresholds = [c.threshold for c in self.contacts]
        if thresholds != sorted(thresholds):
            raise ValueError(
                "EscalationConfig.contacts must be ordered by ascending "
                "threshold (nearer contact first, e.g. partner then "
                "clinician) — got thresholds " + str(thresholds)
            )

    # ── Persistence ──────────────────────────────────────────────────────
    # A companion app's setup flow owns collecting contacts from the
    # user; this just needs to survive being written to and read back
    # from disk between sessions.

    def to_dict(self) -> dict:
        return {
            "enabled":         self.enabled,
            "contacts":        [c.to_dict() for c in self.contacts],
            "cooldown_frames": self.cooldown_frames,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EscalationConfig":
        return cls(
            enabled         = data.get("enabled", False),
            contacts        = [EscalationContact.from_dict(c) for c in data.get("contacts", [])],
            cooldown_frames = data.get("cooldown_frames", 9000),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EscalationConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class EscalationEvent:
    """One escalation, ready to hand to whatever actually sends the
    message (SMS gateway, email, push to a caregiver app, ...)."""
    frame_idx:         int
    unresolved_count:  int
    tier:              int     # index into config.contacts — 0 = first/nearest contact
    contact_name:      str
    contact_method:    str
    message:           str

    def to_dict(self) -> dict:
        return {
            "frame_idx":        self.frame_idx,
            "unresolved_count": self.unresolved_count,
            "tier":             self.tier,
            "contact_name":     self.contact_name,
            "contact_method":   self.contact_method,
            "message":          self.message,
        }


# ── Manager ──────────────────────────────────────────────────────────────

class EscalationManager:
    """
    Tracks a running streak of consecutive UNRESOLVED recommendation
    outcomes and, as that streak clears each contact's threshold in
    turn, raises one EscalationEvent per newly-reached tier — never
    more often than `cooldown_frames` allows between any two
    escalations, and never at all unless `config.enabled`.

    Once the streak resolves (a recommendation's Q_HCI recovers), the
    whole chain resets: the next episode starts at tier 0 again, even
    if a later tier was already notified this time.

    Usage
    ─────
        manager = EscalationManager(EscalationConfig(
            enabled=True,
            contacts=[
                EscalationContact("Amaka", "sms:+234-555-0100", threshold=3),
                EscalationContact("Dr. Bello", "email:bello@clinic.example", threshold=8),
            ],
        ))
        manager.record_outcome(resolved=False)   # call per resolved outcome
        event = manager.maybe_escalate(frame_idx)
    """

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config if config is not None else EscalationConfig()
        self.config.validate()
        self._unresolved_streak = 0
        self._frames_since_last_escalation = self.config.cooldown_frames
        self._notified_tiers: Set[int] = set()

    def record_outcome(self, resolved: bool) -> None:
        """Call once per resolved recommendation outcome — `resolved`
        should be the same reward boolean the bandit received (True =
        Q_HCI recovered, False = it didn't)."""
        if resolved:
            self._unresolved_streak = 0
            self._notified_tiers.clear()   # fully resolved — next episode starts at tier 0
        else:
            self._unresolved_streak += 1

    def maybe_escalate(self, frame_idx: int) -> Optional[EscalationEvent]:
        self._frames_since_last_escalation += 1

        if not self.config.enabled:
            return None
        if self._frames_since_last_escalation < self.config.cooldown_frames:
            return None

        # Notify the highest tier whose threshold has been reached and
        # hasn't already been notified this episode — i.e. escalate
        # forward through the chain, never re-notify the same tier.
        target = None
        for i, contact in enumerate(self.config.contacts):
            if self._unresolved_streak >= contact.threshold and i not in self._notified_tiers:
                target = i
        if target is None:
            return None

        contact = self.config.contacts[target]
        self._notified_tiers.add(target)
        self._frames_since_last_escalation = 0

        message = (
            f"Automated check-in: recommended breaks haven't seemed to help "
            f"{self._unresolved_streak} times in a row. It might be worth checking in."
        )

        return EscalationEvent(
            frame_idx        = frame_idx,
            unresolved_count = self._unresolved_streak,
            tier             = target,
            contact_name     = contact.name,
            contact_method   = contact.method,
            message          = message,
        )

    def reset(self) -> None:
        self._unresolved_streak = 0
        self._frames_since_last_escalation = self.config.cooldown_frames
        self._notified_tiers.clear()


# ── Full 4-step recommender ─────────────────────────────────────────────────

class EscalatingRecommender:
    """
    All four steps composed into one object: rule-based detection
    (step 1), plain-language rendering available via the module-level
    `explain()` function (step 2), personalized action ordering
    (step 3), and tiered human escalation when recommendations keep
    failing to help (step 4).

    Usage
    ─────
        recommender = EscalatingRecommender(
            escalation_config=EscalationConfig(
                enabled=True,
                contacts=[
                    EscalationContact("Amaka", "sms:+234-555-0100", threshold=3),
                    EscalationContact("Dr. Bello", "email:bello@clinic.example", threshold=8),
                ],
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
AURA_EOF_ESCALATE2

echo "Updating aura_sdk/recommend/__init__.py ..."
cat > aura_sdk/recommend/__init__.py << 'AURA_EOF_INIT2'
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
from .escalate import (
    EscalationConfig, EscalationContact, EscalationEvent,
    EscalationManager, EscalatingRecommender,
)

__all__ = [
    "RecommendationEngine", "RecommendationEvent", "ProblemClass",
    "ACTION_LIBRARY", "WHY_TEMPLATES",
    "explain", "Notification", "Severity",
    "ActionBandit", "PersonalizedRecommender",
    "EscalationConfig", "EscalationContact", "EscalationEvent",
    "EscalationManager", "EscalatingRecommender",
]
AURA_EOF_INIT2

echo "Updating aura_sdk_tests/test_recommend_escalate.py ..."
cat > aura_sdk_tests/test_recommend_escalate.py << 'AURA_EOF_TESC2'
"""
aura_sdk_tests/test_recommend_escalate.py
──────────────────────────────────────────────
Language: Python (unittest)
Why: Escalation is the highest-stakes piece of this whole layer — it
reaches other people. These tests focus hardest on the guarantees that
matter most: it is OFF unless explicitly configured with contacts,
contacts escalate in ORDER as the unresolved streak grows (never
skipping or re-notifying the same tier), the whole chain resets once
the problem actually resolves, and configs round-trip through JSON
since a companion app's setup flow is expected to persist them.

Run: python -m unittest aura_sdk_tests.test_recommend_escalate -v
"""
from __future__ import annotations
import json
import os
import tempfile
import unittest

from aura_sdk.recommend import (
    RecommendationEngine, ProblemClass, ACTION_LIBRARY,
    EscalationConfig, EscalationContact, EscalationManager, EscalatingRecommender,
)


PARTNER   = EscalationContact("Amaka", "sms:+234-555-0100", threshold=3)
CLINICIAN = EscalationContact("Dr. Bello", "email:bello@clinic.example", threshold=8)


# ── EscalationConfig consent + persistence ────────────────────────────────

class TestEscalationConfigConsent(unittest.TestCase):

    def test_disabled_by_default_with_no_contacts(self):
        config = EscalationConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.contacts, [])
        config.validate()  # should not raise

    def test_enabling_with_no_contacts_raises(self):
        config = EscalationConfig(enabled=True)
        with self.assertRaises(ValueError):
            config.validate()

    def test_enabling_with_contacts_is_valid(self):
        config = EscalationConfig(enabled=True, contacts=[PARTNER, CLINICIAN])
        config.validate()  # should not raise

    def test_contacts_out_of_threshold_order_raises(self):
        config = EscalationConfig(enabled=True, contacts=[CLINICIAN, PARTNER])  # descending
        with self.assertRaises(ValueError):
            config.validate()


class TestEscalationConfigPersistence(unittest.TestCase):

    def test_to_dict_from_dict_round_trip(self):
        config = EscalationConfig(enabled=True, contacts=[PARTNER, CLINICIAN], cooldown_frames=500)
        restored = EscalationConfig.from_dict(config.to_dict())
        self.assertEqual(restored.enabled, config.enabled)
        self.assertEqual(restored.cooldown_frames, config.cooldown_frames)
        self.assertEqual([c.to_dict() for c in restored.contacts],
                          [c.to_dict() for c in config.contacts])

    def test_to_dict_is_json_safe(self):
        config = EscalationConfig(enabled=True, contacts=[PARTNER, CLINICIAN])
        json.dumps(config.to_dict())  # raises if not serializable

    def test_save_and_load_round_trip_via_file(self):
        config = EscalationConfig(enabled=True, contacts=[PARTNER, CLINICIAN])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "escalation_config.json")
            config.save(path)
            restored = EscalationConfig.load(path)
        self.assertEqual(restored.enabled, True)
        self.assertEqual(len(restored.contacts), 2)
        self.assertEqual(restored.contacts[0].name, "Amaka")
        self.assertEqual(restored.contacts[1].name, "Dr. Bello")

    def test_loading_missing_fields_falls_back_to_defaults(self):
        restored = EscalationConfig.from_dict({})
        self.assertFalse(restored.enabled)
        self.assertEqual(restored.contacts, [])


# ── EscalationManager: tiered triggering ──────────────────────────────────

class TestTieredEscalation(unittest.TestCase):

    def _manager(self, cooldown=0):
        config = EscalationConfig(enabled=True, contacts=[PARTNER, CLINICIAN],
                                   cooldown_frames=cooldown)
        return EscalationManager(config)

    def test_never_escalates_while_disabled(self):
        manager = EscalationManager(EscalationConfig())  # disabled, no contacts
        for _ in range(50):
            manager.record_outcome(resolved=False)
        self.assertIsNone(manager.maybe_escalate(frame_idx=1000))

    def test_first_tier_fires_at_its_threshold(self):
        manager = self._manager()
        for _ in range(2):
            manager.record_outcome(resolved=False)
        self.assertIsNone(manager.maybe_escalate(frame_idx=1))
        manager.record_outcome(resolved=False)  # streak == 3 == PARTNER.threshold
        event = manager.maybe_escalate(frame_idx=2)
        self.assertIsNotNone(event)
        self.assertEqual(event.tier, 0)
        self.assertEqual(event.contact_name, "Amaka")

    def test_second_tier_fires_once_its_own_threshold_is_reached(self):
        manager = self._manager()
        for _ in range(3):
            manager.record_outcome(resolved=False)
        first = manager.maybe_escalate(frame_idx=1)
        self.assertEqual(first.tier, 0)

        for _ in range(5):  # streak now 8 == CLINICIAN.threshold
            manager.record_outcome(resolved=False)
        second = manager.maybe_escalate(frame_idx=2)
        self.assertIsNotNone(second)
        self.assertEqual(second.tier, 1)
        self.assertEqual(second.contact_name, "Dr. Bello")

    def test_same_tier_is_never_notified_twice_in_one_episode(self):
        manager = self._manager()
        for _ in range(3):
            manager.record_outcome(resolved=False)
        first = manager.maybe_escalate(frame_idx=1)
        self.assertIsNotNone(first)

        # Streak keeps growing but hasn't reached tier 2's threshold yet —
        # tier 0 must not fire again.
        manager.record_outcome(resolved=False)
        second = manager.maybe_escalate(frame_idx=2)
        self.assertIsNone(second)

    def test_resolution_resets_the_whole_chain(self):
        manager = self._manager()
        for _ in range(3):
            manager.record_outcome(resolved=False)
        manager.maybe_escalate(frame_idx=1)  # notifies tier 0

        manager.record_outcome(resolved=True)  # resolved — chain resets

        for _ in range(3):
            manager.record_outcome(resolved=False)
        again = manager.maybe_escalate(frame_idx=2)
        # Tier 0 should be eligible again since the episode reset.
        self.assertIsNotNone(again)
        self.assertEqual(again.tier, 0)

    def test_cooldown_applies_between_any_two_escalations(self):
        manager = self._manager(cooldown=10)
        for _ in range(3):
            manager.record_outcome(resolved=False)
        first = manager.maybe_escalate(frame_idx=1)
        self.assertIsNotNone(first)

        for _ in range(5):  # streak reaches tier 2's threshold
            manager.record_outcome(resolved=False)
        # Still within cooldown — even though tier 2 is newly eligible.
        too_soon = manager.maybe_escalate(frame_idx=2)
        self.assertIsNone(too_soon)


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
            recommender.update(alert=True, state=1, q_hci=0.10,
                                features=None, frame_idx=frame + 5)
            frame += 6
            self.assertIsNone(recommender.last_escalation)


class TestEscalatingRecommenderTiersEndToEnd(unittest.TestCase):

    def _make(self, cooldown=0):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        config = EscalationConfig(enabled=True, contacts=[PARTNER, CLINICIAN],
                                   cooldown_frames=cooldown)
        return EscalatingRecommender(base_engine=engine, escalation_config=config,
                                      eval_window_frames=5, recovery_margin=0.05)

    def test_escalates_through_both_tiers_when_qhci_never_recovers(self):
        recommender = self._make()
        frame = 1
        tiers_seen = []
        for _ in range(10):
            feed(recommender, 3, q_hci=0.10, start_frame=frame)
            frame += 3
            recommender.update(alert=True, state=1, q_hci=0.10,
                                features=None, frame_idx=frame + 5)
            frame += 6
            if recommender.last_escalation is not None:
                tiers_seen.append(recommender.last_escalation.tier)
        self.assertIn(0, tiers_seen)
        self.assertIn(1, tiers_seen)
        # Tier 0 must be notified before tier 1.
        self.assertLess(tiers_seen.index(0), tiers_seen.index(1))

    def test_no_escalation_when_qhci_recovers_each_time(self):
        recommender = self._make()
        frame = 1
        for _ in range(10):
            feed(recommender, 3, q_hci=0.10, start_frame=frame)
            frame += 3
            recommender.update(alert=False, state=5, q_hci=0.90,
                                features=None, frame_idx=frame + 5)
            frame += 6
            self.assertIsNone(recommender.last_escalation)


class TestEscalatingRecommenderReset(unittest.TestCase):

    def test_reset_clears_escalation_state_but_keeps_bandit(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        config = EscalationConfig(enabled=True,
                                   contacts=[EscalationContact("Amaka", "sms:+234-555-0100", threshold=1)],
                                   cooldown_frames=0)
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
AURA_EOF_TESC2

echo "Refreshing demo_escalation.py ..."
cat > demo_escalation.py << 'AURA_EOF_DEMOESC2'
"""
Demo: the full 4-step recommend/act layer with a TWO-TIER escalation
chain (partner first, clinician second), driven directly (not via
AuraPipeline) so the scenario is deterministic and legible: Q_HCI stays
persistently low and never recovers, showing recommendations fire at a
realistic pace and escalation climb through both tiers in order.
"""
from aura_sdk.recommend import (
    RecommendationEngine, explain,
    EscalationConfig, EscalationContact, EscalatingRecommender,
)

recommender = EscalatingRecommender(
    base_engine=RecommendationEngine(sustain_frames=5, cooldown_frames=30),
    escalation_config=EscalationConfig(
        enabled=True,
        contacts=[
            EscalationContact("Amaka", "sms:+234-555-0100", threshold=3),
            EscalationContact("Dr. Bello", "email:bello@clinic.example", threshold=8),
        ],
        cooldown_frames=100,
    ),
    eval_window_frames=15,
    recovery_margin=0.05,
)

N_FRAMES = 500
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
        print(f"[frame {frame:4d}] 🚨 TIER {e.tier} ESCALATION -> "
              f"{e.contact_name} ({e.contact_method})")
        print(f"              \"{e.message}\"")

print()
print("── Saving/loading the escalation config, as a companion app's")
print("   setup flow would ──")
recommender.escalation.config.save("/tmp/escalation_config_demo.json")
restored = EscalationConfig.load("/tmp/escalation_config_demo.json")
for c in restored.contacts:
    print(f"  {c.name} ({c.method}) — notified at {c.threshold} unresolved outcomes")
AURA_EOF_DEMOESC2

echo ""
echo "Done. Running the full test suite now..."
python3 -m unittest discover -s aura_sdk_tests -v
