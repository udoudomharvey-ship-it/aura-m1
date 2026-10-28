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
