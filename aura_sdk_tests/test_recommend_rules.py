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
