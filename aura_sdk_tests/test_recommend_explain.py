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
