"""
aura_sdk_tests/test_recommend_personalize.py
────────────────────────────────────────────────
Language: Python (unittest)
Why: Tests the reward loop PersonalizedRecommender adds around
RecommendationEngine + ActionBandit: does it correctly detect Q_HCI
recovery within the eval window and feed that back? A fake bandit
double (not the real, stochastic ActionBandit) is used throughout so
these tests are fully deterministic and isolate the reward-loop logic
from Thompson sampling itself (which is covered separately in
test_recommend_bandit.py).

Run: python -m unittest aura_sdk_tests.test_recommend_personalize -v
"""
from __future__ import annotations
import unittest

from aura_sdk.recommend import RecommendationEngine, ProblemClass, ACTION_LIBRARY
from aura_sdk.recommend.personalize import PersonalizedRecommender


class _FixedBandit:
    """Test double: always chooses a fixed action, records every
    update() call so tests can assert on exactly what reward was given."""

    def __init__(self, fixed_action: str):
        self.fixed_action = fixed_action
        self.calls = []  # list of (problem_class, action, reward)

    def choose(self, problem_class):
        return self.fixed_action

    def update(self, problem_class, action, reward):
        self.calls.append((problem_class, action, reward))


def feed(recommender, n, alert=True, state=1, q_hci=0.1, start_frame=1):
    result = None
    for i in range(n):
        result = recommender.update(alert=alert, state=state, q_hci=q_hci,
                                     features=None, frame_idx=start_frame + i)
    return result


class TestDelegatesSustainAndCooldown(unittest.TestCase):

    def test_no_recommendation_before_sustain_threshold(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        bandit = _FixedBandit(ACTION_LIBRARY[ProblemClass.COGNITIVE_OVERLOAD][0])
        recommender = PersonalizedRecommender(base_engine=engine, bandit=bandit)
        result = feed(recommender, 4)
        self.assertIsNone(result)

    def test_fires_on_reaching_sustain_threshold(self):
        engine = RecommendationEngine(sustain_frames=5, cooldown_frames=0)
        bandit = _FixedBandit(ACTION_LIBRARY[ProblemClass.SUSTAINED_LOW_QUALITY][0])
        recommender = PersonalizedRecommender(base_engine=engine, bandit=bandit)
        result = feed(recommender, 5)
        self.assertIsNotNone(result)


class TestActionOrdering(unittest.TestCase):

    def test_bandit_chosen_action_leads_the_list(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        chosen = ACTION_LIBRARY[ProblemClass.SUSTAINED_LOW_QUALITY][1]  # not the library's first
        bandit = _FixedBandit(chosen)
        recommender = PersonalizedRecommender(base_engine=engine, bandit=bandit)
        result = feed(recommender, 3)
        self.assertEqual(result.suggested_actions[0], chosen)

    def test_no_actions_outside_the_original_library_are_introduced(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        bandit = _FixedBandit(ACTION_LIBRARY[ProblemClass.SUSTAINED_LOW_QUALITY][0])
        recommender = PersonalizedRecommender(base_engine=engine, bandit=bandit)
        result = feed(recommender, 3)
        self.assertEqual(
            set(result.suggested_actions),
            set(ACTION_LIBRARY[result.problem_class]),
        )


class TestRewardLoop(unittest.TestCase):

    def test_recovery_within_window_scores_as_reward_true(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        chosen = ACTION_LIBRARY[ProblemClass.SUSTAINED_LOW_QUALITY][0]
        bandit = _FixedBandit(chosen)
        recommender = PersonalizedRecommender(
            base_engine=engine, bandit=bandit,
            eval_window_frames=10, recovery_margin=0.05,
        )
        event = feed(recommender, 3, q_hci=0.10, start_frame=1)
        self.assertIsNotNone(event)
        # event fired on frame 3 (sustain_frames=3, starting at frame 1),
        # so the eval window closes at frame 3 + 10 = 13.
        recommender.update(alert=False, state=5, q_hci=0.80,
                            features=None, frame_idx=event.frame_idx + 10)
        self.assertEqual(len(bandit.calls), 1)
        self.assertTrue(bandit.calls[0][2])  # reward == True

    def test_no_recovery_within_window_scores_as_reward_false(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        chosen = ACTION_LIBRARY[ProblemClass.SUSTAINED_LOW_QUALITY][0]
        bandit = _FixedBandit(chosen)
        recommender = PersonalizedRecommender(
            base_engine=engine, bandit=bandit,
            eval_window_frames=10, recovery_margin=0.05,
        )
        event = feed(recommender, 3, q_hci=0.10, start_frame=1)
        self.assertIsNotNone(event)
        # Still low — no meaningful recovery.
        recommender.update(alert=True, state=1, q_hci=0.11,
                            features=None, frame_idx=event.frame_idx + 10)
        self.assertEqual(len(bandit.calls), 1)
        self.assertFalse(bandit.calls[0][2])  # reward == False

    def test_outcome_is_not_scored_before_the_window_elapses(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        bandit = _FixedBandit(ACTION_LIBRARY[ProblemClass.SUSTAINED_LOW_QUALITY][0])
        recommender = PersonalizedRecommender(
            base_engine=engine, bandit=bandit,
            eval_window_frames=50, recovery_margin=0.05,
        )
        feed(recommender, 3, q_hci=0.10, start_frame=1)
        # Well before eval_frame (1 + 50) — should not score yet.
        recommender.update(alert=False, state=5, q_hci=0.90,
                            features=None, frame_idx=10)
        self.assertEqual(len(bandit.calls), 0)


class TestReset(unittest.TestCase):

    def test_reset_clears_pending_but_preserves_the_bandit(self):
        engine = RecommendationEngine(sustain_frames=3, cooldown_frames=0)
        bandit = _FixedBandit(ACTION_LIBRARY[ProblemClass.SUSTAINED_LOW_QUALITY][0])
        recommender = PersonalizedRecommender(
            base_engine=engine, bandit=bandit, eval_window_frames=1000,
        )
        feed(recommender, 3, q_hci=0.10, start_frame=1)
        self.assertEqual(len(recommender._pending), 1)

        recommender.reset()
        self.assertEqual(len(recommender._pending), 0)
        # The bandit object itself must be the same instance — reset()
        # must never replace or clear learned preferences.
        self.assertIs(recommender.bandit, bandit)


if __name__ == "__main__":
    unittest.main()
