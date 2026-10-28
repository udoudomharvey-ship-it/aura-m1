"""
aura_sdk_tests/test_recommend_bandit.py
──────────────────────────────────────────
Language: Python (unittest)
Why: `choose()` is intentionally stochastic (that's the point of
Thompson sampling), so these tests avoid asserting on its output
directly and instead test the deterministic pieces: `ranked()` (mean-
based ordering), `update()`'s bookkeeping, and to_dict/from_dict
persistence round-tripping — since a bandit that can't be saved never
actually personalizes anything across sessions.

Run: python -m unittest aura_sdk_tests.test_recommend_bandit -v
"""
from __future__ import annotations
import random
import unittest

from aura_sdk.recommend import ActionBandit, ProblemClass, ACTION_LIBRARY


class TestChoose(unittest.TestCase):

    def test_choose_always_returns_a_valid_action_for_the_class(self):
        bandit = ActionBandit(rng=random.Random(0))
        for pc in ProblemClass:
            for _ in range(20):
                action = bandit.choose(pc)
                self.assertIn(action, ACTION_LIBRARY[pc])


class TestRankedOrdering(unittest.TestCase):

    def test_ranked_is_uniform_before_any_evidence(self):
        bandit = ActionBandit()
        pc = ProblemClass.COGNITIVE_OVERLOAD
        # No updates yet — every arm has the same Beta(1,1) mean (0.5),
        # so `ranked` should just return the library's own order.
        self.assertEqual(bandit.ranked(pc), ACTION_LIBRARY[pc])

    def test_rewarded_action_rises_to_the_top(self):
        bandit = ActionBandit()
        pc = ProblemClass.COGNITIVE_OVERLOAD
        actions = ACTION_LIBRARY[pc]
        loser, winner = actions[0], actions[1]

        for _ in range(10):
            bandit.update(pc, winner, reward=True)
        for _ in range(10):
            bandit.update(pc, loser, reward=False)

        self.assertEqual(bandit.ranked(pc)[0], winner)

    def test_ranking_is_independent_per_problem_class(self):
        bandit = ActionBandit()
        pc_a = ProblemClass.SPATIAL_RESTLESSNESS
        pc_b = ProblemClass.TEMPORAL_ERRATICNESS
        action_a = ACTION_LIBRARY[pc_a][1]
        for _ in range(10):
            bandit.update(pc_a, action_a, reward=True)
        # pc_b's ranking should be untouched by pc_a's updates.
        self.assertEqual(bandit.ranked(pc_b), ACTION_LIBRARY[pc_b])


class TestUpdate(unittest.TestCase):

    def test_reward_true_increases_mean(self):
        bandit = ActionBandit()
        pc = ProblemClass.SUSTAINED_LOW_QUALITY
        action = ACTION_LIBRARY[pc][0]
        before = bandit._arms[(pc, action)].mean()
        bandit.update(pc, action, reward=True)
        after = bandit._arms[(pc, action)].mean()
        self.assertGreater(after, before)

    def test_reward_false_decreases_mean(self):
        bandit = ActionBandit()
        pc = ProblemClass.SUSTAINED_LOW_QUALITY
        action = ACTION_LIBRARY[pc][0]
        before = bandit._arms[(pc, action)].mean()
        bandit.update(pc, action, reward=False)
        after = bandit._arms[(pc, action)].mean()
        self.assertLess(after, before)

    def test_update_on_unknown_action_does_not_raise(self):
        bandit = ActionBandit()
        pc = ProblemClass.SUSTAINED_LOW_QUALITY
        bandit.update(pc, "A brand new action not in the library", reward=True)
        # Should have been added as a fresh arm rather than raising.
        self.assertIn((pc, "A brand new action not in the library"), bandit._arms)


class TestPersistence(unittest.TestCase):

    def test_round_trip_preserves_ranking(self):
        bandit = ActionBandit()
        pc = ProblemClass.COGNITIVE_OVERLOAD
        winner = ACTION_LIBRARY[pc][1]
        for _ in range(15):
            bandit.update(pc, winner, reward=True)

        restored = ActionBandit.from_dict(bandit.to_dict())
        self.assertEqual(restored.ranked(pc), bandit.ranked(pc))

    def test_to_dict_is_json_safe(self):
        import json
        bandit = ActionBandit()
        bandit.update(ProblemClass.COGNITIVE_OVERLOAD,
                       ACTION_LIBRARY[ProblemClass.COGNITIVE_OVERLOAD][0],
                       reward=True)
        json.dumps(bandit.to_dict())  # raises if not serializable


if __name__ == "__main__":
    unittest.main()
