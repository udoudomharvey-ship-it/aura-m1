#!/usr/bin/env bash
# Run from the ROOT of your aura-m1 project, AFTER steps 1 and 2.
#   bash setup_recommend_layer_step3.sh
#
# Adds aura_sdk/recommend/bandit.py (Thompson-sampling personalization)
# and personalize.py (the Q_HCI-recovery reward loop around it), updates
# recommend/__init__.py to export them, adds two new test files, and
# refreshes demo_recommend.py to run through the personalized recommender.

set -euo pipefail

echo "Creating aura_sdk/recommend/bandit.py ..."
cat > aura_sdk/recommend/bandit.py << 'AURA_EOF_BANDIT'
"""
aura_sdk/recommend/bandit.py
──────────────────────────────
Language: Python
Why: Step 1 picks WHAT the problem is via fixed rules and offers a
     fixed list of actions. This module learns, per person, per
     problem class, which of those already-vetted actions actually
     helps — using Q_HCI recovery as a reward signal that's already
     flowing through the pipeline for free, no labeling required.

Thompson sampling multi-armed bandit: one independent set of arms per
ProblemClass, and the arms ARE the actions already defined in
ACTION_LIBRARY (step 1) — this never invents a new action, it only
learns which existing one to suggest first.

This is step 3 of the 4-part recommend/act layer:
    1. Rule-based recommendation table
    2. Plain-language explanation layer
    3. Contextual bandit for personalisation      ← this file
    4. Escalation / human-in-the-loop path
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from .rules import ProblemClass, ACTION_LIBRARY


@dataclass
class _ArmStats:
    """Beta(alpha, beta) posterior over one action's success rate.
    Starts uniform (alpha=beta=1) — no action is favored until evidence
    comes in."""
    alpha: float = 1.0   # roughly: successes + 1
    beta:  float = 1.0   # roughly: failures + 1

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, reward: bool) -> None:
        if reward:
            self.alpha += 1.0
        else:
            self.beta += 1.0


class ActionBandit:
    """
    Thompson-sampling bandit over ACTION_LIBRARY's actions, one set of
    arms per ProblemClass.

    A `rng` can be injected for deterministic tests; production code
    should leave it as the default (a fresh, unseeded Random).

    Usage
    ─────
        bandit = ActionBandit()
        action = bandit.choose(ProblemClass.COGNITIVE_OVERLOAD)
        # ... later, once you know whether it helped ...
        bandit.update(ProblemClass.COGNITIVE_OVERLOAD, action, reward=True)
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng if rng is not None else random.Random()
        self._arms: Dict[Tuple[ProblemClass, str], _ArmStats] = {
            (pc, action): _ArmStats()
            for pc, actions in ACTION_LIBRARY.items()
            for action in actions
        }

    def choose(self, problem_class: ProblemClass) -> str:
        """Sample each arm's posterior; return the action with the
        highest sample — the standard Thompson sampling decision rule."""
        candidates = [
            (action, self._arms[(problem_class, action)].sample(self._rng))
            for action in ACTION_LIBRARY[problem_class]
        ]
        return max(candidates, key=lambda kv: kv[1])[0]

    def ranked(self, problem_class: ProblemClass) -> List[str]:
        """All actions for this class, best-estimated-mean first.
        Deterministic (unlike `choose`) — useful for display or tests."""
        return sorted(
            ACTION_LIBRARY[problem_class],
            key=lambda a: self._arms[(problem_class, a)].mean(),
            reverse=True,
        )

    def update(self, problem_class: ProblemClass, action: str, reward: bool) -> None:
        key = (problem_class, action)
        if key not in self._arms:
            # ACTION_LIBRARY grew since this bandit was created — add a
            # fresh arm rather than raising.
            self._arms[key] = _ArmStats()
        self._arms[key].update(reward)

    # ── Persistence ──────────────────────────────────────────────────────
    # Personalization only means anything if it survives across sessions —
    # this lets a companion app save/restore learned preferences to disk.

    def to_dict(self) -> dict:
        return {
            f"{pc.value}::{action}": {"alpha": arm.alpha, "beta": arm.beta}
            for (pc, action), arm in self._arms.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionBandit":
        bandit = cls()
        for key, stats in data.items():
            pc_value, action = key.split("::", 1)
            bandit._arms[(ProblemClass(pc_value), action)] = _ArmStats(
                alpha=stats["alpha"], beta=stats["beta"],
            )
        return bandit

    def __repr__(self):
        return f"ActionBandit({len(self._arms)} arms)"
AURA_EOF_BANDIT

echo "Creating aura_sdk/recommend/personalize.py ..."
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
from typing import Optional, List

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
    ):
        assert eval_window_frames >= 1
        self.engine = base_engine if base_engine is not None else RecommendationEngine()
        self.bandit = bandit      if bandit      is not None else ActionBandit()
        self.eval_window_frames = eval_window_frames
        self.recovery_margin    = recovery_margin

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

__all__ = [
    "RecommendationEngine", "RecommendationEvent", "ProblemClass",
    "ACTION_LIBRARY", "WHY_TEMPLATES",
    "explain", "Notification", "Severity",
    "ActionBandit", "PersonalizedRecommender",
]
AURA_EOF_INIT

echo "Creating aura_sdk_tests/test_recommend_bandit.py ..."
cat > aura_sdk_tests/test_recommend_bandit.py << 'AURA_EOF_TB'
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
AURA_EOF_TB

echo "Creating aura_sdk_tests/test_recommend_personalize.py ..."
cat > aura_sdk_tests/test_recommend_personalize.py << 'AURA_EOF_TP'
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
AURA_EOF_TP

echo "Refreshing demo_recommend.py ..."
cat > demo_recommend.py << 'AURA_EOF_DEMO3'
"""
Demo: recommend/act layer running on top of the existing
detect/measure/alert pipeline, using the Hybrid RF+Thermal harvester
config — step 1 (rules) + step 2 (plain language) + step 3
(personalization). Since AuraPipeline's `recommender=` param expects
something with an `update()`/`reset()` interface, PersonalizedRecommender
is a drop-in replacement for the plain RecommendationEngine.
"""
from aura_sdk.pipeline import AuraPipeline, PipelineConfig
from aura_sdk.sensors import SyntheticSource
from aura_sdk.harvesting import RFHarvester, ThermalHarvester, HybridHarvester
from aura_sdk.recommend import explain
from aura_sdk.recommend.personalize import PersonalizedRecommender

recommender = PersonalizedRecommender(eval_window_frames=60, recovery_margin=0.05)

pipeline = AuraPipeline(
    source=SyntheticSource(n_frames=1200, seed=42),
    harvester=HybridHarvester(sources=[RFHarvester(), ThermalHarvester()]),
    sinks=[],
    config=PipelineConfig(verbose=False),
    recommender=recommender,
)

pipeline.open()
notifications = []
for frame in pipeline.source.frames():
    tel = pipeline.process_frame(frame)
    if pipeline.last_recommendation:
        notifications.append(explain(pipeline.last_recommendation,
                                      frame_dt_ms=pipeline.config.frame_dt_ms))
pipeline.close()

print(f"{len(notifications)} notification(s) over 1200 frames:\n")
for n in notifications:
    print(f"  [{n.severity.value.upper():8s}] {n.title}")
    print(f"      {n.body}")
    for a in n.suggested_actions:
        print(f"      → {a}")
    print()

print("── Bandit state after this session ──")
from aura_sdk.recommend import ProblemClass, ACTION_LIBRARY
for pc in ProblemClass:
    ranked = recommender.bandit.ranked(pc)
    if ranked != ACTION_LIBRARY[pc]:
        print(f"  {pc.value}: learned order -> {ranked}")
AURA_EOF_DEMO3

echo ""
echo "Done. Running the full test suite now..."
python3 -m unittest discover -s aura_sdk_tests -v
