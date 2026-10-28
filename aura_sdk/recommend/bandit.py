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
