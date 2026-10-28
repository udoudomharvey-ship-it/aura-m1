"""
bandit_eval_harness.py
────────────────────────────────────────────────────────────────
Evaluates the Thompson-sampling personalization layer
(aura_sdk/recommend/{bandit,personalize}.py) in two modes:

MODE 1 -- SYNTHETIC
  You don't get ground truth from real telemetry (you never know what
  Q_HCI *would* have done if the person had ignored the suggestion).
  This mode fakes ground truth on purpose: each action gets an assigned
  "true" recovery probability, frames are simulated through the real
  PersonalizedRecommender/ActionBandit classes, and we check whether the
  bandit's `ranked()` output converges toward the action that is
  actually better -- i.e. does the learning mechanism work at all.
  It also runs a NON-COMPLIANCE robustness variant: some fraction of
  "users" ignore suggested_actions[0] and the environment recovers (or
  doesn't) independent of what was chosen, to quantify how much reward
  noise that injects.

MODE 2 -- REAL LOG
  Wire `on_outcome=` on your real PersonalizedRecommender to append each
  (problem_class, action, reward, frame_idx) tuple as one JSON line to a
  file during an actual run. Point this script at that file and it
  reports reward rate over time, per-arm pull/win counts, and whether
  ranking has stabilized -- on your real data, no simulation involved.

Run:
    python3 bandit_eval_harness.py synthetic
    python3 bandit_eval_harness.py synthetic --noncompliance 0.4
    python3 bandit_eval_harness.py real --log path/to/outcomes.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from aura_sdk.recommend import (
    ActionBandit, ProblemClass, ACTION_LIBRARY, PersonalizedRecommender,
)
from aura_sdk.hemm.extractor import EntropyFeatures


# ── Synthetic ground truth ──────────────────────────────────────────────
# For each ProblemClass, assign one action a HIGHER true recovery
# probability than the other. The bandit doesn't get to see these
# numbers -- it only sees the boolean reward personalize.py derives from
# simulated Q_HCI movement. If ranked() converges to the higher-prob
# action, the learning loop works end to end.
TRUE_RECOVERY_PROB: Dict[Tuple[ProblemClass, str], float] = {}
for pc, actions in ACTION_LIBRARY.items():
    # deterministic across runs: first action 0.35 (worse), second 0.70 (better)
    TRUE_RECOVERY_PROB[(pc, actions[0])] = 0.35
    TRUE_RECOVERY_PROB[(pc, actions[1])] = 0.70


def _features_for(problem_class: ProblemClass) -> EntropyFeatures:
    """Synthetic entropy vector that will make RecommendationEngine's
    _classify() land on the requested problem_class."""
    base = 0.2
    if problem_class == ProblemClass.SPATIAL_RESTLESSNESS:
        return EntropyFeatures(h_sp=0.9, h_te=base, h_in=base)
    if problem_class == ProblemClass.TEMPORAL_ERRATICNESS:
        return EntropyFeatures(h_sp=base, h_te=0.9, h_in=base)
    if problem_class == ProblemClass.COGNITIVE_OVERLOAD:
        return EntropyFeatures(h_sp=base, h_te=base, h_in=0.9)
    # SUSTAINED_LOW_QUALITY: no dominant term
    return EntropyFeatures(h_sp=0.3, h_te=0.3, h_in=0.3)


def run_synthetic(
    n_episodes_per_class: int = 60,
    noncompliance_rate: float = 0.0,
    seed: int = 0,
    checkpoint_every: int = 10,
) -> dict:
    """
    Drives a real PersonalizedRecommender through simulated episodes.

    One "episode" = a sustained alert firing a recommendation for one
    ProblemClass, then enough frames of simulated Q_HCI to resolve the
    outcome. noncompliance_rate is the fraction of episodes where the
    simulated person does NOT follow suggested_actions[0] -- in that
    case recovery is drawn from a fixed baseline (0.5) independent of
    which action the bandit chose, modeling reward noise from ignored
    suggestions.
    """
    rng = random.Random(seed)
    bandit_rng = random.Random(seed + 1)
    bandit = ActionBandit(rng=bandit_rng)

    outcomes: List[dict] = []  # one row per resolved outcome, in order

    def on_outcome(pc, action, reward, frame_idx):
        outcomes.append({
            "problem_class": pc.value, "action": action,
            "reward": reward, "frame_idx": frame_idx,
        })

    recommender = PersonalizedRecommender(
        bandit=bandit, eval_window_frames=150, recovery_margin=0.05,
        on_outcome=on_outcome,
    )

    frame_idx = 0
    checkpoints = []  # (episode_num, {pc: ranked_list})

    classes = list(ProblemClass)
    total_episodes = n_episodes_per_class * len(classes)
    episode_order = classes * n_episodes_per_class
    rng.shuffle(episode_order)

    for ep_num, pc in enumerate(episode_order, start=1):
        # 1) drive sustain_frames worth of alert=True at this class's
        #    feature signature so RecommendationEngine actually fires.
        feats = _features_for(pc)
        event = None
        for _ in range(recommender.engine.sustain_frames):
            event = recommender.update(
                alert=True, state=2, q_hci=0.5, features=feats, frame_idx=frame_idx,
            )
            frame_idx += 1
        assert event is not None, f"episode {ep_num}: recommendation did not fire"
        assert event.problem_class == pc, (
            f"episode {ep_num}: expected {pc}, classifier picked {event.problem_class} "
            "-- synthetic features need adjusting"
        )
        chosen_action = event.suggested_actions[0]

        # 2) decide the "true" outcome for this episode
        if rng.random() < noncompliance_rate:
            recovers = rng.random() < 0.5  # baseline, action-independent
        else:
            recovers = rng.random() < TRUE_RECOVERY_PROB[(pc, chosen_action)]
        q_hci_after = 0.5 + (0.10 if recovers else -0.02)

        # 3) advance frames (alert=False, i.e. things calmed down) until
        #    eval_frame, feeding q_hci_after so _resolve_pending scores it
        cooldown_needed = recommender.engine.cooldown_frames + 1
        steps = max(recommender.eval_window_frames, cooldown_needed) + 1
        for i in range(steps):
            q = q_hci_after if i > 0 else 0.5
            recommender.update(alert=False, state=None, q_hci=q,
                                features=None, frame_idx=frame_idx)
            frame_idx += 1

        if ep_num % checkpoint_every == 0:
            snapshot = {c.value: list(bandit.ranked(c)) for c in classes}
            checkpoints.append((ep_num, snapshot))

    return {
        "outcomes": outcomes,
        "checkpoints": checkpoints,
        "final_ranked": {c.value: bandit.ranked(c) for c in classes},
        "final_means": {
            f"{c.value}::{a}": bandit._arms[(c, a)].mean()
            for c in classes for a in ACTION_LIBRARY[c]
        },
        "true_probs": {f"{pc.value}::{a}": p for (pc, a), p in TRUE_RECOVERY_PROB.items()},
        "n_episodes": total_episodes,
        "noncompliance_rate": noncompliance_rate,
    }


def report_synthetic(result: dict) -> None:
    print(f"\n=== Synthetic run: {result['n_episodes']} episodes, "
          f"noncompliance_rate={result['noncompliance_rate']} ===\n")

    correct = 0
    for pc in ProblemClass:
        actions = ACTION_LIBRARY[pc]
        true_best = max(actions, key=lambda a: TRUE_RECOVERY_PROB[(pc, a)])
        learned_best = result["final_ranked"][pc.value][0]
        hit = learned_best == true_best
        correct += hit
        mean_best = result["final_means"][f"{pc.value}::{learned_best}"]
        print(f"{pc.value:24s} true_best={true_best[:30]:32s} "
              f"learned_best={learned_best[:30]:32s} "
              f"{'OK' if hit else 'MISS'}  (posterior mean={mean_best:.3f})")

    print(f"\n{correct}/{len(list(ProblemClass))} problem classes converged "
          f"to the true-best action.\n")

    print("Convergence over episodes (does the SPATIAL_RESTLESSNESS ranking "
          "stabilize as episodes accrue?):")
    pc0 = ProblemClass.SPATIAL_RESTLESSNESS.value
    for ep_num, snap in result["checkpoints"]:
        print(f"  after {ep_num:4d} episodes: {snap[pc0]}")


def run_real_log(log_path: str) -> None:
    """Parses a JSONL file of {problem_class, action, reward, frame_idx}
    rows -- exactly what on_outcome should be appending during a real
    PersonalizedRecommender session -- and reports arm statistics
    without any simulation."""
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("Log is empty -- no outcomes have resolved yet. Note eval_window_frames "
              "(default 150 frames, ~5s at 30fps) must elapse after a recommendation "
              "fires before any row appears here, so a short session may log zero rows "
              "even though recommendations fired.")
        return

    per_arm = defaultdict(lambda: {"pulls": 0, "wins": 0})
    for r in rows:
        key = (r["problem_class"], r["action"])
        per_arm[key]["pulls"] += 1
        per_arm[key]["wins"] += int(r["reward"])

    print(f"\n=== Real-log analysis: {len(rows)} resolved outcomes ===\n")
    for (pc, action), stats in sorted(per_arm.items()):
        rate = stats["wins"] / stats["pulls"]
        print(f"{pc:24s} {action[:32]:34s} pulls={stats['pulls']:4d}  "
              f"empirical_reward_rate={rate:.3f}")

    total_pulls = sum(s["pulls"] for s in per_arm.values())
    print(f"\nTotal resolved outcomes: {total_pulls}")
    if total_pulls < 30:
        print("WARNING: fewer than 30 resolved outcomes total across all arms -- "
              "posterior means are still dominated by the Beta(1,1) prior and any "
              "ranking should be treated as noise, not a learned preference yet.")

    # flag arms that have NEVER been pulled -- means choose() is starving them
    all_arms = {(pc.value, a) for pc, acts in ACTION_LIBRARY.items() for a in acts}
    seen_arms = set(per_arm.keys())
    unpulled = all_arms - seen_arms
    if unpulled:
        print(f"\nArms with zero resolved outcomes so far ({len(unpulled)}):")
        for pc, a in sorted(unpulled):
            print(f"  {pc:24s} {a}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_syn = sub.add_parser("synthetic", help="Validate learning mechanics against known ground truth")
    p_syn.add_argument("--episodes-per-class", type=int, default=60)
    p_syn.add_argument("--noncompliance", type=float, default=0.0,
                        help="Fraction of episodes where the simulated person ignores "
                             "suggested_actions[0] (0.0-1.0). Models reward noise from "
                             "unfollowed suggestions.")
    p_syn.add_argument("--seed", type=int, default=0)

    p_real = sub.add_parser("real", help="Analyze a real on_outcome log")
    p_real.add_argument("--log", required=True, help="Path to JSONL outcome log")

    args = parser.parse_args()

    if args.mode == "synthetic":
        result = run_synthetic(
            n_episodes_per_class=args.episodes_per_class,
            noncompliance_rate=args.noncompliance,
            seed=args.seed,
        )
        report_synthetic(result)
    else:
        run_real_log(args.log)


if __name__ == "__main__":
    sys.exit(main())
