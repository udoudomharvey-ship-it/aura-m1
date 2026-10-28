"""
Run the full Aura-M1 pipeline with the real, saved escalation config —
the one produced by frontend/aura-m1-escalation-setup.html — wired in.

This is the piece that connects everything built so far:
  detect (ThresholdDetector) -> measure (HEMM entropy + BNN Q_HCI)
  -> recommend (rule-based table) -> explain (plain language)
  -> personalize (bandit) -> escalate (your real contacts, from the UI)

Looks for escalation_config.json in the current directory first, then
falls back to ~/Downloads (where a browser save usually lands) if it's
not been moved into the project yet.
"""
import os
import json
from pathlib import Path

from aura_sdk.pipeline import AuraPipeline, PipelineConfig
from aura_sdk.sensors import SyntheticSource
from aura_sdk_tests.scenario_synthetic_source import ScenarioSyntheticSource
from aura_sdk.harvesting import RFHarvester, ThermalHarvester, HybridHarvester
from aura_sdk.recommend import (
    RecommendationEngine, EscalationConfig, EscalatingRecommender, explain,
    send_escalation, ProblemClass,
)


def find_escalation_config() -> Path:
    candidates = [
        Path("escalation_config.json"),
        Path.home() / "Downloads" / "escalation_config.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Couldn't find escalation_config.json in the project folder or "
        "~/Downloads. Save it again from the setup screen, or move the "
        "downloaded file into this project folder."
    )


def log_outcome(problem_class, action, reward, frame_idx):
    with open("outcomes.jsonl", "a") as f:
        f.write(json.dumps({
            "problem_class": problem_class.value,
            "action": action,
            "reward": reward,
            "frame_idx": frame_idx,
        }) + "\n")


class LoggingEscalatingRecommender(EscalatingRecommender):
    def _on_outcome(self, problem_class, action, reward, frame_idx):
        super()._on_outcome(problem_class, action, reward, frame_idx)
        log_outcome(problem_class, action, reward, frame_idx)


config_path = find_escalation_config()
print(f"Loading escalation config from: {config_path}\n")

escalation_config = EscalationConfig.load(str(config_path))
escalation_config.validate()

print("Escalation chain:")
if not escalation_config.contacts:
    print("  (none configured — escalation is off)")
for i, c in enumerate(escalation_config.contacts):
    print(f"  tier {i}: {c.name} ({c.method}) — notified at {c.threshold} unresolved outcomes")
send_mode = "REAL SENDING ENABLED" if os.environ.get("AURA_SEND_REAL_ESCALATIONS") == "1" else "dry run (no real emails will be sent)"
print(f"  send mode: {send_mode}")
print()

recommender = LoggingEscalatingRecommender(
    base_engine=RecommendationEngine(sustain_frames=5, cooldown_frames=30),
    escalation_config=escalation_config,
    eval_window_frames=60,
    recovery_margin=0.05,
)

pipeline = AuraPipeline(
    source=ScenarioSyntheticSource(
        phases=[
            (ProblemClass.SPATIAL_RESTLESSNESS, 400),
            (ProblemClass.COGNITIVE_OVERLOAD, 400),
            (ProblemClass.SUSTAINED_LOW_QUALITY, 400),
        ],
        seed=42,
    ),
    harvester=HybridHarvester(sources=[RFHarvester(), ThermalHarvester()]),
    sinks=[],
    config=PipelineConfig(verbose=False),
    recommender=recommender,
)

pipeline.open()
notifications, escalations = [], []
for frame in pipeline.source.frames():
    pipeline.process_frame(frame)
    if pipeline.last_recommendation:
        notifications.append(explain(pipeline.last_recommendation,
                                      frame_dt_ms=pipeline.config.frame_dt_ms))
    if recommender.last_escalation:
        escalations.append(recommender.last_escalation)
        send_escalation(recommender.last_escalation)
pipeline.close()

stats = pipeline.session_stats()
print(stats.report())
print()
print(f"{len(notifications)} recommendation(s) fired.")
for n in notifications:
    print(f"  [{n.severity.value.upper():8s}] {n.title} — {n.suggested_actions[0]}")

print()
print(f"{len(escalations)} escalation(s) fired.")
for e in escalations:
    print(f"  tier {e.tier} -> {e.contact_name} ({e.contact_method}): \"{e.message}\"")
