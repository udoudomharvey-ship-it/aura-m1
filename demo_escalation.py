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
