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
