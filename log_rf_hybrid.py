"""
Runs the real AuraPipeline for RF-only and Hybrid (RF+Thermal) configurations,
120 frames each, logging per-frame Q_HCI to q_hci_rf_hybrid.csv.
"""

import csv

from aura_sdk.pipeline import AuraPipeline, PipelineConfig
from aura_sdk.sensors import SyntheticSource
from aura_sdk.harvesting import RFHarvester, ThermalHarvester, HybridHarvester

PHASE_NAMES = ["Low Workload", "Moderate Workload", "High Workload"]


class Capture:
    def open(self):
        pass

    def close(self):
        pass

    def write(self, r):
        rows.append(r)


configs = {
    "RF Harvester only": lambda: RFHarvester(),
    "Hybrid (RF + Thermal)": lambda: HybridHarvester(sources=[RFHarvester(), ThermalHarvester()]),
}

all_rows = []

for config_name, make_harvester in configs.items():
    rows = []
    capture = Capture()
    pipeline = AuraPipeline(
        source=SyntheticSource(n_frames=120),
        harvester=make_harvester(),
        sinks=[capture],
        config=PipelineConfig(verbose=False),
    )
    stats = pipeline.run()
    print(f"{config_name}: total_frames={stats.total_frames}  fired_frames={stats.fired_frames}  idle_power_w={stats.idle_power_w}")

    for r in rows:
        phase = PHASE_NAMES[min((r.frame_idx - 1) // 40, 2)]
        all_rows.append({
            "config": config_name,
            "frame": r.frame_idx,
            "phase": phase,
            "fired": r.fired,
            "q_hci": r.q_hci,
        })

print()

with open("q_hci_rf_hybrid.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["config", "frame", "phase", "fired", "q_hci"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Wrote {len(all_rows)} rows to q_hci_rf_hybrid.csv\n")
with open("q_hci_rf_hybrid.csv") as f:
    print(f.read())
