"""
Runs the real AuraPipeline (120-frame SyntheticSource, ThermalHarvester)
and logs every fired frame's real Q_HCI to q_hci_per_frame.csv.
"""

import csv

from aura_sdk.pipeline import AuraPipeline, PipelineConfig
from aura_sdk.sensors import SyntheticSource
from aura_sdk.harvesting import ThermalHarvester

PHASE_NAMES = ["Low Workload", "Moderate Workload", "High Workload"]


class Capture:
    def open(self):
        pass

    def close(self):
        pass

    def write(self, r):
        rows.append(r)


rows = []
capture = Capture()

pipeline = AuraPipeline(
    source=SyntheticSource(n_frames=120),
    harvester=ThermalHarvester(),
    sinks=[capture],
    config=PipelineConfig(verbose=False),
)

stats = pipeline.run()

print(f"total_frames={stats.total_frames}  fired_frames={stats.fired_frames}  idle_power_w={stats.idle_power_w}\n")

csv_rows = []
for r in rows:
    phase = PHASE_NAMES[min((r.frame_idx - 1) // 40, 2)]
    csv_rows.append({
        "frame": r.frame_idx + 1,
        "phase": phase,
        "fired": r.fired,
        "q_hci": r.q_hci,
    })

with open("q_hci_per_frame.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "phase", "fired", "q_hci"])
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"Wrote {len(csv_rows)} rows to q_hci_per_frame.csv\n")
with open("q_hci_per_frame.csv") as f:
    print(f.read())
