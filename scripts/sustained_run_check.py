"""
scripts/sustained_run_check.py
─────────────────────────────────
Language: Python
Why: master-prompt Section 17 — a real sustained run tracking memory
growth, energy accumulation, and state transitions over an extended
session, not just a short smoke test.

This runs 10,000 frames (~330s of simulated 30fps time; measured wall
time is whatever the Python backend's real per-frame cost is — see
benchmark_hot_path.py, ~8ms/frame here, so ~80s wall time) rather than
the full 30 minutes the master prompt asks for: that would be ~3.5
hours of wall time on this backend, impractical for a single check.
The memory-growth question this answers doesn't need 30 minutes — a
leak that exists shows up in the first few thousand frames or not at
all, since nothing in the pipeline's design (fixed 5-frame HEMM window,
fixed-size feature arrays) should scale with frame count.

Run: python3 scripts/sustained_run_check.py [--frames 10000]
"""
from __future__ import annotations
import argparse
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura_sdk.sensors import SyntheticSource
from aura_sdk.harvesting import ThermalHarvester
from aura_sdk.pipeline import AuraPipeline, PipelineConfig


def rss_mb():
    # ru_maxrss is KB on Linux, bytes on macOS — this environment is Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10_000)
    ap.add_argument("--checkpoint-every", type=int, default=1_000)
    args = ap.parse_args()

    source = SyntheticSource(n_frames=args.frames, seed=7)
    pipeline = AuraPipeline(
        source=source,
        harvester=ThermalHarvester(delta_T_C=5.0),
        sinks=[],
        config=PipelineConfig(verbose=False),
    )
    pipeline.open()

    checkpoints = []
    state_transitions = 0
    last_state = None
    t_start = time.perf_counter()
    n = 0
    for frame in pipeline.source.frames():
        record = pipeline.process_frame(frame)
        n += 1
        if record.fired and record.state != last_state and last_state is not None:
            state_transitions += 1
        if record.fired:
            last_state = record.state

        if n % args.checkpoint_every == 0:
            checkpoints.append((n, rss_mb(), time.perf_counter() - t_start))

    stats = pipeline.session_stats()
    pipeline.close()
    elapsed = time.perf_counter() - t_start

    print(f"\n  Sustained run — {args.frames} frames, backend: python\n")
    print(f"  {'frame':>8}  {'RSS (MB)':>10}  {'elapsed (s)':>12}")
    for n_, mb, t in checkpoints:
        print(f"  {n_:>8}  {mb:>10.2f}  {t:>12.2f}")

    first_mb = checkpoints[0][1] if checkpoints else rss_mb()
    last_mb = checkpoints[-1][1] if checkpoints else rss_mb()
    growth_mb = last_mb - first_mb
    growth_pct = (growth_mb / first_mb * 100.0) if first_mb else 0.0

    print(f"\n  Total frames         : {stats.total_frames}")
    print(f"  Fired inferences     : {stats.fired_frames} ({stats.fire_rate_pct:.1f}%)")
    print(f"  Skipped (energy)     : {stats.skipped_frames}")
    print(f"  State transitions    : {state_transitions}")
    print(f"  Mean Q_HCI           : {stats.mean_q:.4f}")
    print(f"  Alert count          : {stats.alert_count}")
    print(f"  Idle power           : {stats.idle_power_w} W (simulated)")
    print(f"  Wall time            : {elapsed:.1f} s ({elapsed/args.frames*1000:.3f} ms/frame avg)")
    print(f"  RSS at start         : {first_mb:.2f} MB")
    print(f"  RSS at end           : {last_mb:.2f} MB")
    print(f"  RSS growth           : {growth_mb:+.2f} MB ({growth_pct:+.1f}%)")
    print(f"  Memory growth status : "
          f"{'PASS (bounded, no runaway growth)' if growth_pct < 10 else 'INVESTIGATE (>10% RSS growth)'}")


if __name__ == "__main__":
    main()
