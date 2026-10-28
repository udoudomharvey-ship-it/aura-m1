"""
scripts/benchmark_hot_path.py
────────────────────────────────
Language: Python
Why: master-prompt Section 16 demands measured, not assumed, performance
against the 30 fps (33.3 ms budget) / <=3.8 ms hot-path targets. This
benchmarks each stage of the real pipeline (not a reimplementation)
individually and end-to-end, and reports mean/median/p95/p99/min/max/stdev.

This measures the PYTHON fallback backend only — the one actually running
in every environment this session had access to (no Rust toolchain
available to compile aura_core_rs here). The thesis's 3.8 ms target was
set with the compiled Rust/firmware backend in mind; the Python numbers
below are the honest baseline for "backend: python" deployments, and the
gap they show is itself the quantified case for finishing the PyO3 build.

Run: python3 scripts/benchmark_hot_path.py [--frames 2000]
"""
from __future__ import annotations
import argparse
import statistics as stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura_sdk.sensors import SyntheticSource
from aura_sdk.hemm import HEMMExtractor
from aura_sdk.inference.bnn import AuraBNN
from aura_sdk.ncl.circuits import QualityStateEncoder, ThresholdDetector


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(name, samples_ms):
    s = sorted(samples_ms)
    return {
        "stage": name,
        "mean_ms": stats.mean(s),
        "median_ms": stats.median(s),
        "p95_ms": percentile(s, 0.95),
        "p99_ms": percentile(s, 0.99),
        "min_ms": min(s),
        "max_ms": max(s),
        "stdev_ms": stats.pstdev(s) if len(s) > 1 else 0.0,
        "n": len(s),
    }


def print_row(r):
    print(f"  {r['stage']:<18} mean={r['mean_ms']:.4f}ms  median={r['median_ms']:.4f}ms  "
          f"p95={r['p95_ms']:.4f}ms  p99={r['p99_ms']:.4f}ms  "
          f"min={r['min_ms']:.4f}ms  max={r['max_ms']:.4f}ms  stdev={r['stdev_ms']:.4f}ms  n={r['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2000)
    args = ap.parse_args()

    src = SyntheticSource(n_frames=args.frames + 1, seed=99)
    with src:
        frames = [f.joints for f in src.frames()]

    extractor = HEMMExtractor()
    bnn = AuraBNN()
    encoder = QualityStateEncoder()
    detector = ThresholdDetector(0.42)

    t_hemm, t_bnn, t_ncl, t_total = [], [], [], []

    prev = frames[0]
    for curr in frames[1:]:
        t0 = time.perf_counter()
        feat = extractor.compute(prev, curr)
        t1 = time.perf_counter()
        result = bnn.infer(feat.h_sp, feat.h_te, feat.h_in)
        t2 = time.perf_counter()
        encoder.encode(result.q_hci)
        detector.update(result.q_hci)
        t3 = time.perf_counter()

        t_hemm.append((t1 - t0) * 1000)
        t_bnn.append((t2 - t1) * 1000)
        t_ncl.append((t3 - t2) * 1000)
        t_total.append((t3 - t0) * 1000)
        prev = curr

    rows = [
        summarize("HEMM (Hsp+Hte+Hin)", t_hemm),
        summarize("BNN inference", t_bnn),
        summarize("NCL encoding", t_ncl),
        summarize("Total hot path", t_total),
    ]

    print(f"\n  Aura-M1 hot-path benchmark — backend: python — {args.frames} frames")
    print(f"  Target: 30 fps (33.3ms budget/frame), hot-path <= 3.8ms\n")
    for r in rows:
        print_row(r)

    total = rows[-1]
    frame_budget_ms = 1000.0 / 30.0
    print(f"\n  Frame budget (30 fps)   : {frame_budget_ms:.3f} ms")
    print(f"  Hot-path target         : 3.800 ms")
    print(f"  Measured mean hot path  : {total['mean_ms']:.4f} ms  "
          f"({'PASS' if total['mean_ms'] <= 3.8 else 'FAIL'} vs 3.8ms target, "
          f"{'PASS' if total['mean_ms'] <= frame_budget_ms else 'FAIL'} vs 30fps budget)")
    print(f"  Measured p99 hot path   : {total['p99_ms']:.4f} ms  "
          f"({'PASS' if total['p99_ms'] <= 3.8 else 'FAIL'} vs 3.8ms target)")
    print(f"\n  NOTE: this is the Python fallback backend (no compiled Rust\n"
          f"  extension available in this environment) — not the backend the\n"
          f"  thesis's 3.8ms target was set against. Re-run against 'backend:\n"
          f"  rust' after compiling aura_core_rs (see Dockerfile.rust-backend)\n"
          f"  for the number that actually validates or refutes that target.\n")

    return rows


if __name__ == "__main__":
    main()
