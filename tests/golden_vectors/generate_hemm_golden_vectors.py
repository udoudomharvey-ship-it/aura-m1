"""
tests/golden_vectors/generate_hemm_golden_vectors.py
──────────────────────────────────────────────────────
Language: Python
Why: Cross-language parity for H_sp / H_te / H_in cannot be verified by
     comparing final QHCI values alone (the audit's own finding was that
     H_te/H_in were silently degenerate while QHCI still "looked" plausible
     downstream). This script generates deterministic golden test vectors —
     fixed input frames with pre-computed expected outputs — that BOTH the
     Python and Rust test suites load and check against independently.

Reproducibility
────────────────
Frames are generated with the same 32-bit LCG (Numerical Recipes constants:
multiplier 1664525, increment 1013904223) already used elsewhere in this
codebase for seeded synthetic data (SyntheticSource, BNN weight init). This
specific LCG was chosen — instead of Python's `random` module — because it
is trivial to reproduce bit-for-bit in Rust with plain u32 wrapping
arithmetic, which is what aura_core's test suite does. Do not swap this for
`random.Random` or `numpy` — that would break Rust reproducibility.

Regenerate with:
    python3 tests/golden_vectors/generate_hemm_golden_vectors.py

This OVERWRITES hemm_golden_vectors.json. Only do this if the HEMM
algorithm itself changes intentionally — if you're just adding a new
golden case, do it by hand or extend this script and note why in a
commit message, since existing cases must stay stable for regression
purposes.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aura_sdk.hemm.extractor import (  # noqa: E402
    _hsp_py, _hte_windowed_py, _hin_windowed_py, HEMM_WINDOW_CAP,
)

N_JOINTS = 25


def lcg_frame(seed: int) -> tuple[list[tuple[float, float, float]], int]:
    """Generate one 25-joint frame from an LCG state, returning (frame, next_seed)."""
    state = seed
    frame = []
    for _ in range(N_JOINTS):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        x = (state / 0xFFFFFFFF) * 0.4 - 0.2
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        y = (state / 0xFFFFFFFF) * 0.4 - 0.2
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        z = (state / 0xFFFFFFFF) * 0.4 - 0.2
        frame.append((x, y, z))
    return frame, state


def make_case(name: str, seed: int, n_frames: int, vel_max: float = 0.5) -> dict:
    frames = []
    s = seed
    for _ in range(n_frames):
        frame, s = lcg_frame(s)
        frames.append(frame)

    # IMPORTANT: HemmWindow / HEMMExtractor cap history at HEMM_WINDOW_CAP
    # frames, evicting the oldest ones as new frames arrive (a real sliding
    # window, not an unbounded history). To match that actual production
    # behaviour, expected values here must be computed over the same
    # last-CAP-frames slice a live HemmWindow would actually hold — not
    # over the full unbounded `frames` list. For n_frames <= CAP this is a
    # no-op; it only matters for cases with more history than CAP (see
    # "longer_history_seed_9001" below).
    windowed_frames = frames[-HEMM_WINDOW_CAP:]

    h_sp = _hsp_py(frames[-1])
    h_te = _hte_windowed_py(windowed_frames, vel_max)
    h_in = _hin_windowed_py(windowed_frames, vel_max)

    return {
        "name": name,
        "seed": seed,
        "n_frames": n_frames,
        "vel_max": vel_max,
        "frames": frames,
        "expected": {"h_sp": h_sp, "h_te": h_te, "h_in": h_in},
    }


def make_static_case(name: str, value: float, n_frames: int, vel_max: float = 0.5) -> dict:
    """A fully static skeleton — H_te / H_in should be exactly 0."""
    frame = [(value, value, value)] * N_JOINTS
    frames = [frame] * n_frames
    h_sp = _hsp_py(frames[-1])
    h_te = _hte_windowed_py(frames, vel_max)
    h_in = _hin_windowed_py(frames, vel_max)
    return {
        "name": name,
        "seed": None,
        "n_frames": n_frames,
        "vel_max": vel_max,
        "frames": frames,
        "expected": {"h_sp": h_sp, "h_te": h_te, "h_in": h_in},
    }


def main():
    cases = [
        make_case("full_window_seed_100", seed=100, n_frames=HEMM_WINDOW_CAP),
        make_case("full_window_seed_777", seed=777, n_frames=HEMM_WINDOW_CAP),
        make_case("partial_window_2frames_seed_42", seed=42, n_frames=2),
        make_case("partial_window_3frames_seed_42", seed=42, n_frames=3),
        make_case("partial_window_4frames_seed_42", seed=42, n_frames=4),
        make_static_case("static_skeleton_zero_entropy", value=0.3, n_frames=5),
        make_case("longer_history_seed_9001", seed=9001, n_frames=8),
    ]

    out_path = os.path.join(os.path.dirname(__file__), "hemm_golden_vectors.json")
    with open(out_path, "w") as f:
        json.dump({
            "description": (
                "HEMM cross-language golden vectors. Frames generated with a "
                "32-bit LCG (multiplier=1664525, increment=1013904223), "
                "reproducible bit-for-bit in Rust. Expected h_sp/h_te/h_in "
                "computed by the Python reference implementation "
                "(aura_sdk/hemm/extractor.py) after the windowed-entropy fix. "
                "Rust's aura_core::hemm::HemmWindow must match these within "
                "1e-4 to be considered at parity."
            ),
            "n_joints": N_JOINTS,
            "window_cap": HEMM_WINDOW_CAP,
            "cases": cases,
        }, f, indent=2)

    print(f"Wrote {len(cases)} golden vector cases to {out_path}")
    for c in cases:
        e = c["expected"]
        print(f"  {c['name']:38s} h_sp={e['h_sp']:.6f} h_te={e['h_te']:.6f} h_in={e['h_in']:.6f}")


if __name__ == "__main__":
    main()
