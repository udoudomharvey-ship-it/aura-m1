"""
tests/golden_vectors/generate_bnn_golden_vectors.py
──────────────────────────────────────────────────────
Language: Python
Why: cross-language parity regression protection for the BNN forward pass,
using the canonical model (models/aura_m1_bnn_v1.json) as the fixed weight
source — Python, Rust, and firmware must all produce the same state/QHCI/
logits for the same input, now that all three load the same weights.

Coverage note: each of the 3 HEMM inputs is quantised to +1/-1 before
reaching the hidden layer (`_quantise(x) = 1 if x >= 0.5 else -1`), so
there are only 2^3 = 8 possible distinct hidden-layer input patterns for
ANY continuous (h_sp, h_te, h_in) input — the hidden layer literally
cannot distinguish h_sp=0.51 from h_sp=0.99, only whether it's >=0.5.
This script tests all 8 combinations exhaustively rather than sampling
toward "5 states" — with the current untrained placeholder weights, only
4 of the 5 quality states are actually reachable (state 2 does not occur
for any of the 8 inputs). That is expected behaviour for random,
untrained weights on a binarised 3-input space, not a bug — see
models/aura_m1_bnn_v1.json's "note" field. If/when the canonical model is
replaced with real trained weights, re-run this script; it may reach all
5 states, or a different subset, depending on what training produces.

Also includes boundary-condition cases: inputs exactly at the 0.5
quantisation threshold (where floating-point tie-breaking matters) and
alert-threshold cases (Q_HCI relative to the default 0.42 alert cutoff).

Regenerate with:
    python3 tests/golden_vectors/generate_bnn_golden_vectors.py
"""
import json
import os
import sys
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aura_sdk.inference.bnn import _bnn_forward, _MODEL_META  # noqa: E402

ALERT_THRESHOLD = 0.42


def make_case(name: str, h_sp: float, h_te: float, h_in: float) -> dict:
    state, q_hci, logits = _bnn_forward(h_sp, h_te, h_in)
    return {
        "name": name,
        "input": {"h_sp": h_sp, "h_te": h_te, "h_in": h_in},
        "expected": {
            "state": state,
            "q_hci": q_hci,
            "logits": logits,
            "alert": q_hci < ALERT_THRESHOLD,
        },
    }


def main():
    cases = []

    # Exhaustive coverage: all 8 possible quantised input combinations.
    # 0.3 / 0.7 chosen as clearly-below / clearly-above the 0.5 threshold
    # (not boundary cases themselves — those are separate, below).
    for i, (hsp_side, hte_side, hin_side) in enumerate(itertools.product([0.3, 0.7], repeat=3)):
        label = "".join("hi" if v == 0.7 else "lo" for v in (hsp_side, hte_side, hin_side))
        cases.append(make_case(f"quantised_cell_{i}_{label}", hsp_side, hte_side, hin_side))

    # Boundary conditions: exactly at the 0.5 quantisation threshold.
    # _quantise uses `x >= 0.5 -> +1`, so 0.5 itself quantises to +1 same
    # as 0.7 would — these cases pin down that convention explicitly.
    cases.append(make_case("boundary_all_exactly_0.5", 0.5, 0.5, 0.5))
    cases.append(make_case("boundary_h_sp_exactly_0.5", 0.5, 0.3, 0.3))
    cases.append(make_case("boundary_just_below_0.5", 0.499999, 0.3, 0.3))

    # Extreme corners of the continuous input range (still just 2 of the
    # 8 quantised cells, but useful as sanity/regression anchors since
    # they're unambiguous, not near any floating-point boundary).
    cases.append(make_case("extreme_all_zero", 0.0, 0.0, 0.0))
    cases.append(make_case("extreme_all_one", 1.0, 1.0, 1.0))

    out = {
        "description": (
            "BNN cross-language golden vectors using the canonical model "
            "(models/aura_m1_bnn_v1.json). Coverage is exhaustive over all "
            "8 possible quantised input combinations plus boundary and "
            "extreme cases -- see this script's module docstring for why "
            "only 4 of 5 states are reachable with the current placeholder "
            "weights."
        ),
        "model_checksum": _MODEL_META["checksum"],
        "model_version": _MODEL_META["model_version"],
        "alert_threshold": ALERT_THRESHOLD,
        "cases": cases,
    }

    out_path = os.path.join(os.path.dirname(__file__), "bnn_golden_vectors.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {len(cases)} BNN golden vector cases to {out_path}")
    states_seen = sorted(set(c["expected"]["state"] for c in cases))
    print(f"States reached across all cases: {states_seen}")
    for c in cases:
        e = c["expected"]
        print(f"  {c['name']:32s} state={e['state']} q_hci={e['q_hci']:.4f} alert={e['alert']}")


if __name__ == "__main__":
    main()
