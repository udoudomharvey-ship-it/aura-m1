"""
aura_sdk_tests/test_hemm_golden_vectors.py
─────────────────────────────────────────────
Language: Python
Why: Cross-language parity regression protection for H_sp / H_te / H_in.

These tests load tests/golden_vectors/hemm_golden_vectors.json (fixed input
frames + pre-computed expected outputs) and re-verify the Python reference
implementation still produces those exact values. The companion Rust test
(aura_core/tests/hemm_golden_vectors.rs) loads the SAME JSON file and checks
aura_core::hemm::HemmWindow against it — if both pass, Python and Rust are
verified numerically at parity for H_sp/H_te/H_in, not just for final QHCI.

This is the regression test that would have caught the original bug: prior
to the windowed-entropy fix, h_te/h_in were silently ≈0.0 for every input
(single-sample one-hot histogram ⇒ zero entropy). If that ever comes back
— in Python OR Rust — these tests catch it immediately rather than relying
on an eyeball check of a downstream QHCI number.

Run: python -m unittest aura_sdk_tests.test_hemm_golden_vectors -v
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from aura_sdk.hemm.extractor import _hsp_py, _hte_windowed_py, _hin_windowed_py, HEMM_WINDOW_CAP


GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "golden_vectors", "hemm_golden_vectors.json"
)


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


class TestHEMMGoldenVectors(unittest.TestCase):

    TOLERANCE = 1e-4

    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()

    def test_golden_file_exists_and_has_cases(self):
        self.assertGreater(len(self.golden["cases"]), 0)

    def _run_case(self, case):
        frames = [[tuple(j) for j in frame] for frame in case["frames"]]
        vel_max = case["vel_max"]
        expected = case["expected"]

        # Mirror actual production behaviour: HemmWindow/HEMMExtractor cap
        # history at HEMM_WINDOW_CAP frames, evicting the oldest as new ones
        # arrive. For cases with more history than that (e.g.
        # longer_history_seed_9001), only the last CAP frames are what a
        # live window would actually hold — matches how the golden vector
        # generator computed "expected" in the first place.
        windowed_frames = frames[-HEMM_WINDOW_CAP:]

        h_sp = _hsp_py(frames[-1])
        h_te = _hte_windowed_py(windowed_frames, vel_max)
        h_in = _hin_windowed_py(windowed_frames, vel_max)

        self.assertAlmostEqual(h_sp, expected["h_sp"], delta=self.TOLERANCE,
            msg=f"[{case['name']}] H_sp mismatch")
        self.assertAlmostEqual(h_te, expected["h_te"], delta=self.TOLERANCE,
            msg=f"[{case['name']}] H_te mismatch")
        self.assertAlmostEqual(h_in, expected["h_in"], delta=self.TOLERANCE,
            msg=f"[{case['name']}] H_in mismatch")

    def test_all_golden_cases_reproduce_exactly(self):
        for case in self.golden["cases"]:
            with self.subTest(case=case["name"]):
                self._run_case(case)

    def test_full_window_case_is_not_degenerate(self):
        """Explicit regression guard: the audit's original bug made H_te and
        H_in silently ≈0.0 for every input. A full 5-frame window with
        varied motion must NOT reproduce that degenerate behaviour."""
        case = next(c for c in self.golden["cases"] if c["name"] == "full_window_seed_100")
        self.assertGreater(case["expected"]["h_te"], 0.05,
            "H_te golden value is suspiciously close to 0 — degenerate bug may have returned")
        self.assertGreater(case["expected"]["h_in"], 0.05,
            "H_in golden value is suspiciously close to 0 — degenerate bug may have returned")

    def test_static_skeleton_case_is_legitimately_zero(self):
        """Sanity check the other direction: a genuinely static skeleton
        SHOULD produce ≈0 H_te/H_in — confirms the fix tracks real motion
        rather than always returning a nonzero floor value."""
        case = next(c for c in self.golden["cases"] if c["name"] == "static_skeleton_zero_entropy")
        self.assertLess(abs(case["expected"]["h_te"]), 1e-6)
        self.assertLess(abs(case["expected"]["h_in"]), 1e-6)

    def test_partial_window_2frames_matches_known_degenerate_case(self):
        """2 buffered frames = exactly 1 velocity sample per (joint /
        joint-pair) = still a one-hot histogram = still ≈0 entropy. This is
        the ONE case where near-zero is still expected post-fix — it documents
        the honest boundary of the fix (needs >=3 frames / >=2 pairs for a
        real distribution), rather than a silent regression."""
        case = next(c for c in self.golden["cases"] if c["name"] == "partial_window_2frames_seed_42")
        self.assertLess(abs(case["expected"]["h_te"]), 1e-6)
        self.assertLess(abs(case["expected"]["h_in"]), 1e-6)

    def test_partial_window_3frames_is_meaningfully_nonzero(self):
        """3 buffered frames = 2 velocity samples = first point where the
        histogram is no longer forced one-hot. Confirms the fix takes effect
        as soon as mathematically possible, not only once the window is full."""
        case = next(c for c in self.golden["cases"] if c["name"] == "partial_window_3frames_seed_42")
        self.assertGreater(case["expected"]["h_te"], 0.0)
        self.assertGreater(case["expected"]["h_in"], 0.0)


if __name__ == "__main__":
    unittest.main()
