"""
aura_sdk_tests/test_bnn_golden_vectors.py
────────────────────────────────────────────
Language: Python
Why: regression protection for the BNN forward pass against the canonical
model. Companion to aura_core/tests/bnn_golden_vectors.rs, which loads the
same JSON and checks Rust's BnnWeights::canonical() + AuraBnn against it.

Run: python -m unittest aura_sdk_tests.test_bnn_golden_vectors -v
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

from aura_sdk.inference.bnn import _bnn_forward, _MODEL_META

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "golden_vectors", "bnn_golden_vectors.json"
)


class TestBNNGoldenVectors(unittest.TestCase):

    Q_TOLERANCE = 1e-4
    LOGIT_TOLERANCE = 1e-3

    @classmethod
    def setUpClass(cls):
        with open(GOLDEN_PATH) as f:
            cls.golden = json.load(f)

    def test_golden_file_matches_current_canonical_model(self):
        """If the canonical model gets regenerated, the golden vectors must
        be regenerated too — this test fails loudly rather than silently
        comparing against stale expected values from a different model."""
        self.assertEqual(
            self.golden["model_checksum"], _MODEL_META["checksum"],
            "bnn_golden_vectors.json was generated against a different "
            "canonical model than the one currently loaded — re-run "
            "tests/golden_vectors/generate_bnn_golden_vectors.py"
        )

    def test_all_golden_cases_reproduce_exactly(self):
        for case in self.golden["cases"]:
            with self.subTest(case=case["name"]):
                inp = case["input"]
                expected = case["expected"]
                state, q_hci, logits = _bnn_forward(inp["h_sp"], inp["h_te"], inp["h_in"])

                self.assertEqual(state, expected["state"],
                    f"[{case['name']}] state mismatch")
                self.assertAlmostEqual(q_hci, expected["q_hci"], delta=self.Q_TOLERANCE,
                    msg=f"[{case['name']}] q_hci mismatch")
                for i, (actual_l, expected_l) in enumerate(zip(logits, expected["logits"])):
                    self.assertAlmostEqual(actual_l, expected_l, delta=self.LOGIT_TOLERANCE,
                        msg=f"[{case['name']}] logit[{i}] mismatch")
                alert = q_hci < self.golden["alert_threshold"]
                self.assertEqual(alert, expected["alert"],
                    f"[{case['name']}] alert flag mismatch")

    def test_quantisation_boundary_convention_is_consistent(self):
        """h_sp/h_te/h_in exactly at 0.5 must quantise the same way as
        values clearly above 0.5 (the `>= 0.5` convention) — pins down
        floating-point tie-breaking behaviour explicitly rather than
        leaving it to coincidence."""
        boundary = next(c for c in self.golden["cases"] if c["name"] == "boundary_all_exactly_0.5")
        hi_cell = next(c for c in self.golden["cases"] if c["name"] == "quantised_cell_7_hihihi")
        self.assertEqual(boundary["expected"]["state"], hi_cell["expected"]["state"])
        self.assertAlmostEqual(boundary["expected"]["q_hci"], hi_cell["expected"]["q_hci"], delta=1e-6)

    def test_reachable_states_are_documented_not_silently_missing(self):
        """Which states are reachable depends on the canonical model's
        weights and is expected to change when retrained. Untrained
        placeholder: {1,3,4,5} reachable. Retrained on real labeled data
        (see retrain_bnn.py's LABEL_TO_STATE): only {1,5} reachable, since
        the current labeling scheme only distinguishes focused (state 5)
        from three problem labels (all mapped to state 1) -- no data
        pushes the model toward the intermediate states 2-4. Expected,
        not a regression."""
        states = set(c["expected"]["state"] for c in self.golden["cases"])
        self.assertEqual(states, {1, 5},
            "state coverage changed - update the expected set above to match")


if __name__ == "__main__":
    unittest.main()
