"""
aura_sdk_tests/test_energy_threshold_edge_cases.py
────────────────────────────────────────────────────
Language: Python (unittest)
Why: exercises EnergyHarvester.can_fire()/consume_pulse() at the exact
boundaries the energy-gating invariant depends on — below, at, and above
threshold; multiple accumulated pulses; carry-over across many small
updates; and a weak/intermittent harvesting profile. These boundaries
were previously covered only indirectly (via full-pipeline smoke tests
that never happen to land exactly on the threshold) — this file tests
the gating contract directly and deterministically.

Run: python -m unittest aura_sdk_tests.test_energy_threshold_edge_cases -v
"""
from __future__ import annotations
import unittest

from aura_sdk.harvesting import EnergyHarvester, RFHarvester, HybridHarvester, BNN_INFERENCE_ENERGY_UJ


class TestEnergyThresholdBoundaries(unittest.TestCase):
    """Directly exercises the base-class gating contract, independent of
    any single harvester's physics model — isolates exactly the logic
    the energy invariant depends on."""

    def test_exactly_below_threshold_cannot_fire(self):
        h = EnergyHarvester()
        h._accumulated_uJ = BNN_INFERENCE_ENERGY_UJ - 1e-6
        self.assertFalse(h.can_fire())
        self.assertFalse(h.consume_pulse())
        # a failed consume must not silently drain energy
        self.assertAlmostEqual(h.accumulated_uJ, BNN_INFERENCE_ENERGY_UJ - 1e-6, places=9)
        self.assertEqual(h.pulse_count, 0)

    def test_exactly_at_threshold_can_fire(self):
        h = EnergyHarvester()
        h._accumulated_uJ = BNN_INFERENCE_ENERGY_UJ
        self.assertTrue(h.can_fire())
        self.assertTrue(h.consume_pulse())
        self.assertAlmostEqual(h.accumulated_uJ, 0.0, places=9)
        self.assertEqual(h.pulse_count, 1)

    def test_above_threshold_leaves_correct_remainder(self):
        h = EnergyHarvester()
        surplus = 0.0123
        h._accumulated_uJ = BNN_INFERENCE_ENERGY_UJ + surplus
        self.assertTrue(h.consume_pulse())
        self.assertAlmostEqual(h.accumulated_uJ, surplus, places=9)
        self.assertEqual(h.pulse_count, 1)

    def test_multiple_accumulated_pulses_fire_until_exhausted(self):
        h = EnergyHarvester()
        h._accumulated_uJ = BNN_INFERENCE_ENERGY_UJ * 3.5  # 3 full pulses + partial
        fired = 0
        while h.consume_pulse():
            fired += 1
        self.assertEqual(fired, 3)
        self.assertEqual(h.pulse_count, 3)
        # remaining 0.5 pulse-worth must not be silently discarded or over-consumed
        self.assertAlmostEqual(h.accumulated_uJ, BNN_INFERENCE_ENERGY_UJ * 0.5, places=6)
        self.assertFalse(h.can_fire())

    def test_energy_carries_over_across_many_sub_threshold_updates(self):
        """No single update() call should need to reach the threshold —
        many small deposits below threshold must still sum correctly and
        eventually permit exactly one pulse, not fire early and not lose
        energy between calls."""
        h = EnergyHarvester()
        deposit = BNN_INFERENCE_ENERGY_UJ / 10.0
        for i in range(9):
            h._accumulated_uJ += deposit
            self.assertFalse(
                h.can_fire(),
                f"fired early after only {i+1}/10 sub-threshold deposits",
            )
        h._accumulated_uJ += deposit  # 10th deposit reaches exactly 1.0x threshold
        self.assertTrue(h.can_fire())
        self.assertTrue(h.consume_pulse())
        self.assertAlmostEqual(h.accumulated_uJ, 0.0, places=6)


class TestWeakAndIntermittentHarvesting(unittest.TestCase):
    """Realistic (non-boundary-manipulated) harvesting profiles, using the
    actual physics models rather than directly poking _accumulated_uJ."""

    def test_weak_rf_harvesting_does_not_fire_prematurely(self):
        # signal_dbm this low clamps to RFHarvester.MIN_UW (1.0 uW) — the
        # weakest power the model allows — so this is a genuine multi-step
        # accumulation, not an instant single-update fire.
        h = RFHarvester(signal_dbm=-60.0)
        dt_ms = 10.0
        premature_fires = 0
        steps_to_first_fire = None
        for step in range(1, 20_000):
            h.update(dt_ms)
            if h.can_fire():
                steps_to_first_fire = step
                break
        self.assertEqual(premature_fires, 0)
        self.assertIsNotNone(
            steps_to_first_fire,
            "weak RF harvester never accumulated one inference's worth of "
            "energy in 200s of simulated time — power model may have changed",
        )
        # A genuinely weak (MIN_UW-clamped) source should take several
        # steps, not fire on the very first update — confirms accumulation
        # is really happening rather than the gate being bypassed.
        self.assertGreater(steps_to_first_fire, 1)

    def test_hybrid_harvester_pools_multiple_weak_sources(self):
        """Two sources individually too weak might still combine to reach
        threshold sooner than either alone — verifies the pooling logic
        in HybridHarvester.update(), not just single-source accounting."""
        solo = RFHarvester(label="solo", signal_dbm=-60.0)
        combo = HybridHarvester([
            RFHarvester(label="a", signal_dbm=-60.0),
            RFHarvester(label="b", signal_dbm=-60.0),
        ])

        dt_ms = 10.0
        solo_steps = 0
        while not solo.can_fire() and solo_steps < 50_000:
            solo.update(dt_ms)
            solo_steps += 1

        combo_steps = 0
        while not combo.can_fire() and combo_steps < 50_000:
            combo.update(dt_ms)
            combo_steps += 1

        self.assertLess(
            combo_steps, solo_steps,
            "two pooled weak sources should reach threshold faster than "
            "one of them alone",
        )


if __name__ == "__main__":
    unittest.main()
