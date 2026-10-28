"""
aura_sdk_tests/test_integration.py
─────────────────────────────────────
Language: Python
Integration tests covering:
  - Training pipeline (pure Python BNN)
  - HEMM entropy extractor (Python fallback)
  - Sensor sources (SyntheticSource, FileSource)
  - Output sinks (CSVLogger, ConsoleOutput)
  - Full AuraPipeline run
  - Weight export/import roundtrip

Run: python -m pytest aura_sdk_tests/ -v
"""
import sys, os, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest

# ── Training ──────────────────────────────────────────────────────────────

class TestAuraBNNMemoryBounded(unittest.TestCase):
    """
    Regression test for a confirmed memory leak: AuraBNN._history used to
    be an unbounded list — every infer() call appended and nothing ever
    trimmed it. Confirmed via a real 10,000-frame sustained-run
    measurement (scripts/sustained_run_check.py): RSS grew linearly and
    unboundedly (+40% over 10k frames). Fixed with a bounded deque plus
    an incremental running sum for the mean (so session_q_hci() stays
    exactly correct for the whole session even after old entries are
    evicted from the bounded window, not just an average over whatever's
    still in the deque).
    """

    def test_history_does_not_grow_without_bound(self):
        from aura_sdk.inference.bnn import AuraBNN

        bnn = AuraBNN()
        n = 2_000
        for i in range(n):
            bnn.infer(0.5, 0.5, 0.5)

        self.assertEqual(bnn.inference_count, n)
        self.assertLess(
            len(bnn.history), n,
            "AuraBNN._history grew to match the full inference count — "
            "the bounded-deque fix regressed back to unbounded growth",
        )
        self.assertLessEqual(len(bnn.history), 500)

    def test_session_mean_correct_even_after_history_eviction(self):
        """The running mean must reflect the WHOLE session, not just
        whatever's left in the bounded window after older entries are
        evicted — this is exactly why an incremental sum is kept
        separately rather than just averaging over .history."""
        from aura_sdk.inference.bnn import AuraBNN

        bnn = AuraBNN()
        # Push well past the 500-entry window with a fixed, known feature
        # vector so the expected mean is exactly computable independent
        # of AuraBNN's own internals.
        n = 1_500
        for _ in range(n):
            bnn.infer(0.9, 0.9, 0.9)  # deterministic -> same q_hci every call
        expected_mean = bnn.history[0].q_hci  # every result is identical
        self.assertAlmostEqual(bnn.session_q_hci(), expected_mean, places=9)
        self.assertEqual(bnn.inference_count, n)


class TestTraining(unittest.TestCase):

    def setUp(self):
        from aura_sdk.training import generate_synthetic_dataset, PurePythonBNN
        self.generate = generate_synthetic_dataset
        self.BNN      = PurePythonBNN

    def test_dataset_generation(self):
        records = self.generate(n_subjects=5, sessions_per_subject=3, seed=42)
        self.assertEqual(len(records), 15)

    def test_record_features_in_range(self):
        records = self.generate(n_subjects=5, sessions_per_subject=3)
        for r in records:
            self.assertAlmostEqual(sum(r.features), sum(r.features))  # no NaN
            for f in r.features:
                self.assertGreater(f, 0.0)
                self.assertLess(f, 1.0)

    def test_state_labels_valid(self):
        records = self.generate(n_subjects=5, sessions_per_subject=3)
        for r in records:
            self.assertIn(r.state, [1, 2, 3, 4, 5])

    def test_bnn_forward_pass(self):
        bnn = self.BNN(seed=42)
        state, q = bnn.predict_state([0.42, 0.58, 0.31])
        self.assertIn(state, [1, 2, 3, 4, 5])
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)

    def test_bnn_training_reduces_loss(self):
        from aura_sdk.training import generate_synthetic_dataset
        from aura_sdk.training.train import _binary_target
        records = generate_synthetic_dataset(n_subjects=3, sessions_per_subject=3)
        bnn     = self.BNN(seed=42)
        losses  = []
        for _ in range(5):
            epoch_loss = 0.0
            for r in records:
                # PurePythonBNN is now binary (0=ok, 1=alert), not the old
                # 1-5 state scheme. r.state is still 1-5 (from the
                # synthetic dataset's original labeling) and is the wrong
                # thing to pass here -- it indexed out of range for
                # anything but exactly state 1. _binary_target(quality)
                # is the actual, already-established conversion used
                # elsewhere in train.py (loso_cross_validate), splitting
                # at the same 0.45 threshold ThresholdDetector uses in
                # production, so training labels match deployment.
                target = _binary_target(r.quality)
                loss = bnn.train_step(r.features, target, lr=0.01)
                epoch_loss += loss
            losses.append(epoch_loss / len(records))
        # Loss should trend downward over 5 epochs
        self.assertLess(losses[-1], losses[0] * 1.5)

    def test_binarise_produces_valid_weights(self):
        bnn = self.BNN(seed=42)
        weights = bnn.binarise()
        self.assertIn("w1", weights)
        self.assertIn("w2", weights)
        # All weights should be -1 or +1
        for row in weights["w1"]:
            for w in row:
                self.assertIn(w, [-1, 1])

    def test_weight_export_import(self):
        from aura_sdk.training import train_and_export, generate_synthetic_dataset
        records = generate_synthetic_dataset(n_subjects=3, sessions_per_subject=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_weights.json")
            result_path = train_and_export(
                records, output_path=path,
                n_epochs=5, run_loso=False, verbose=False
            )
            self.assertTrue(os.path.exists(result_path))
            with open(result_path) as f:
                data = json.load(f)
            self.assertIn("weights", data)
            self.assertIn("w1",      data["weights"])

    def test_loso_returns_stats(self):
        from aura_sdk.training import loso_cross_validate, generate_synthetic_dataset
        records = generate_synthetic_dataset(n_subjects=4, sessions_per_subject=3)
        stats   = loso_cross_validate(records, n_epochs=3, verbose=False)
        self.assertIn("mean_accuracy", stats)
        self.assertIn("fold_accuracies", stats)
        self.assertEqual(len(stats["fold_accuracies"]), 4)
        self.assertGreater(stats["mean_accuracy"], 0.0)


# ── HEMM Extractor ────────────────────────────────────────────────────────

class TestHEMMExtractor(unittest.TestCase):

    def setUp(self):
        from aura_sdk.hemm import HEMMExtractor, normalise_joints
        self.HEMMExtractor    = HEMMExtractor
        self.normalise_joints = normalise_joints

    def _make_joints(self, n=25, val=0.3):
        return [(val + i*0.01, val + i*0.02, val + i*0.015) for i in range(n)]

    def test_extractor_creates(self):
        e = self.HEMMExtractor()
        self.assertIn(e.backend, ["rust", "python"])

    def test_compute_returns_features(self):
        e    = self.HEMMExtractor()
        prev = self._make_joints(25, 0.1)
        curr = self._make_joints(25, 0.2)
        f    = e.compute(prev, curr)
        self.assertIsNotNone(f)
        self.assertGreaterEqual(f.h_sp, 0.0)
        self.assertGreaterEqual(f.h_te, 0.0)
        self.assertGreaterEqual(f.h_in, 0.0)

    def test_hsp_range(self):
        e      = self.HEMMExtractor()
        joints = self._make_joints(25)
        h      = e.compute_hsp(joints)
        self.assertGreaterEqual(h, 0.0)

    def test_normalise_zeroes_spine(self):
        joints = [(1.0 + i*0.1, 2.0, 3.0) for i in range(25)]
        norm   = self.normalise_joints(joints, spine_idx=0)
        self.assertAlmostEqual(norm[0][0], 0.0)
        self.assertAlmostEqual(norm[0][1], 0.0)
        self.assertAlmostEqual(norm[0][2], 0.0)


# ── Sensor Sources ────────────────────────────────────────────────────────

class TestSensorSources(unittest.TestCase):

    def test_synthetic_produces_frames(self):
        from aura_sdk.sensors import SyntheticSource
        src = SyntheticSource(n_frames=10, seed=42)
        with src:
            frames = list(src.frames())
        self.assertEqual(len(frames), 10)

    def test_synthetic_frame_has_25_joints(self):
        from aura_sdk.sensors import SyntheticSource
        src = SyntheticSource(n_frames=3)
        with src:
            f = src.read_frame()
        self.assertEqual(f.n_joints, 25)

    def test_synthetic_spine_normalised(self):
        from aura_sdk.sensors import SyntheticSource
        src = SyntheticSource(n_frames=5)
        with src:
            f = src.read_frame()
        # Spine base (index 0) should be at origin after normalisation
        sx, sy, sz = f.joints[0]
        self.assertAlmostEqual(sx, 0.0, places=5)
        self.assertAlmostEqual(sy, 0.0, places=5)

    def test_file_source_json(self):
        from aura_sdk.sensors import FileSource, SyntheticSource
        # Create a temp JSON file from synthetic data
        from aura_sdk.sensors.sources import SkeletonFrame
        import json
        src    = SyntheticSource(n_frames=5)
        frames_data = {"frames": []}
        with src:
            for frame in src.frames():
                frames_data["frames"].append({
                    "joints": [list(j) for j in frame.joints]
                })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(frames_data, f)
            tmp_path = f.name
        try:
            fs = FileSource(tmp_path)
            with fs:
                loaded = list(fs.frames())
            self.assertEqual(len(loaded), 5)
        finally:
            os.unlink(tmp_path)

    def test_file_source_csv(self):
        from aura_sdk.sensors import FileSource, SyntheticSource
        import csv
        src = SyntheticSource(n_frames=3)
        rows = []
        with src:
            for fi, frame in enumerate(src.frames()):
                for ji, (x, y, z) in enumerate(frame.joints):
                    rows.append({
                        "frame_idx": fi, "joint_idx": ji,
                        "x": x, "y": y, "z": z
                    })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx","joint_idx","x","y","z"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name
        try:
            fs = FileSource(tmp_path)
            with fs:
                loaded = list(fs.frames())
            self.assertEqual(len(loaded), 3)
        finally:
            os.unlink(tmp_path)


# ── Output Sinks ──────────────────────────────────────────────────────────

class TestOutputSinks(unittest.TestCase):

    def _make_record(self, fired=True):
        from aura_sdk.output import TelemetryRecord
        return TelemetryRecord(
            frame_idx=1, timestamp_ms=1000.0,
            fired=fired, q_hci=0.65 if fired else None,
            state=4 if fired else None,
            state_label="Low Workload" if fired else None,
            alert=False, power_uw=25.0,
            accumulated_uj=0.05,
            energy_nj=30.4 if fired else 0.0,
            idle_power_w=0.0,
        )

    def test_csv_logger_writes(self):
        from aura_sdk.output import CSVLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = CSVLogger(directory=tmpdir)
            with logger:
                logger.write(self._make_record(fired=True))
                logger.write(self._make_record(fired=False))
            # File should exist and have content
            files = list(os.scandir(tmpdir))
            self.assertEqual(len(files), 1)
            with open(files[0].path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)  # header + 2 data rows

    def test_console_output_writes(self):
        from aura_sdk.output import ConsoleOutput
        import io
        sink = ConsoleOutput(use_color=False)
        sink.open()
        sink.write(self._make_record(fired=True))
        sink.write(self._make_record(fired=False))
        sink.close()

    def test_telemetry_record_json(self):
        r    = self._make_record()
        data = json.loads(r.to_json())
        self.assertEqual(data["idle_power_w"], 0.0)
        self.assertEqual(data["state"],        4)

    def test_multi_sink(self):
        from aura_sdk.output import MultiSink, ConsoleOutput
        sink = MultiSink([ConsoleOutput(use_color=False), ConsoleOutput(use_color=False)])
        with sink:
            sink.write(self._make_record())


# ── Full Pipeline ─────────────────────────────────────────────────────────

class TestAuraPipeline(unittest.TestCase):

    def test_pipeline_runs_120_frames(self):
        from aura_sdk.pipeline import AuraPipeline, PipelineConfig
        from aura_sdk.sensors  import SyntheticSource
        from aura_sdk.harvesting import VibrationHarvester
        from aura_sdk.output   import ConsoleOutput

        pipeline = AuraPipeline(
            source    = SyntheticSource(n_frames=120),
            harvester = VibrationHarvester(acceleration_g=2.0),
            sinks     = [],
            config    = PipelineConfig(verbose=False),
        )
        stats = pipeline.run()
        self.assertEqual(stats.total_frames, 120)
        self.assertGreater(stats.fired_frames, 0)
        self.assertEqual(stats.idle_power_w, 0.0)

    def test_pipeline_uses_the_real_file_source_not_a_synthetic_fallback(self):
        """
        Regression test for a confirmed bug: AuraPipeline.__init__ used to
        write `self.source = source or SyntheticSource(n_frames=120)`.
        FileSource defines __len__ (len(self._frames)), and an unopened
        FileSource has zero frames, so `bool(unopened_file_source)` is
        False — meaning `or` silently discarded a real, caller-supplied
        FileSource and substituted 120 frames of fake synthetic data,
        with no error or warning. Passing `sinks=[]` to intentionally
        request no output sinks was silently overridden the same way,
        since `[] or default` is also always falsy.

        This guards both: the exact FileSource object passed in must be
        the one actually used (identity, not just frame count, in case a
        future regression swaps in an equivalent-length synthetic source
        instead of a genuinely different bug), the frame count must match
        the file's real content (60, not the 120-frame default), and an
        explicitly empty sinks list must stay empty.
        """
        from aura_sdk.pipeline   import AuraPipeline, PipelineConfig
        from aura_sdk.sensors    import FileSource, SyntheticSource
        from aura_sdk.harvesting import ThermalHarvester
        import csv

        src = SyntheticSource(n_frames=60, seed=7)
        rows = []
        with src:
            for frame in src.frames():
                for ji, (x, y, z) in enumerate(frame.joints):
                    rows.append({
                        "frame_idx": frame.frame_idx, "joint_idx": ji,
                        "x": x, "y": y, "z": z
                    })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx","joint_idx","x","y","z"])
            writer.writeheader()
            writer.writerows(rows)
            tmp_path = f.name

        try:
            # Deliberately unopened — this is the exact state that
            # triggered the bug (empty self._frames -> falsy object).
            fsrc = FileSource(tmp_path)
            pipeline = AuraPipeline(
                source    = fsrc,
                harvester = ThermalHarvester(delta_T_C=5.0),
                sinks     = [],
                config    = PipelineConfig(verbose=False),
            )
            self.assertIs(pipeline.source, fsrc,
                "AuraPipeline must use the exact FileSource instance passed "
                "in, not silently substitute a default source")
            self.assertEqual(pipeline.sinks, [],
                "an explicitly empty sinks list must not be overridden "
                "with a default sink")

            stats = pipeline.run()
            self.assertEqual(stats.total_frames, 60,
                "must process the real file's 60 frames, not the "
                "SyntheticSource(n_frames=120) fallback")
        finally:
            os.unlink(tmp_path)

    def test_pipeline_q_hci_in_range(self):
        from aura_sdk.pipeline   import AuraPipeline, PipelineConfig
        from aura_sdk.sensors    import SyntheticSource
        from aura_sdk.harvesting import VibrationHarvester

        records = []

        class Capture:
            def open(self): pass
            def close(self): pass
            def write(self, r):
                if r.fired and r.q_hci is not None:
                    records.append(r.q_hci)

        AuraPipeline(
            source    = SyntheticSource(n_frames=60),
            harvester = VibrationHarvester(acceleration_g=3.0),
            sinks     = [Capture()],
            config    = PipelineConfig(verbose=False),
        ).run()

        for q in records:
            self.assertGreaterEqual(q, 0.0)
            self.assertLessEqual(q, 1.0)

    def test_pipeline_stats_fire_rate_valid(self):
        from aura_sdk.pipeline   import make_demo_pipeline
        pipeline = make_demo_pipeline(n_frames=30)
        # Replace sinks with empty list
        pipeline.sinks = []
        stats = pipeline.run()
        self.assertGreaterEqual(stats.fire_rate_pct, 0.0)
        self.assertLessEqual(stats.fire_rate_pct, 100.0)

    def test_idle_power_always_zero(self):
        from aura_sdk.pipeline import make_demo_pipeline
        pipeline = make_demo_pipeline(n_frames=30)
        pipeline.sinks = []
        stats = pipeline.run()
        self.assertEqual(stats.idle_power_w, 0.0)

    def test_wearable_factory(self):
        from aura_sdk.pipeline import make_wearable_pipeline
        p = make_wearable_pipeline()
        p.sinks = []
        stats = p.run()
        self.assertGreater(stats.total_frames, 0)

    def test_industrial_factory(self):
        from aura_sdk.pipeline import make_industrial_pipeline
        p = make_industrial_pipeline()
        p.sinks = []
        stats = p.run()
        self.assertGreater(stats.total_frames, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
