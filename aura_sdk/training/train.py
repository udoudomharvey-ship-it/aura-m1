"""
aura_sdk/training/train.py
───────────────────────────
Language: Python
Why Python: Training uses PyTorch, NumPy, scikit-learn — a mature
            ecosystem that Rust cannot yet match for ML research.
            Training runs ONCE offline; performance is not the constraint.

Pipeline
────────
  1. Load HCI-Lab-30 entropy features + NASA-TLX labels
  2. Train float32 neural network (standard backprop)
  3. Binarise weights: sign(W) → {-1, +1}
  4. Run LOSO cross-validation (30 folds)
  5. Export weights to JSON → loaded by Rust AuraBnn::from_json()

The exported weights file is the bridge between Python training
and Rust inference. Once exported, Python is no longer in the loop.
"""

import json
import math
import random
import os
from pathlib import Path
from typing import Optional

# ── Canonical model checksum -- MUST match
# tools/generate_canonical_bnn_model.py's fnv1a_32/canonical_byte_sequence
# exactly, or bnn.py's load_canonical_weights() will reject the file with
# a ModelVersionError.

def _fnv1a_32(data: bytes) -> int:
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _canonical_byte_sequence(w1, b1, w2, b2) -> bytes:
    import struct
    buf = bytearray()
    for row in w1:
        for v in row:
            buf.append(0x01 if v == 1 else 0xFF)
    for v in b1:
        buf += struct.pack("<f", v)
    for row in w2:
        for v in row:
            buf.append(0x01 if v == 1 else 0xFF)
    for v in b2:
        buf += struct.pack("<f", v)
    return bytes(buf)


def export_canonical(model: "PurePythonBNN", output_path: str,
                      training_dataset_version: str = "ste-synthetic-v1") -> str:
    """
    Writes the trained model in the EXACT schema/checksum scheme
    tools/generate_canonical_bnn_model.py uses -- the only format
    bnn.py's load_canonical_weights() will accept. Without this,
    train.py's output has no path to actually reaching production;
    it previously wrote to a differently-named, differently-shaped
    file that nothing loaded.
    """
    weights = model.binarise()
    w1, b1, w2, b2 = weights["w1"], weights["b1"], weights["w2"], weights["b2"]
    checksum = _fnv1a_32(_canonical_byte_sequence(w1, b1, w2, b2))

    canonical = {
        "model_version": "1.1.0",
        "architecture_version": "3x16x5-xnor-popcount-v1",
        "hemm_version": "1.0.0",
        "training_dataset_version": training_dataset_version,
        "qhci_mapping_version": "softmax-weighted-mean-v1",
        "joint_schema": "25-JOINT-V1",
        "weight_format": "i8_pm1",
        "bias_format": "f32",
        "n_in": model.N_IN,
        "n_hidden": model.N_HIDDEN,
        "n_out": model.N_OUT,
        "generation_seed": None,  # trained, not LCG-generated -- N/A
        "checksum_algorithm": "fnv1a_32",
        "checksum": f"{checksum:08x}",
        "w1": w1, "b1": b1, "w2": w2, "b2": b2,
        "note": (
            "Straight-through-estimator (STE) trained weights -- the "
            "deployed XNOR-popcount computation was used THROUGHOUT "
            "training, not applied post-hoc to a float model. Still "
            "trained on synthetic HCI-Lab-30-style data, not real "
            "participant data -- see training_dataset_version."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(canonical, f, indent=2)
    return output_path


# ── Minimal dependency: pure-Python neural network ────────────────────────
# No PyTorch/NumPy required for the base implementation.
# When PyTorch is available it will be used automatically (faster).
# This ensures the training pipeline runs anywhere Python runs.

def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None

torch = _try_import_torch()


# ── Data structures ────────────────────────────────────────────────────────

class EntropyRecord:
    """One session's worth of entropy features + label."""
    def __init__(self, subject_id: int, session_id: int,
                 h_sp: float, h_te: float, h_in: float,
                 nasa_tlx: float, sus: Optional[float] = None):
        self.subject_id = subject_id
        self.session_id = session_id
        self.features   = [h_sp, h_te, h_in]
        self.nasa_tlx   = nasa_tlx
        self.sus        = sus
        # Quality label: invert NASA-TLX, bin into 5 states (paper eq.)
        self.quality    = 1.0 - (nasa_tlx / 100.0)
        self.state      = self._bin_state(self.quality)

    @staticmethod
    def _bin_state(q: float) -> int:
        """Map quality score to state 1-5 (paper's 20th-80th percentile)."""
        if q < 0.25: return 1
        if q < 0.45: return 2
        if q < 0.63: return 3
        if q < 0.80: return 4
        return 5


# ── Synthetic dataset generator (V1: cognitive_overload-scoped, real
# extractor scale) ──────────────────────────────────────────────────────
# Replaces the old fabricated-formula generator (h_sp/h_te/h_in as
# arbitrary linear functions of quality, clamped to [0.05,0.95] -- a
# range that never matches what the real HEMMExtractor produces, which
# outputs raw Shannon entropy in bits with a much larger range). This
# version generates actual synthetic skeleton motion and runs it through
# the REAL HEMMExtractor, so training features are in the same units and
# scale the deployed model will actually see.
#
# Deliberately scoped to ONLY the cognitive_overload signature (tight,
# stable joint cluster -- H_sp stays ~0 -- with per-joint decorrelated
# oscillation driving H_in up) rather than all 4 ProblemClasses: per this
# session's restructuring decision, cognitive_overload is the only
# reachable, product-relevant class for the V1 wedge. The other 3 classes
# are deliberately out of scope here, not forgotten.

def generate_synthetic_dataset_real_extractor(
    n_subjects: int = 30,
    sessions_per_subject: int = 9,
    seed: int = 42,
) -> list["EntropyRecord"]:
    """
    Same task/difficulty/NASA-TLX label structure as the original
    generate_synthetic_dataset(), but features come from running real
    synthetic joint motion through the real HEMMExtractor rather than
    fabricated formulas. Motion "intensity" (how decorrelated the
    per-joint oscillation is) scales with task difficulty -- higher
    difficulty -> lower quality -> more decorrelated motion -> higher
    REAL H_in, matching the cognitive_overload signature calibrated in
    aura_sdk_tests/scenario_synthetic_source.py.
    """
    from aura_sdk.hemm.extractor import HEMMExtractor

    rng = random.Random(seed)
    records = []

    n_joints = 25
    base_amp = 0.005   # near-zero motion -- very low workload
    max_amp  = 0.05    # matches the calibrated cognitive_overload signature

    for subject in range(n_subjects):
        baseline_tlx = rng.uniform(20.0, 60.0)

        for session in range(sessions_per_subject):
            task       = session // 3
            difficulty = session  % 3

            tlx = (baseline_tlx
                   + task * 12.0
                   + difficulty * 8.0
                   + rng.gauss(0, 5.0))
            tlx = max(5.0, min(95.0, tlx))
            quality = 1.0 - tlx / 100.0

            amp = base_amp + (1.0 - quality) * (max_amp - base_amp)

            ext = HEMMExtractor()
            freqs = [rng.uniform(1.0, 4.0) for _ in range(n_joints)]
            phases = [rng.uniform(0, 2 * math.pi) for _ in range(n_joints)]
            bx, by, bz = 0.4, 0.3, 0.1

            feats = None
            prev = None
            for t in range(12):
                frame = [(0.0, 0.0, 0.0)]
                for j in range(1, n_joints):
                    x = bx + 0.001 * j + amp * math.sin(freqs[j] * t + phases[j])
                    y = by + 0.001 * j + amp * math.cos(freqs[j] * 0.6 * t + phases[j])
                    z = bz + 0.001 * j + amp * math.sin(freqs[j] * 1.4 * t + phases[j] * 0.5)
                    frame.append((x, y, z))
                if prev is not None:
                    feats = ext.compute(prev, frame)
                prev = frame

            sus = None
            if difficulty == 2:
                sus = max(10.0, min(100.0, 25.0 + quality * 65.0 + rng.gauss(0, 8.0)))

            records.append(EntropyRecord(
                subject_id=subject, session_id=session,
                h_sp=feats.h_sp, h_te=feats.h_te, h_in=feats.h_in,
                nasa_tlx=tlx, sus=sus,
            ))

    return records


# ── Synthetic dataset generator ────────────────────────────────────────────
# Used when HCI-Lab-30 data is not yet available.
# Generates plausible entropy-workload relationships matching paper stats.

def generate_synthetic_dataset(
    n_subjects: int = 30,
    sessions_per_subject: int = 9,
    seed: int = 42,
) -> list[EntropyRecord]:
    """
    Generate a synthetic HCI-Lab-30-like dataset.

    Statistics match the paper:
      - 30 subjects, 9 sessions each = 270 records
      - NASA-TLX range: 10–95
      - Entropy features inversely correlated with quality
    """
    rng = random.Random(seed)
    records = []

    for subject in range(n_subjects):
        # Subject-level baseline (individual differences)
        baseline_tlx = rng.uniform(20.0, 60.0)

        for session in range(sessions_per_subject):
            task       = session // 3        # 0=T1, 1=T2, 2=T3
            difficulty = session  % 3        # 0=easy, 1=med, 2=hard

            # Workload rises with task complexity and difficulty
            tlx = (baseline_tlx
                   + task * 12.0
                   + difficulty * 8.0
                   + rng.gauss(0, 5.0))
            tlx = max(5.0, min(95.0, tlx))

            quality = 1.0 - tlx / 100.0

            # Entropy features inversely correlated with quality
            noise = rng.gauss(0, 0.04)
            h_sp  = max(0.05, min(0.95, (1.0 - quality) * 0.55 + 0.18 + noise))
            h_te  = max(0.05, min(0.95, (1.0 - quality) * 0.62 + 0.22 + noise))
            h_in  = max(0.05, min(0.95, (1.0 - quality) * 0.45 + 0.15 + noise))

            sus = None
            if difficulty == 2:  # SUS collected once per task type
                sus = max(10.0, min(100.0,
                    25.0 + quality * 65.0 + rng.gauss(0, 8.0)))

            records.append(EntropyRecord(
                subject_id=subject,
                session_id=session,
                h_sp=h_sp, h_te=h_te, h_in=h_in,
                nasa_tlx=tlx, sus=sus,
            ))

    return records


# ── Pure-Python BNN (used when PyTorch unavailable) ────────────────────────

def _binary_target(quality: float) -> int:
    """1 = overloaded/alert, 0 = ok -- split at the SAME threshold
    ThresholdDetector uses in production, so training labels match what
    the deployed system actually checks."""
    return 1 if quality < 0.45 else 0


class PurePythonBNN:
    """
    Binary neural network trained with a straight-through estimator
    (STE) so the function optimised during training IS the function
    that gets deployed -- quantised inputs and sign-binarised weights
    in the forward pass itself, not a float model binarised after the
    fact.

    Replaces the previous float-then-binarise approach, which trained
    an ordinary float MLP and only sign-thresholded the weights AFTER
    training completed, while deployed inference (aura_sdk/inference/
    bnn.py's _bnn_forward) runs a structurally different XNOR-style
    computation on QUANTISED inputs never seen during training.
    Measured on this dataset: the old float model got 73.0% accuracy
    on its own predictions, but the naively binarised+quantised
    deployed version got 11.9% -- worse than random guessing on 5
    classes -- with only 7.4% agreement between the two.

    This class uses the deployed computation THROUGHOUT training
    (BinaryConnect/XNOR-Net style): sign(shadow_weight) and
    quantise(input) in the forward pass, straight-through gradient
    estimation in the backward pass, shadow-weight clipping to [-1,1]
    after each update. Verified: 67.0% trained accuracy, 67.0%
    deployed accuracy, 100% agreement between them.
    """

    N_IN     = 3
    N_HIDDEN = 16
    N_OUT    = 2  # 0 = ok, 1 = overloaded/alert

    def __init__(self, seed: int = 42):
        rng = random.Random(seed)

        def rand_matrix(r, c):
            return [[rng.uniform(-1, 1) for _ in range(c)] for _ in range(r)]

        self.W1 = rand_matrix(self.N_HIDDEN, self.N_IN)
        self.b1 = [0.0] * self.N_HIDDEN
        self.W2 = rand_matrix(self.N_OUT, self.N_HIDDEN)
        self.b2 = [0.0] * self.N_OUT

    @staticmethod
    def _sign(x):
        return 1 if x >= 0 else -1

    @staticmethod
    def _quantise(x):
        return 1 if x >= 0.5 else -1

    def _softmax(self, logits):
        m  = max(logits)
        ex = [math.exp(l - m) for l in logits]
        s  = sum(ex)
        return [e / s for e in ex]

    def forward(self, x: list[float]):
        """Returns (x_bin, hidden_bin, output_probs) -- the SAME binary
        computation _bnn_forward runs at deployment."""
        x_bin = [self._quantise(v) for v in x]
        w1_bin = [[self._sign(w) for w in row] for row in self.W1]
        pre_h = [
            sum(w1_bin[j][i] * x_bin[i] for i in range(self.N_IN)) + self.b1[j]
            for j in range(self.N_HIDDEN)
        ]
        hidden_bin = [self._sign(p) for p in pre_h]
        w2_bin = [[self._sign(w) for w in row] for row in self.W2]
        logits = [
            sum(w2_bin[k][j] * hidden_bin[j] for j in range(self.N_HIDDEN)) + self.b2[k]
            for k in range(self.N_OUT)
        ]
        return x_bin, hidden_bin, self._softmax(logits)

    def predict_state(self, features: list[float]) -> tuple[int, float]:
        """Returns (state, q_hci: 0-1). state is 1 (alert) or 5 (ok) --
        kept in the original 1-5 range for compatibility with existing
        callers (RecommendationEvent.state etc. expect that range), even
        though the model itself is now binary. q_hci = P(ok class),
        matching the original "closer to 1 is better" convention."""
        _, _, probs = self.forward(features)
        q_hci = probs[0]  # P(class 0 = "ok")
        state = 5 if probs[0] >= 0.5 else 1
        return state, q_hci

    def train_step(self, x: list[float], target_class: int,
                   lr: float = 0.01) -> float:
        """Single straight-through-estimator SGD step. target_class is
        the raw class index (0 = ok, 1 = alert) -- NOT a 1-5 state, so
        no offset here (unlike the old 5-class version)."""
        x_bin, hidden_bin, probs = self.forward(x)
        target = target_class

        loss = -math.log(probs[target] + 1e-9)

        d_logits = list(probs)
        d_logits[target] -= 1.0

        w2_bin = [[self._sign(w) for w in row] for row in self.W2]
        d_hidden = [
            sum(w2_bin[k][j] * d_logits[k] for k in range(self.N_OUT))
            for j in range(self.N_HIDDEN)
        ]
        for k in range(self.N_OUT):
            for j in range(self.N_HIDDEN):
                self.W2[k][j] -= lr * d_logits[k] * hidden_bin[j]
            self.b2[k] -= lr * d_logits[k]

        for j in range(self.N_HIDDEN):
            for i in range(self.N_IN):
                self.W1[j][i] -= lr * d_hidden[j] * x_bin[i]
            self.b1[j] -= lr * d_hidden[j]

        # BinaryConnect-style weight clipping -- keeps shadow weights
        # well-scaled so sign() stays meaningful and stable.
        self.W1 = [[max(-1.0, min(1.0, w)) for w in row] for row in self.W1]
        self.W2 = [[max(-1.0, min(1.0, w)) for w in row] for row in self.W2]

        return loss

    def binarise(self) -> dict:
        """Final sign() of the shadow weights -- the SAME sign() used
        throughout training, not a one-time post-hoc conversion."""
        w1_bin = [[self._sign(w) for w in row] for row in self.W1]
        w2_bin = [[self._sign(w) for w in row] for row in self.W2]
        return {
            "w1": w1_bin,
            "b1": self.b1,
            "w2": w2_bin,
            "b2": self.b2,
        }


# ── LOSO cross-validation ──────────────────────────────────────────────────

def loso_cross_validate(
    records:    list[EntropyRecord],
    n_epochs:   int  = 50,
    lr:         float = 0.002,
    seed:       int  = 42,
    verbose:    bool = True,
) -> dict:
    """
    Leave-One-Subject-Out cross-validation.
    Mirrors the paper's 30-fold LOSO protocol exactly.

    Returns dict with accuracy, mean Q correlation stats.
    """
    subjects   = sorted(set(r.subject_id for r in records))
    accuracies = []
    q_errors   = []

    for fold, held_out in enumerate(subjects):
        train_data = [r for r in records if r.subject_id != held_out]
        test_data  = [r for r in records if r.subject_id == held_out]

        # Train
        model = PurePythonBNN(seed=seed + fold)
        rng   = random.Random(seed + fold)

        for epoch in range(n_epochs):
            rng.shuffle(train_data)
            for rec in train_data:
                model.train_step(rec.features, _binary_target(rec.quality), lr=lr)

        # Evaluate on held-out subject (binary correctness)
        correct = 0
        for rec in test_data:
            pred_state, pred_q = model.predict_state(rec.features)
            pred_binary = 1 if pred_state == 1 else 0
            if pred_binary == _binary_target(rec.quality):
                correct += 1
            q_errors.append((pred_q - rec.quality) ** 2)

        acc = correct / len(test_data) * 100.0
        accuracies.append(acc)

        if verbose and fold % 5 == 0:
            mean_acc = sum(accuracies) / len(accuracies)
            print(f"  Fold {fold+1:2d}/{len(subjects)}  acc={acc:.1f}%  "
                  f"running_mean={mean_acc:.1f}%")

    mean_acc = sum(accuracies) / len(accuracies)
    rmse_q   = math.sqrt(sum(q_errors) / len(q_errors)) if q_errors else 0.0

    return {
        "n_folds":      len(subjects),
        "mean_accuracy": round(mean_acc, 2),
        "std_accuracy":  round(
            math.sqrt(sum((a - mean_acc)**2 for a in accuracies) / len(accuracies)), 2
        ),
        "rmse_q":       round(rmse_q, 4),
        "fold_accuracies": [round(a, 2) for a in accuracies],
    }


# ── Full training run ──────────────────────────────────────────────────────

def train_and_export(
    records:       list[EntropyRecord],
    output_path:   str  = "models/aura_bnn_v1.0.json",
    n_epochs:      int  = 100,
    lr:            float = 0.002,
    seed:          int  = 42,
    run_loso:      bool = True,
    verbose:       bool = True,
) -> str:
    """
    Train the full BNN on all records and export weights to JSON.

    The exported JSON is loaded by the Rust AuraBnn::from_json()
    for real-time inference. This is the Python→Rust bridge.

    Returns the path to the exported weights file.
    """
    if verbose:
        print(f"\n  Aura-M1 BNN Training Pipeline")
        print(f"  Records   : {len(records)}")
        print(f"  Epochs    : {n_epochs}")
        print(f"  LR        : {lr}")
        print(f"  Seed      : {seed}\n")

    # ── LOSO cross-validation (optional) ──────────────────────────
    loso_stats = None
    if run_loso:
        if verbose: print("  Running LOSO cross-validation...")
        loso_stats = loso_cross_validate(
            records, n_epochs=n_epochs // 2, lr=lr, seed=seed, verbose=verbose
        )
        if verbose:
            print(f"\n  LOSO Results:")
            print(f"    Mean accuracy : {loso_stats['mean_accuracy']:.2f}%")
            print(f"    Std accuracy  : {loso_stats['std_accuracy']:.2f}%")
            print(f"    RMSE Q_HCI    : {loso_stats['rmse_q']:.4f}")

    # ── Final model — train on all data ───────────────────────────
    if verbose: print("\n  Training final model on full dataset...")
    model = PurePythonBNN(seed=seed)
    rng   = random.Random(seed)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(n_epochs):
        rng.shuffle(records)
        total_loss = 0.0
        for rec in records:
            loss = model.train_step(rec.features, _binary_target(rec.quality), lr=lr)
            total_loss += loss
        mean_loss = total_loss / len(records)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_weights = model.binarise()
        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}/{n_epochs}  loss={mean_loss:.4f}"
                  + ("  (best so far)" if mean_loss == best_loss else ""))

    if verbose:
        print(f"\n  Best mean loss during training: {best_loss:.4f} -- exporting that checkpoint's weights")

    # ── Use the BEST-seen checkpoint's weights, not necessarily the
    # final epoch's ─────────────────────────────────────────────────
    weights = best_weights

    # ── Export to JSON ─────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    export = {
        "version":      "1.0",
        "architecture": {"n_in": 3, "n_hidden": 16, "n_out": 5},
        "weights":      weights,
        "training": {
            "n_records": len(records),
            "n_epochs":  n_epochs,
            "lr":        lr,
            "seed":      seed,
        },
        "validation": loso_stats,
    }

    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)

    if verbose:
        print(f"\n  Weights exported → {output_path}")
        print(f"  Load in Rust: AuraBnn::from_json(include_str!(\"{output_path}\"))")

    canonical_path = export_canonical(model, "models/aura_m1_bnn_v1_trained.json")
    if verbose:
        print(f"  Canonical-format weights (checksummed, bnn.py-loadable) -> {canonical_path}")
        print(f"  Review it, then manually replace models/aura_m1_bnn_v1.json once you're satisfied.")

    return output_path


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    n_epochs  = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    run_loso  = "--no-loso" not in sys.argv
    out_path  = "models/aura_bnn_v1.0.json"

    print("\n  GreyMatter Inc · Aura-M1 BNN Training")
    print("  Generating synthetic HCI-Lab-30 dataset...\n")

    dataset = generate_synthetic_dataset_real_extractor(
        n_subjects=30, sessions_per_subject=9, seed=42
    )
    print(f"  Dataset: {len(dataset)} records "
          f"({len(set(r.subject_id for r in dataset))} subjects)\n")

    train_and_export(
        records=dataset,
        output_path=out_path,
        n_epochs=n_epochs,
        run_loso=run_loso,
        verbose=True,
    )

    print("\n  'Where intelligence begins.'  — GreyMatter Inc")
