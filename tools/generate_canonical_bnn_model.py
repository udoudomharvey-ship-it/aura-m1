"""
tools/generate_canonical_bnn_model.py
────────────────────────────────────────
Language: Python
Why this exists: an audit found THREE genuinely different weight-generation
schemes across Python, Rust, and firmware, all nominally claiming "seed=42":

  - Python  (aura_sdk/inference/bnn.py):  random.Random(42) — Mersenne
    Twister — one continuous stream for W1, b1, W2, b2 in that order.
  - Rust    (aura_core/src/inference/mod.rs): a hand-rolled 32-bit LCG,
    ONE stream (state=42) for W1+W2, a SEPARATE stream (state=99) for b1+b2.
  - Firmware (aura_firmware/src/main.rs): the same LCG family, but W1 used
    seed=42 and W2 used a completely different, separately-seeded stream
    at seed=1337 — and biases were not generated or applied at all.

None of these three produce matching weight values, despite all three
being labelled "seed=42" in comments. This script replaces all of them
with ONE canonical generation scheme, run once, whose output is committed
as models/aura_m1_bnn_v1.json. Python, Rust, and firmware are all updated
to LOAD this file's values (or literal copies of them) rather than each
independently regenerating from a seed.

Generation scheme (deliberately the simplest possible — one continuous
32-bit LCG stream, Numerical Recipes constants, in this exact order):
    state = 42
    W1 (16x3 signs)   ← consume 48 LCG steps, sign = top bit
    b1 (16 floats)    ← consume 16 more LCG steps, value = (state/u32::MAX) - 0.5
    W2 (5x16 signs)   ← consume 80 more LCG steps, sign = top bit
    b2 (5 floats)     ← consume 5 more LCG steps, value = (state/u32::MAX) - 0.5

Checksum: FNV-1a 32-bit over the exact byte sequence described in
`canonical_byte_sequence()` below. FNV-1a was chosen (over e.g. SHA-256)
specifically because it's ~10 lines to hand-implement identically in Rust
with zero external dependencies — important since aura_firmware is a
no_std, no-crate-bloat target. It's an integrity/version-mismatch guard,
not a cryptographic guarantee.

Regenerate with:
    python3 tools/generate_canonical_bnn_model.py

Only regenerate if you deliberately want a NEW canonical weight set (e.g.
after real training). Doing so changes model_version and invalidates every
downstream copy — see MODEL_VERSION below.
"""
import json
import os

N_IN, N_HIDDEN, N_OUT = 3, 16, 5
MODEL_VERSION = "1.0.0"
ARCHITECTURE_VERSION = "3x16x5-xnor-popcount-v1"
HEMM_VERSION = "1.0.0"          # matches HEMM_WINDOW_CAP=5 windowed entropy fix
TRAINING_DATASET_VERSION = "seeded-placeholder-v1"  # NOT trained on real data
QHCI_MAPPING_VERSION = "softmax-weighted-mean-v1"
JOINT_SCHEMA = "25-JOINT-V1"

LCG_MULT = 1_664_525
LCG_INC = 1_013_904_223
MASK32 = 0xFFFFFFFF


class Lcg:
    """32-bit LCG, Numerical Recipes constants. Bit-for-bit reproducible
    in Rust with plain u32 wrapping_mul/wrapping_add — the whole point."""
    def __init__(self, seed: int):
        self.state = seed & MASK32

    def next_u32(self) -> int:
        self.state = (self.state * LCG_MULT + LCG_INC) & MASK32
        return self.state

    def next_sign(self) -> int:
        s = self.next_u32()
        return 1 if (s >> 31) == 1 else -1

    def next_bias(self) -> float:
        s = self.next_u32()
        return (s / MASK32) - 0.5


def generate_canonical_weights(seed: int = 42):
    rng = Lcg(seed)
    w1 = [[rng.next_sign() for _ in range(N_IN)] for _ in range(N_HIDDEN)]
    b1 = [rng.next_bias() for _ in range(N_HIDDEN)]
    w2 = [[rng.next_sign() for _ in range(N_HIDDEN)] for _ in range(N_OUT)]
    b2 = [rng.next_bias() for _ in range(N_OUT)]
    return w1, b1, w2, b2


def fnv1a_32(data: bytes) -> int:
    """FNV-1a 32-bit hash. Public domain algorithm, ~10 lines in any
    language, no dependencies. Used as an integrity/version checksum."""
    h = 0x811C9DC5  # FNV offset basis
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime
    return h


def canonical_byte_sequence(w1, b1, w2, b2) -> bytes:
    """
    Deterministic byte encoding used ONLY for the checksum — not the JSON
    storage format. Order: all w1 signs (as 0x01/0xFF for +1/-1), then all
    b1 as little-endian f32, then all w2 signs, then all b2 as f32. This
    exact byte layout must be reproduced identically by the Rust checksum
    verifier — see aura_core::inference::bnn_model checksum test.
    """
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


def build_canonical_model(seed: int = 42) -> dict:
    w1, b1, w2, b2 = generate_canonical_weights(seed)
    checksum = fnv1a_32(canonical_byte_sequence(w1, b1, w2, b2))
    return {
        "model_version": MODEL_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "hemm_version": HEMM_VERSION,
        "training_dataset_version": TRAINING_DATASET_VERSION,
        "qhci_mapping_version": QHCI_MAPPING_VERSION,
        "joint_schema": JOINT_SCHEMA,
        "weight_format": "i8_pm1",
        "bias_format": "f32",
        "n_in": N_IN,
        "n_hidden": N_HIDDEN,
        "n_out": N_OUT,
        "generation_seed": seed,
        "checksum_algorithm": "fnv1a_32",
        "checksum": f"{checksum:08x}",
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "note": (
            "Weights are deterministic placeholders (LCG seed=42), NOT "
            "trained on real HCI participant data. This is the single "
            "canonical source of truth for Python/Rust/firmware — none of "
            "them should independently regenerate weights from a seed "
            "anymore. Retraining replaces this file's w1/b1/w2/b2 and bumps "
            "model_version; the generation scheme documented in this "
            "script's docstring stays fixed regardless, so the checksum "
            "verification mechanism keeps working."
        ),
    }


def main():
    model = build_canonical_model()
    out_path = os.path.join(os.path.dirname(__file__), "..", "models", "aura_m1_bnn_v1.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Wrote canonical model to {out_path}")
    print(f"  model_version: {model['model_version']}")
    print(f"  checksum:      {model['checksum']}")
    print(f"  w1[0]: {model['w1'][0]}")
    print(f"  b1[0]: {model['b1'][0]:.6f}")
    print(f"  w2[0]: {model['w2'][0]}")
    print(f"  b2[0]: {model['b2'][0]:.6f}")


if __name__ == "__main__":
    main()
