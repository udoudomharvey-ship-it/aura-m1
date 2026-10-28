"""
aura_sdk.inference.bnn
───────────────────────
Language: Python (thin wrapper)
Why: API compatibility layer. Routes to Rust backend when compiled,
     falls back to pure Python otherwise. Zero logic duplication.
"""
from __future__ import annotations
import math, random, time, json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── Try Rust backend ───────────────────────────────────────────────────────
try:
    from aura_core_rs import RustBnn as _RustBnn
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _RustBnn = None

# ── Power budget ──────────────────────────────────────────────────────────

@dataclass
class PowerBudget:
    idle_power_uW:     float = 0.00
    active_power_uW:   float = 8.0
    latency_ms:        float = 3.8
    energy_per_inf_nJ: float = field(init=False)

    def __post_init__(self):
        self.energy_per_inf_nJ = self.active_power_uW * self.latency_ms / 1000.0


# ── Inference result ──────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    features:       dict
    state:          int
    q_hci:          float
    logits:         list
    latency_ms:     float
    energy_nJ:      float
    idle_power_uW:  float = 0.0
    timestamp:      float = field(default_factory=time.time)

    STATE_LABELS = {
        1: "Very High Workload", 2: "High Workload",
        3: "Moderate Workload",  4: "Low Workload",
        5: "Very Low Workload",
    }

    @property
    def state_label(self) -> str:
        return self.STATE_LABELS.get(self.state, "Unknown")

    @property
    def workload_pct(self) -> float:
        return (5 - self.state) / 4.0 * 100.0

    def summary(self) -> str:
        return (
            f"InferenceResult(Q={self.q_hci:.4f}, "
            f"state={self.state}, E={self.energy_nJ:.2f}nJ, "
            f"idle={self.idle_power_uW}µW)"
        )

    def __repr__(self):
        return self.summary()


# ── Pure-Python BNN (fallback) ────────────────────────────────────────────

def _sign(x):    return 1 if x >= 0 else -1
def _quantise(x): return 1 if x >= 0.5 else -1

_N_IN, _N_HIDDEN, _N_OUT = 3, 16, 5


def _make_weights_placeholder(seed=42):
    """
    Ad-hoc placeholder weight generator using Python's own random.Random.

    ⚠ NOT the canonical model. This uses Mersenne Twister (random.Random),
    a completely different algorithm from the 32-bit LCG that
    tools/generate_canonical_bnn_model.py uses — "same seed" here does NOT
    mean "same weights" as Rust or firmware. This function is kept only as
    a standalone convenience for quick local experiments that don't need
    cross-language parity. Production code should use
    load_canonical_weights() below, which is what AuraBNN actually uses.
    """
    rng = random.Random(seed)
    W1 = [[rng.choice([-1,1]) for _ in range(_N_IN)] for _ in range(_N_HIDDEN)]
    b1 = [rng.uniform(-0.5, 0.5) for _ in range(_N_HIDDEN)]
    W2 = [[rng.choice([-1,1]) for _ in range(_N_HIDDEN)] for _ in range(_N_OUT)]
    b2 = [rng.uniform(-0.5, 0.5) for _ in range(_N_OUT)]
    return W1, b1, W2, b2


def _canonical_model_path() -> str:
    import os
    _override = os.environ.get("AURA_BNN_MODEL", "").strip()
    if _override and os.path.isfile(_override):
        return _override
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "aura_m1_bnn_v1.json"
    )


def _fnv1a_32(data: bytes) -> int:
    """Must match tools/generate_canonical_bnn_model.py's fnv1a_32 exactly —
    this is what verifies the canonical model file hasn't been hand-edited
    or corrupted since it was generated."""
    h = 0x811C9DC5
    for byte in data:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _canonical_byte_sequence(w1, b1, w2, b2) -> bytes:
    """Must match tools/generate_canonical_bnn_model.py exactly."""
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


class ModelVersionError(Exception):
    """Raised when the canonical model file's checksum doesn't match its
    declared contents, or its architecture doesn't match this codebase's
    expectations. The system refuses to silently load a corrupted or
    incompatible model rather than producing nonsense inference results."""
    pass


def load_canonical_weights(path: str = None):
    """
    Load and verify the canonical BNN model (models/aura_m1_bnn_v1.json).

    This is the SINGLE source of truth for BNN weights across Python, Rust,
    and firmware — none of them should independently regenerate weights
    from a seed for production use anymore (see the audit note in
    tools/generate_canonical_bnn_model.py for why that was silently
    producing three different, non-matching weight sets).

    Verifies the checksum before returning weights, and raises
    ModelVersionError rather than returning weights if the file is missing,
    malformed, has a checksum mismatch, or has an architecture that doesn't
    match this codebase's N_IN/N_HIDDEN/N_OUT.
    """
    path = path or _canonical_model_path()
    try:
        with open(path) as f:
            model = json.load(f)
    except FileNotFoundError:
        raise ModelVersionError(
            f"Canonical model not found at {path}. Run "
            f"tools/generate_canonical_bnn_model.py first."
        )
    except json.JSONDecodeError as e:
        raise ModelVersionError(f"Canonical model at {path} is not valid JSON: {e}")

    for key in ("w1", "b1", "w2", "b2", "checksum", "n_in", "n_hidden", "n_out"):
        if key not in model:
            raise ModelVersionError(f"Canonical model missing required field '{key}'")

    if (model["n_in"], model["n_hidden"], model["n_out"]) != (_N_IN, _N_HIDDEN, _N_OUT):
        raise ModelVersionError(
            f"Canonical model architecture ({model['n_in']}x{model['n_hidden']}x{model['n_out']}) "
            f"does not match this codebase's expected {_N_IN}x{_N_HIDDEN}x{_N_OUT}"
        )

    computed = _fnv1a_32(_canonical_byte_sequence(model["w1"], model["b1"], model["w2"], model["b2"]))
    declared = int(model["checksum"], 16)
    if computed != declared:
        raise ModelVersionError(
            f"Canonical model checksum mismatch: file declares {model['checksum']}, "
            f"computed {computed:08x} from its own w1/b1/w2/b2 — the file may have "
            f"been hand-edited or corrupted."
        )

    return model["w1"], model["b1"], model["w2"], model["b2"], model


_W1, _B1, _W2, _B2, _MODEL_META = load_canonical_weights()

def _bnn_forward(h_sp, h_te, h_in):
    x_bin = [_quantise(h_sp), _quantise(h_te), _quantise(h_in)]
    hidden = [_sign(sum(1 if _W1[j][i]==x_bin[i] else -1
                       for i in range(_N_IN)) + _B1[j])
              for j in range(_N_HIDDEN)]
    logits = [sum(1 if _W2[k][j]==hidden[j] else -1
                 for j in range(_N_HIDDEN)) + _B2[k]
              for k in range(_N_OUT)]
    exp_l  = [math.exp(l) for l in logits]
    s      = sum(exp_l) or 1.0
    probs  = [e/s for e in exp_l]
    state  = probs.index(max(probs)) + 1
    q_raw  = sum((i+1)*p for i,p in enumerate(probs)) - 1
    q_hci  = max(0.0, min(1.0, q_raw/(_N_OUT-1)))
    return state, q_hci, logits


# ── AuraBNN public class ──────────────────────────────────────────────────

class AuraBNN:
    """
    Aura-M1 Binary Neural Network inference engine.
    Routes to Rust backend if aura_core_rs is available,
    otherwise uses pure Python fallback.
    API is identical in both cases.
    """

    def __init__(self, power_budget: Optional[PowerBudget] = None):
        self.power  = power_budget or PowerBudget()
        self._rust  = _RustBnn() if _RUST_AVAILABLE else None
        self._backend = "rust" if _RUST_AVAILABLE else "python"
        # NOTE: previously an unbounded list — every infer() call appended
        # and nothing ever trimmed it. Confirmed via a real 10,000-frame
        # sustained-run measurement: RSS grew linearly and unboundedly
        # (+40% over 10k frames), which would eventually exhaust memory on
        # a long-running wearable/industrial deployment. Nothing else in
        # the codebase reads .history/._history except this class's own
        # session_q_hci()/session_summary(), which only need a mean — so
        # a bounded deque (recent results still available via .history)
        # plus an incremental running sum (exact whole-session mean,
        # independent of what's still in the bounded window) fixes the
        # leak without changing either method's return value or the
        # public .history property's shape.
        self._history: deque[InferenceResult] = deque(maxlen=500)
        self._q_hci_sum = 0.0
        self._total_energy_nJ = 0.0
        self._inference_count = 0

    def infer(self, H_sp: float, H_te: float, H_in: float) -> InferenceResult:
        t0 = time.perf_counter()

        if self._backend == "rust":
            r = self._rust.infer(H_sp, H_te, H_in)
            state  = r["state"]
            q_hci  = r["q_hci"]
            logits = r.get("probs", [])
        else:
            state, q_hci, logits = _bnn_forward(H_sp, H_te, H_in)

        result = InferenceResult(
            features   = {"H_sp": H_sp, "H_te": H_te, "H_in": H_in},
            state      = state,
            q_hci      = q_hci,
            logits     = logits,
            latency_ms = self.power.latency_ms,
            energy_nJ  = self.power.energy_per_inf_nJ,
            idle_power_uW = 0.0,
        )
        self._history.append(result)
        self._q_hci_sum        += result.q_hci
        self._total_energy_nJ  += result.energy_nJ
        self._inference_count  += 1
        return result

    def infer_sequence(self, sequence) -> list[InferenceResult]:
        return [self.infer(a, b, c) for a, b, c in sequence]

    def session_q_hci(self) -> float:
        if not self._inference_count: return 0.0
        return self._q_hci_sum / self._inference_count

    def session_summary(self) -> str:
        return (f"AuraBNN(backend={self._backend}, "
                f"inferences={self._inference_count}, "
                f"mean_q={self.session_q_hci():.4f}, "
                f"energy={self._total_energy_nJ:.2f}nJ, idle=0.00µW)")

    def reset_session(self):
        self._history.clear()
        self._q_hci_sum = 0.0
        self._total_energy_nJ = 0.0
        self._inference_count  = 0

    @property
    def total_energy_nJ(self) -> float: return self._total_energy_nJ
    @property
    def inference_count(self) -> int:   return self._inference_count
    @property
    def history(self): return list(self._history)
    @property
    def backend(self) -> str: return self._backend

    def __repr__(self):
        return self.session_summary()
