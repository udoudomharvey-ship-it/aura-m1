"""
aura_sdk/hemm/extractor.py
───────────────────────────
Language: Python (thin wrapper)
Why: The Python layer is the interface. The Rust core does
     the actual computation. This module detects whether
     aura_core_rs is available and routes accordingly.

     If Rust bindings are compiled  → calls aura_core_rs.RustEntropy
     If not yet compiled            → falls back to pure-Python reference

The API is identical in both cases. Callers never need to know
which implementation is running.
"""

from __future__ import annotations
import math
from typing import NamedTuple

# ── Try to import Rust backend ─────────────────────────────────────────────
try:
    from aura_core_rs import RustEntropy as _RustEntropy
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _RustEntropy = None


# ── Data types ─────────────────────────────────────────────────────────────

class EntropyFeatures(NamedTuple):
    """
    HEMM feature vector φ_t = [H_sp, H_te, H_in].
    Immutable — created once per frame, passed to BNN.
    """
    h_sp: float   # Spatial entropy     (HOJ3D K=12)
    h_te: float   # Temporal entropy    (velocity histograms)
    h_in: float   # Inter-joint entropy (co-movement histograms)

    def as_list(self) -> list[float]:
        return [self.h_sp, self.h_te, self.h_in]

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.h_sp, self.h_te, self.h_in)

    def __repr__(self):
        return (f"EntropyFeatures("
                f"H_sp={self.h_sp:.4f}, "
                f"H_te={self.h_te:.4f}, "
                f"H_in={self.h_in:.4f})")


# ── Pure-Python reference implementation ──────────────────────────────────
# Used when Rust bindings are not yet compiled.
# Produces identical outputs — just slower.

_EPS      = 1e-9
_K        = 12    # HOJ3D sectors
_VEL_BINS = 16    # velocity histogram bins


def _shannon(dist: list[float]) -> float:
    """Shannon entropy of a probability distribution."""
    return -sum(p * math.log2(p + _EPS) for p in dist if p > 0)


def _sector_index(x: float, y: float, z: float) -> int:
    """Assign joint to one of K=12 spherical sectors."""
    az  = (math.atan2(z, x) + math.pi) / (2.0 * math.pi)
    az_s = int(az * 8) % 8
    el_s = 0 if y < -0.5 else (1 if y < 0.5 else 2)
    return (az_s * 3 + el_s) % _K


def _hsp_py(joints: list[tuple]) -> float:
    """H_sp — spatial entropy (pure Python reference)."""
    hist = [0] * _K
    for x, y, z in joints:
        hist[_sector_index(x, y, z)] += 1
    total = len(joints)
    dist  = [c / total for c in hist]
    return _shannon(dist)


def _hte_py(prev: list[tuple], curr: list[tuple], vel_max: float = 0.5) -> float:
    """
    H_te — temporal entropy from a SINGLE (prev, curr) pair.

    ⚠ WARNING — mathematically degenerate. A histogram built from exactly
    one velocity sample per joint is always one-hot, and the Shannon
    entropy of a one-hot distribution is always ≈0.0 regardless of input.
    This was found during a full-repository audit and confirmed
    numerically: this function returns ≈0.0 for every input, always.
    Retained only for backward compatibility / direct callers who
    explicitly want the single-pair value. Real callers should use
    `HEMMExtractor.compute()`, which maintains a rolling window internally
    via `_hte_windowed_py` below.
    """
    bin_w   = vel_max / _VEL_BINS
    total_h = 0.0
    for (px, py, pz), (cx, cy, cz) in zip(prev, curr):
        vel = math.sqrt((cx-px)**2 + (cy-py)**2 + (cz-pz)**2)
        b   = min(int(vel / bin_w), _VEL_BINS - 1)
        hist = [0.0] * _VEL_BINS
        hist[b] = 1.0
        total_h += _shannon(hist)
    return total_h / max(1, len(prev))


def _hin_py(prev: list[tuple], curr: list[tuple], vel_max: float = 0.5) -> float:
    """
    H_in — inter-joint entropy from a SINGLE (prev, curr) pair.

    ⚠ WARNING — mathematically degenerate, for the same reason as
    `_hte_py` above (one-hot single-sample histogram ⇒ zero entropy).
    Retained only for backward compatibility. Use `HEMMExtractor.compute()`
    for a real value.
    """
    n     = len(prev)
    bin_w = vel_max / _VEL_BINS

    # Precompute velocity bins for all joints
    bins = []
    for (px, py, pz), (cx, cy, cz) in zip(prev, curr):
        vel = math.sqrt((cx-px)**2 + (cy-py)**2 + (cz-pz)**2)
        bins.append(min(int(vel / bin_w), _VEL_BINS - 1))

    total_h = 0.0
    pairs   = 0
    for a in range(n):
        for b in range(a + 1, n):
            hist = [0.0] * (_VEL_BINS * _VEL_BINS)
            hist[bins[a] * _VEL_BINS + bins[b]] = 1.0
            total_h += _shannon(hist)
            pairs   += 1

    return total_h / max(1, pairs)


# ── Windowed reference implementation — the actual fix ─────────────────────
#
# `_hte_py` / `_hin_py` above build a histogram from a single velocity
# sample, which is always one-hot and therefore always has zero entropy —
# confirmed numerically during the repository audit (see docstrings above).
# `HEMM_WINDOW_CAP` frames of history are accumulated instead, mirroring
# aura_core::hemm::HemmWindow exactly (same CAP=5, same 16 bins, same
# epsilon, same "partial window during warm-up" behaviour), so Python and
# Rust stay numerically comparable frame-for-frame.
HEMM_WINDOW_CAP = 5


def _hte_windowed_py(frames: list[list[tuple]], vel_max: float = 0.5) -> float:
    """H_te over 2..=HEMM_WINDOW_CAP buffered frames (chronological order)."""
    if len(frames) < 2:
        return 0.0
    bin_w = vel_max / _VEL_BINS
    n_joints = len(frames[0])
    n_pairs = len(frames) - 1
    total_h = 0.0
    for j in range(n_joints):
        hist = [0] * _VEL_BINS
        for w in range(1, len(frames)):
            px, py, pz = frames[w - 1][j]
            cx, cy, cz = frames[w][j]
            vel = math.sqrt((cx-px)**2 + (cy-py)**2 + (cz-pz)**2)
            b = min(int(vel / bin_w), _VEL_BINS - 1)
            hist[b] += 1
        dist = [c / n_pairs for c in hist]
        total_h += _shannon(dist)
    return total_h / n_joints


def _hin_windowed_py(frames: list[list[tuple]], vel_max: float = 0.5) -> float:
    """H_in over 2..=HEMM_WINDOW_CAP buffered frames (chronological order)."""
    if len(frames) < 2:
        return 0.0
    bin_w = vel_max / _VEL_BINS
    n_joints = len(frames[0])
    n_pairs = len(frames) - 1

    total_h = 0.0
    pair_count = 0
    for a in range(n_joints):
        for b in range(a + 1, n_joints):
            hist = [0] * (_VEL_BINS * _VEL_BINS)
            for w in range(1, len(frames)):
                pax, pay, paz = frames[w - 1][a]
                cax, cay, caz = frames[w][a]
                vel_a = math.sqrt((cax-pax)**2 + (cay-pay)**2 + (caz-paz)**2)
                pbx, pby, pbz = frames[w - 1][b]
                cbx, cby, cbz = frames[w][b]
                vel_b = math.sqrt((cbx-pbx)**2 + (cby-pby)**2 + (cbz-pbz)**2)
                bin_a = min(int(vel_a / bin_w), _VEL_BINS - 1)
                bin_b = min(int(vel_b / bin_w), _VEL_BINS - 1)
                hist[bin_a * _VEL_BINS + bin_b] += 1
            dist = [c / n_pairs for c in hist]
            total_h += _shannon(dist)
            pair_count += 1
    return total_h / pair_count if pair_count else 0.0


# ── Public extractor ───────────────────────────────────────────────────────

class HEMMExtractor:
    """
    HEMM entropy feature extractor.

    Automatically uses the Rust backend if aura_core_rs is available.
    Falls back to pure Python otherwise.
    The API is identical in both cases.

    STATE: maintains an internal rolling window of the last HEMM_WINDOW_CAP
    (5) frames so that H_te / H_in are computed from a real multi-sample
    distribution rather than a single (prev, curr) pair — a single-sample
    estimate is mathematically degenerate (always ≈0.0 entropy; see
    `_hte_py` / `_hin_py` docstrings for why). This means one HEMMExtractor
    instance should be used for an entire session/stream, the same way the
    Rust-accelerated `RustEntropy` binding works — not a fresh instance
    per frame.

    Usage
    ─────
        extractor = HEMMExtractor()
        features  = extractor.compute(prev_joints, curr_joints)
        print(features.h_sp, features.h_te, features.h_in)

    Joint format
    ─────────────
        List of (x, y, z) tuples in metres.
        Must be normalised (spine-base at origin) before calling.
        Length must be 25 (Kinect v2) or 20 (Kinect v1).
    """

    def __init__(self, vel_max: float = 0.5):
        self.vel_max  = vel_max
        self._backend = "rust" if _RUST_AVAILABLE else "python"
        self._rust    = _RustEntropy() if _RUST_AVAILABLE else None
        # Rolling window state for the pure-Python path. When the Rust
        # backend is active, `self._rust` (a stateful RustEntropy instance)
        # maintains its own equivalent window internally instead.
        self._window: list[list[tuple]] = []
        self._seeded = False

    @property
    def backend(self) -> str:
        """Returns 'rust' or 'python'."""
        return self._backend

    def reset(self) -> None:
        """Clear the rolling window (start a fresh session/stream)."""
        self._window = []
        self._seeded = False
        if self._backend == "rust":
            self._rust.reset()

    def compute(
        self,
        prev_joints: list[tuple[float, float, float]],
        curr_joints: list[tuple[float, float, float]],
    ) -> EntropyFeatures:
        """
        Compute all three entropy descriptors.

        H_sp comes from `curr_joints` alone (unaffected by windowing — it
        aggregates over 25 joints within one frame, not over time). H_te
        and H_in come from the rolling window of buffered frames: on the
        first call, both `prev_joints` and `curr_joints` seed the window;
        every call after that only pushes `curr_joints` (since it becomes
        `prev_joints` on the next call, and pushing it twice would
        double-count that frame).

        Parameters
        ──────────
        prev_joints : list of (x, y, z) tuples — previous frame
        curr_joints : list of (x, y, z) tuples — current frame

        Returns
        ───────
        EntropyFeatures(h_sp, h_te, h_in)
        """
        if self._backend == "rust":
            h_sp, h_te, h_in = self._rust.compute(
                prev_joints, curr_joints, self.vel_max
            )
        else:
            if not self._seeded:
                self._window.append(list(prev_joints))
                self._seeded = True
            self._window.append(list(curr_joints))
            if len(self._window) > HEMM_WINDOW_CAP:
                self._window = self._window[-HEMM_WINDOW_CAP:]

            h_sp = _hsp_py(curr_joints)
            h_te = _hte_windowed_py(self._window, self.vel_max)
            h_in = _hin_windowed_py(self._window, self.vel_max)

        return EntropyFeatures(
            h_sp=max(0.0, h_sp),
            h_te=max(0.0, h_te),
            h_in=max(0.0, h_in),
        )

    def compute_hsp(self, joints: list[tuple]) -> float:
        """Compute H_sp only."""
        if self._backend == "rust":
            return self._rust.compute_hsp(joints)
        return _hsp_py(joints)

    def compute_hte(self, prev: list[tuple], curr: list[tuple]) -> float:
        """
        Compute H_te from exactly two frames (single-pair passthrough).
        ⚠ Mathematically degenerate — see `_hte_py` docstring. Prefer
        `compute()`, which uses the rolling window.
        """
        if self._backend == "rust":
            return self._rust.compute_hte(prev, curr, self.vel_max)
        return _hte_py(prev, curr, self.vel_max)

    def compute_hin(self, prev: list[tuple], curr: list[tuple]) -> float:
        """
        Compute H_in from exactly two frames (single-pair passthrough).
        ⚠ Mathematically degenerate — see `_hin_py` docstring. Prefer
        `compute()`, which uses the rolling window.
        """
        if self._backend == "rust":
            return self._rust.compute_hin(prev, curr, self.vel_max)
        return _hin_py(prev, curr, self.vel_max)

    def __repr__(self):
        return f"HEMMExtractor(backend={self._backend}, vel_max={self.vel_max})"


# ── Normalisation ──────────────────────────────────────────────────────────

def normalise_joints(
    joints: list[tuple[float, float, float]],
    spine_idx: int = 0,
) -> list[tuple[float, float, float]]:
    """
    Normalise joint positions relative to spine-base joint.
    All positions become relative to (0,0,0).

    Parameters
    ──────────
    joints    : list of (x, y, z) tuples
    spine_idx : index of the spine-base joint (default 0)
    """
    if not joints:
        return joints
    sx, sy, sz = joints[spine_idx]
    return [(x - sx, y - sy, z - sz) for x, y, z in joints]
