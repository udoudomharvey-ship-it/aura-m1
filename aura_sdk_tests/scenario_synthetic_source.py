"""
aura_sdk_tests/scenario_synthetic_source.py
────────────────────────────────────────────
A test-only SyntheticSource variant that can be aimed at a SPECIFIC
ProblemClass, unlike aura_sdk.sensors.sources.SyntheticSource (which
only ramps one scalar `_q` and always classifies as COGNITIVE_OVERLOAD
in practice -- see the investigation that led here).

IMPORTANT -- only 3 of the 4 ProblemClasses are reachable:

    ProblemClass.SPATIAL_RESTLESSNESS   -- reachable, calibrated below
    ProblemClass.COGNITIVE_OVERLOAD     -- reachable, calibrated below
    ProblemClass.SUSTAINED_LOW_QUALITY  -- reachable, calibrated below
    ProblemClass.TEMPORAL_ERRATICNESS   -- NOT REACHABLE, any input

TEMPORAL_ERRATICNESS requires H_te to be the single dominant term, but
H_te (marginal entropy of ONE joint's own velocity histogram) can
mathematically never exceed H_in (joint entropy over a PAIR of joints'
velocity histograms) for ANY motion pattern -- joint entropy is always
>= either marginal's entropy: H(X,Y) >= H(X), a basic information-
theory fact (a marginal is a deterministic projection of the joint
distribution, and entropy cannot decrease under such coarsening).
Confirmed here by exhaustive sweep across activity levels (1..25
active joints, many seeds): H_te never exceeded H_in, not once; the
closest case (all joints moving in perfect lockstep) landed on an
EXACT tie decided by floating-point rounding noise (~1e-15), not
signal. This generator does not attempt that class. Worth fixing at
the source (rules.py's dominance check and/or how H_in is scaled/
normalized relative to H_te) -- flagged separately, out of scope here.

Also worth noting for anyone extending this file: H_sp's sector index
uses atan2(z, x), which is numerically unstable near the origin (tiny
sign flips scatter across all 12 sectors). Since HEMMExtractor requires
spine-normalised joints (joint 0 at exact origin), any OTHER joint you
want classified as "in the same tight cluster" must sit at a stable
nonzero bias well away from the origin singularity -- NOT literally
near (0,0,0) -- or its sector assignment becomes noise-dominated. All
three generators below account for this.

Calibrated against the REAL aura_sdk.hemm.extractor.HEMMExtractor and
RecommendationEngine._classify()'s dominance logic (avg of last 5
frames' features, 0.45 dominance share), WITH normalization applied
(HEMMExtractor's documented precondition) -- verified across 5 seeds
each, well clear of the threshold, not a knife-edge tuning:

    spatial_restlessness:  h_sp share = 1.00   (h_te, h_in exactly 0)
    cognitive_overload:    h_in share = 0.63-0.65
    sustained_low_quality: no term clears 0.45 (falls back correctly)

Usage
─────
    from scenario_synthetic_source import ScenarioSyntheticSource
    from aura_sdk.recommend import ProblemClass

    source = ScenarioSyntheticSource(
        phases=[
            (ProblemClass.SPATIAL_RESTLESSNESS,  40),
            (ProblemClass.COGNITIVE_OVERLOAD,    40),
            (ProblemClass.SUSTAINED_LOW_QUALITY, 40),
        ],
        seed=0,
    )
    for frame in source.frames():
        ...  # feed into your real pipeline exactly like SyntheticSource
"""
from __future__ import annotations
import math
import random
from typing import List, Optional, Tuple

from aura_sdk.recommend import ProblemClass
from aura_sdk.sensors.sources import SkeletonFrame, SkeletonSource

N_JOINTS = 25

REACHABLE_CLASSES = frozenset({
    ProblemClass.SPATIAL_RESTLESSNESS,
    ProblemClass.COGNITIVE_OVERLOAD,
    ProblemClass.SUSTAINED_LOW_QUALITY,
})


def _normalise(joints, spine_idx: int = 0):
    sx, sy, sz = joints[spine_idx]
    return [(x - sx, y - sy, z - sz) for x, y, z in joints]


def _wide_layout(spread: float = 1.0):
    """25 joints spread across many spherical sectors (wide azimuth/
    elevation coverage) -- drives H_sp up once normalised."""
    layout = []
    for j in range(N_JOINTS):
        angle = (j / N_JOINTS) * 2 * math.pi
        elev  = -1.0 + 2.0 * (j / N_JOINTS)
        layout.append((spread*math.cos(angle), elev*spread, spread*math.sin(angle)))
    return layout


def _gen_spatial_restlessness(n_frames: int, rng: random.Random) -> List[List[Tuple]]:
    """Wide static layout + a tiny SHARED (correlated) jitter each frame.
    Moving together doesn't change the relative sector layout, so H_sp
    stays high while H_te/H_in stay at exactly 0 (fully correlated
    motion cancels out after normalisation)."""
    layout = _wide_layout()
    frames = []
    for _ in range(n_frames):
        dx, dy, dz = (rng.uniform(-0.01, 0.01) for _ in range(3))
        raw = [(x+dx, y+dy, z+dz) for x, y, z in layout]
        frames.append(_normalise(raw))
    return frames


def _gen_cognitive_overload(n_frames: int, rng: random.Random,
                             amp: float = 0.02,
                             bias: Tuple[float, float, float] = (0.4, 0.3, 0.1)
                             ) -> List[List[Tuple]]:
    """Authored DIRECTLY in normalised (joint-0-at-origin) space to avoid
    the atan2 instability near the origin. Joint 0 fixed exactly at
    origin (the normalisation reference). Joints 1..24 held at a shared
    nonzero bias -- far from the origin singularity, so their sector
    assignment is stable -- plus a small per-joint decorrelated
    oscillation (amp << bias magnitude, so it never crosses a sector
    boundary). Decorrelated velocities across joints drive H_in up,
    which (per the H_in>=H_te proof above) automatically beats H_te too."""
    bx, by, bz = bias
    freqs  = [rng.uniform(1.0, 4.0) for _ in range(N_JOINTS)]
    phases = [rng.uniform(0, 2*math.pi) for _ in range(N_JOINTS)]
    frames = []
    for t in range(n_frames):
        frame = [(0.0, 0.0, 0.0)]  # joint 0: fixed, IS the normalisation origin
        for j in range(1, N_JOINTS):
            x = bx + 0.001*j + amp*math.sin(freqs[j]*t + phases[j])
            y = by + 0.001*j + amp*math.cos(freqs[j]*0.6*t + phases[j])
            z = bz + 0.001*j + amp*math.sin(freqs[j]*1.4*t + phases[j]*0.5)
            frame.append((x, y, z))
        frames.append(frame)  # already normalised by construction
    return frames


def _gen_sustained_low_quality(n_frames: int, rng: random.Random,
                                spread: float = 0.4, mag_scale: float = 0.15
                                ) -> List[List[Tuple]]:
    """Moderate spread + moderate INDEPENDENT per-joint jitter -- no
    single term clears the 0.45 dominance share, landing in the
    catch-all class exactly as rules.py intends for ambiguous cases."""
    layout = _wide_layout(spread=spread)
    frames = [_normalise(layout)]
    cur = list(layout)
    for _ in range(n_frames - 1):
        nxt = [(x+rng.uniform(-mag_scale, mag_scale),
                y+rng.uniform(-mag_scale, mag_scale),
                z+rng.uniform(-mag_scale, mag_scale)) for x, y, z in cur]
        cur = nxt
        frames.append(_normalise(cur))
    return frames


_GENERATORS = {
    ProblemClass.SPATIAL_RESTLESSNESS:  _gen_spatial_restlessness,
    ProblemClass.COGNITIVE_OVERLOAD:    _gen_cognitive_overload,
    ProblemClass.SUSTAINED_LOW_QUALITY: _gen_sustained_low_quality,
}


class ScenarioSyntheticSource(SkeletonSource):
    """
    Deterministic synthetic skeleton source that can target specific
    ProblemClasses, phase by phase -- for exercising the recommend /
    personalize / escalate layers against more than one category, which
    aura_sdk.sensors.sources.SyntheticSource cannot do (see module
    docstring).

    `phases`: list of (ProblemClass, n_frames) tuples, played in order.
    Only SPATIAL_RESTLESSNESS, COGNITIVE_OVERLOAD, and
    SUSTAINED_LOW_QUALITY are accepted -- TEMPORAL_ERRATICNESS raises
    ValueError immediately, since no input can reach it (see module
    docstring for why).
    """

    def __init__(self, phases: List[Tuple[ProblemClass, int]], seed: int = 0):
        for pc, _ in phases:
            if pc not in REACHABLE_CLASSES:
                raise ValueError(
                    f"{pc} is not reachable via HEMM entropy features -- "
                    "see ScenarioSyntheticSource's module docstring. Only "
                    f"{sorted(c.value for c in REACHABLE_CLASSES)} are supported."
                )
        self.phases = phases
        self._seed = seed
        self._rng = random.Random(seed)
        self._all_frames: List[List[Tuple]] = []
        for pc, n_frames in phases:
            self._all_frames.extend(_GENERATORS[pc](n_frames, self._rng))
        self.n_frames = len(self._all_frames)
        self._frame = 0
        self._open = False

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def is_open(self) -> bool:
        return self._open and self._frame < self.n_frames

    def read_frame(self) -> Optional[SkeletonFrame]:
        if not self.is_open():
            return None
        joints = self._all_frames[self._frame]
        frame = SkeletonFrame(joints=joints, frame_idx=self._frame, source="scenario_synthetic")
        self._frame += 1
        return frame.normalised()  # no-op given the generators already normalise, kept for parity with SyntheticSource

    def __repr__(self):
        return f"ScenarioSyntheticSource(n_frames={self.n_frames}, seed={self._seed})"
