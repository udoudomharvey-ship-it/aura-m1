"""
aura_sdk.ncl.circuits
─────────────────────
Language: Python
Pre-built NCL circuit blocks used by the Aura-M1 pipeline.

    from aura_sdk.ncl.circuits import QualityStateEncoder, ThresholdDetector
"""
from __future__ import annotations
from .gates import DualRail, RailState


# ── Quality state encoder ──────────────────────────────────────────────────

class QualityStateEncoder:
    """
    Encodes a float Q_HCI ∈ [0,1] → one of 5 quality state DualRail signals.

    State boundaries (paper's 20th–80th percentile binning):
      State 1 : Q < 0.25   — Very high workload
      State 2 : Q < 0.45   — High workload
      State 3 : Q < 0.63   — Moderate workload
      State 4 : Q < 0.80   — Low workload
      State 5 : Q >= 0.80  — Very low workload
    """
    BOUNDARIES = [0.25, 0.45, 0.63, 0.80]
    N_STATES   = 5

    STATE_LABELS = {
        1: "Very High Workload",
        2: "High Workload",
        3: "Moderate Workload",
        4: "Low Workload",
        5: "Very Low Workload",
    }

    def __init__(self):
        self.rails = [DualRail(f"q_state_{i+1}") for i in range(self.N_STATES)]
        self._last_state: int | None = None

    def encode(self, q: float) -> int:
        """
        Encode Q value → state (1–5).
        Sets the corresponding DualRail to DATA1, all others to DATA0.
        Returns the state index (1–5).
        """
        assert 0.0 <= q <= 1.0, f"Q_HCI must be in [0,1], got {q}"
        state = self.N_STATES
        for i, boundary in enumerate(self.BOUNDARIES):
            if q < boundary:
                state = i + 1
                break
        for rail in self.rails:
            rail.set_null()
        self.rails[state - 1].set_data(True)
        self._last_state = state
        return state

    def reset(self):
        for rail in self.rails:
            rail.set_null()
        self._last_state = None

    @property
    def active_state(self) -> int | None:
        return self._last_state

    @property
    def active_rail(self) -> DualRail | None:
        if self._last_state is None:
            return None
        return self.rails[self._last_state - 1]

    def state_label(self, state: int) -> str:
        return self.STATE_LABELS.get(state, "Unknown")

    def __repr__(self):
        s = self._last_state
        return f"QualityStateEncoder(state={s}, {self.state_label(s) if s else 'NULL'})"


# ── Threshold detector ────────────────────────────────────────────────────

class ThresholdDetector:
    """
    Fires a DualRail alert when Q_HCI drops below a threshold.
    Used for real-time workload alerting.
    """

    def __init__(self, threshold: float = 0.45, label: str = "alert"):
        assert 0.0 < threshold < 1.0
        self.threshold  = threshold
        self.alert_rail = DualRail(f"{label}_rail")
        self._triggered = False

    def update(self, q: float) -> bool:
        """Update with new Q. Returns True if alert is triggered."""
        triggered = q < self.threshold
        self.alert_rail.set_data(triggered)
        self._triggered = triggered
        return triggered

    def reset(self):
        self.alert_rail.set_null()
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered

    def __repr__(self):
        return f"ThresholdDetector(threshold={self.threshold}, {'ALERT' if self._triggered else 'clear'})"
