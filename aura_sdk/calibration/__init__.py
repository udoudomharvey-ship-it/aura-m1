"""
aura_sdk.calibration
─────────────────────
Language: Python
User-specific quality state boundary calibration.

    from aura_sdk.calibration import UserCalibrator, CalibrationProfile
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field


@dataclass
class CalibrationProfile:
    """
    Personalised Q_HCI state boundaries for one user.
    Replaces the population-average boundaries from the paper.
    """
    user_id:    str
    boundaries: list[float] = field(
        default_factory=lambda: [0.25, 0.45, 0.63, 0.80]
    )
    calibrated: bool = False

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"user_id": self.user_id,
                       "boundaries": self.boundaries,
                       "calibrated": self.calibrated}, f)

    @classmethod
    def load(cls, path: str) -> "CalibrationProfile":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def encode_state(self, q: float) -> int:
        for i, b in enumerate(self.boundaries):
            if q < b:
                return i + 1
        return 5


class UserCalibrator:
    """
    Collects entropy features during a short calibration session
    and fits personalised quality state boundaries.

    Usage
    ─────
        cal = UserCalibrator(user_id="subject_01")
        with SyntheticSource(n_frames=60) as src:
            for frame in src.frames():
                cal.add_frame(frame)
        profile = cal.fit()
        profile.save("profiles/subject_01.json")
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self._q_samples: list[float] = []

    def add_q_sample(self, q: float) -> None:
        self._q_samples.append(q)

    def fit(self) -> CalibrationProfile:
        """
        Fit boundaries from collected Q samples using percentile binning.
        Falls back to paper defaults if fewer than 10 samples collected.
        """
        if len(self._q_samples) < 10:
            return CalibrationProfile(user_id=self.user_id)

        s = sorted(self._q_samples)
        n = len(s)
        boundaries = [
            s[int(n * 0.20)],
            s[int(n * 0.40)],
            s[int(n * 0.60)],
            s[int(n * 0.80)],
        ]
        return CalibrationProfile(
            user_id    = self.user_id,
            boundaries = boundaries,
            calibrated = True,
        )

    def reset(self) -> None:
        self._q_samples.clear()


__all__ = ["CalibrationProfile", "UserCalibrator"]
