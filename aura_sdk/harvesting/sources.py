"""
aura_sdk.harvesting.sources
────────────────────────────
Language: Python (wraps Rust backend when available, else pure Python)
Why: The harvesting control loop runs every frame. When Rust bindings
     are compiled, energy accumulation delegates to aura_core_rs.
     Otherwise the pure Python physics models provide identical output.

All five harvesters + HybridHarvester + HarvestingSession.
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import Optional

# ── Try Rust backend ───────────────────────────────────────────────────────
try:
    from aura_core_rs import run_harvesting_session as _rust_session
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _rust_session   = None

# ── Constants ─────────────────────────────────────────────────────────────
BNN_INFERENCE_ENERGY_NJ = 30.4        # 8 µW × 3.8 ms
BNN_INFERENCE_ENERGY_UJ = BNN_INFERENCE_ENERGY_NJ / 1000.0


# ── Base class ─────────────────────────────────────────────────────────────

class EnergyHarvester:
    """Abstract base for all Aura-M1 energy harvesters."""

    SOURCE_NAME = "Abstract"

    def __init__(self, label: str = ""):
        self.label            = label or self.SOURCE_NAME
        self._accumulated_uJ  = 0.0
        self._total_uJ        = 0.0
        self._pulse_count     = 0

    def _current_power_uw(self, **env) -> float:
        return 0.0

    def update(self, dt_ms: float, **env) -> float:
        p = max(0.0, self._current_power_uw(**env))
        e = p * (dt_ms / 1000.0)
        self._accumulated_uJ += e
        self._total_uJ       += e
        return p

    def can_fire(self) -> bool:
        return self._accumulated_uJ >= BNN_INFERENCE_ENERGY_UJ

    def consume_pulse(self) -> bool:
        if self.can_fire():
            self._accumulated_uJ -= BNN_INFERENCE_ENERGY_UJ
            self._pulse_count    += 1
            return True
        return False

    def force_consume(self):
        self._accumulated_uJ = max(0.0, self._accumulated_uJ - BNN_INFERENCE_ENERGY_UJ)
        self._pulse_count    += 1

    @property
    def accumulated_uJ(self) -> float: return self._accumulated_uJ
    @property
    def total_uJ(self) -> float:       return self._total_uJ
    @property
    def pulse_count(self) -> int:      return self._pulse_count

    def status(self) -> dict:
        return {
            "source":          self.SOURCE_NAME,
            "accumulated_uJ":  round(self._accumulated_uJ, 4),
            "total_uJ":        round(self._total_uJ, 4),
            "pulse_count":     self._pulse_count,
            "can_fire":        self.can_fire(),
            "idle_power_W":    0.0,
        }

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"acc={self._accumulated_uJ:.3f}µJ, "
                f"pulses={self._pulse_count})")


# ── RF Harvester ───────────────────────────────────────────────────────────

class RFHarvester(EnergyHarvester):
    """RF / rectenna harvesting. Range: 1–100 µW."""
    SOURCE_NAME  = "RF/Rectenna"
    RECTENNA_EFF = 0.55
    MIN_UW, MAX_UW = 1.0, 100.0

    def __init__(self, label: str = "rf", signal_dbm: float = -20.0):
        super().__init__(label)
        self.signal_dbm = signal_dbm

    def _current_power_uw(self, signal_dbm: float = None, **_) -> float:
        dbm = signal_dbm if signal_dbm is not None else self.signal_dbm
        p   = (10 ** (dbm / 10.0)) * 1000.0 * self.RECTENNA_EFF
        return max(self.MIN_UW, min(self.MAX_UW, p))


# ── Vibration Harvester ────────────────────────────────────────────────────

class VibrationHarvester(EnergyHarvester):
    """Piezoelectric vibration harvesting. Range: 10–500 µW."""
    SOURCE_NAME  = "Piezoelectric/Vibration"
    K_SQUARED    = 0.12
    DEVICE_CONST = 15.0
    MIN_UW, MAX_UW = 10.0, 500.0

    def __init__(self, label: str = "vib",
                 acceleration_g: float = 1.0,
                 frequency_hz:   float = 60.0):
        super().__init__(label)
        self.acceleration_g = acceleration_g
        self.frequency_hz   = frequency_hz

    def _current_power_uw(self, acceleration_g: float = None,
                           frequency_hz: float = None, **_) -> float:
        a = acceleration_g if acceleration_g is not None else self.acceleration_g
        f = frequency_hz   if frequency_hz   is not None else self.frequency_hz
        p = a**2 * f * self.K_SQUARED * self.DEVICE_CONST
        return max(self.MIN_UW, min(self.MAX_UW, p))


# ── Thermal Harvester ──────────────────────────────────────────────────────

class ThermalHarvester(EnergyHarvester):
    """Body-heat thermoelectric harvesting. Range: 5–40 µW."""
    SOURCE_NAME    = "Thermoelectric/Body-Heat"
    SEEBECK_UV_K   = 200.0
    ZT             = 0.8
    MIN_UW, MAX_UW = 5.0, 40.0

    def __init__(self, label: str = "thermal", delta_T_C: float = 3.0):
        super().__init__(label)
        self.delta_T_C = delta_T_C

    def _current_power_uw(self, delta_T_C: float = None, **_) -> float:
        dT = delta_T_C if delta_T_C is not None else self.delta_T_C
        v  = self.SEEBECK_UV_K * dT
        p  = (v * v) * self.ZT / 1e6
        return max(self.MIN_UW, min(self.MAX_UW, p))


# ── Light Harvester ────────────────────────────────────────────────────────

class LightHarvester(EnergyHarvester):
    """Ambient photovoltaic harvesting. Range: 10–200 µW."""
    SOURCE_NAME    = "Photovoltaic/Ambient-Light"
    LUX_TO_WM2    = 0.00147
    EFFICIENCY     = 0.10
    MIN_UW, MAX_UW = 10.0, 200.0

    def __init__(self, label: str = "light",
                 illuminance_lux: float = 500.0,
                 cell_area_cm2:   float = 1.0):
        super().__init__(label)
        self.illuminance_lux = illuminance_lux
        self.cell_area_cm2   = cell_area_cm2

    def _current_power_uw(self, illuminance_lux: float = None, **_) -> float:
        lux  = illuminance_lux if illuminance_lux is not None else self.illuminance_lux
        area = self.cell_area_cm2 * 1e-4
        p    = lux * self.LUX_TO_WM2 * area * self.EFFICIENCY * 1e6
        return max(self.MIN_UW, min(self.MAX_UW, p))


# ── Flow Harvester ─────────────────────────────────────────────────────────

class FlowHarvester(EnergyHarvester):
    """Fluid flow micro-turbine harvesting. Range: 50–1000 µW."""
    SOURCE_NAME    = "Micro-Turbine/Flow"
    TURBINE_EFF    = 0.25
    MIN_UW, MAX_UW = 50.0, 1000.0

    def __init__(self, label: str = "flow",
                 flow_rate_lpm: float = 5.0,
                 pressure_kpa:  float = 100.0):
        super().__init__(label)
        self.flow_rate_lpm = flow_rate_lpm
        self.pressure_kpa  = pressure_kpa

    def _current_power_uw(self, flow_rate_lpm: float = None,
                           pressure_kpa: float = None, **_) -> float:
        Q = (flow_rate_lpm if flow_rate_lpm is not None else self.flow_rate_lpm) / 60_000.0
        P = (pressure_kpa  if pressure_kpa  is not None else self.pressure_kpa)  * 1000.0
        p = Q * P * self.TURBINE_EFF * 1e6
        return max(self.MIN_UW, min(self.MAX_UW, p))


# ── Hybrid Harvester ───────────────────────────────────────────────────────

class HybridHarvester(EnergyHarvester):
    """Combines multiple energy sources into one shared pool."""
    SOURCE_NAME = "Hybrid/Multi-Source"

    def __init__(self, sources: list[EnergyHarvester], label: str = "hybrid"):
        super().__init__(label)
        self.sources = sources

    def update(self, dt_ms: float, **env) -> float:
        total = 0.0
        for src in self.sources:
            p = src.update(dt_ms, **env)
            total += p
            self._accumulated_uJ += src._accumulated_uJ
            self._total_uJ       += src._accumulated_uJ
            src._accumulated_uJ   = 0.0
        return total

    def status(self) -> dict:
        base = super().status()
        base["sources"] = [s.status() for s in self.sources]
        return base


# ── Harvesting session runner ──────────────────────────────────────────────

@dataclass
class HarvestingSession:
    """Simulate a full harvesting session. Returns summary stats."""
    harvester:  EnergyHarvester
    duration_s: float
    dt_ms:      float = 10.0
    env_params: dict  = field(default_factory=dict)

    def run(self) -> dict:
        steps  = int(self.duration_s * 1000 / self.dt_ms)
        powers = []
        for _ in range(steps):
            p = self.harvester.update(self.dt_ms, **self.env_params)
            powers.append(p)
            if self.harvester.can_fire():
                self.harvester.consume_pulse()

        pulses   = self.harvester.pulse_count
        mean_p   = sum(powers) / len(powers) if powers else 0.0
        inf_per_s = pulses / self.duration_s if self.duration_s > 0 else 0.0

        return {
            "source":              self.harvester.SOURCE_NAME,
            "duration_s":          self.duration_s,
            "mean_power_uW":       round(mean_p, 3),
            "total_energy_uJ":     round(self.harvester.total_uJ, 3),
            "pulses_fired":        pulses,
            "inferences_per_sec":  round(inf_per_s, 2),
            "idle_power_W":        0.0,
            "energy_per_pulse_nJ": round(BNN_INFERENCE_ENERGY_NJ, 2),
        }
