//! aura_core::harvesting
//!
//! Ambient energy harvesting models — Rust implementation.
//!
//! Language: Rust
//! Why Rust: The harvesting control loop runs every frame (33ms).
//!           Power accumulation, threshold checking, and pulse
//!           firing must be deterministic and allocation-free.
//!           The physics models use f32 arithmetic that LLVM
//!           optimises aggressively.
//!
//! Five sources:
//!   RF          — rectenna, 1–100 µW
//!   Vibration   — piezoelectric, 10–500 µW
//!   Thermal     — body heat thermoelectric, 5–40 µW
//!   Light       — ambient photovoltaic, 10–200 µW
//!   Flow        — micro-turbine, 50–1000 µW
//!
//! Energy model
//! ─────────────
//!   One BNN inference pulse = 30.4 nJ  (8 µW × 3.8 ms)
//!   Idle power between pulses = 0.00 W (NCL NULL state)

use crate::{BNN_PULSE_UJ, IDLE_POWER_W};

// ── Harvesting status snapshot ─────────────────────────────────────────────

/// Point-in-time status of a harvester.
/// Returned by every harvester after each update tick.
#[derive(Debug, Clone, Copy)]
pub struct HarvestStatus {
    /// Current instantaneous power in µW
    pub power_uw:       f32,
    /// Accumulated energy available for firing in µJ
    pub accumulated_uj: f32,
    /// Total energy harvested since creation in µJ
    pub total_uj:       f32,
    /// Number of inference pulses fired so far
    pub pulse_count:    u64,
    /// Whether enough energy is available to fire right now
    pub can_fire:       bool,
    /// Idle power — always 0.0 W (NCL NULL state)
    pub idle_power_w:   f32,
}

// ── Harvester trait ────────────────────────────────────────────────────────

/// Common interface for all Aura-M1 energy sources.
/// Implemented by every concrete harvester type.
pub trait Harvester {
    /// Advance the harvester by `dt_ms` milliseconds.
    /// Returns the instantaneous power in µW.
    fn update(&mut self, dt_ms: f32) -> f32;

    /// Returns true if enough energy is stored for one BNN pulse.
    fn can_fire(&self) -> bool;

    /// Consume energy for one BNN inference pulse.
    /// Returns true if successful, false if insufficient energy.
    fn consume_pulse(&mut self) -> bool;

    /// Current accumulated energy in µJ.
    fn accumulated_uj(&self) -> f32;

    /// Total harvested energy in µJ since creation.
    fn total_uj(&self) -> f32;

    /// Number of inference pulses fired.
    fn pulse_count(&self) -> u64;

    /// Full status snapshot.
    fn status(&self) -> HarvestStatus;

    /// Source name for display.
    fn name(&self) -> &'static str;
}

// ── Shared energy bookkeeping ──────────────────────────────────────────────

/// Internal energy accounting state shared by all harvesters.
/// Not public — accessed only through the Harvester trait.
#[derive(Debug, Clone, Default)]
struct EnergyState {
    accumulated_uj: f32,
    total_uj:       f32,
    pulse_count:    u64,
}

impl EnergyState {
    fn deposit(&mut self, energy_uj: f32) {
        self.accumulated_uj += energy_uj;
        self.total_uj       += energy_uj;
    }

    fn can_fire(&self) -> bool {
        self.accumulated_uj >= BNN_PULSE_UJ
    }

    fn consume(&mut self) -> bool {
        if self.can_fire() {
            self.accumulated_uj -= BNN_PULSE_UJ;
            self.pulse_count    += 1;
            true
        } else {
            false
        }
    }
}

// ── RF Harvester ───────────────────────────────────────────────────────────

/// RF / Radio frequency harvesting via rectenna.
///
/// Physics model:
///   P_received_µW = 10^(dbm/10) × 1000
///   P_harvested   = P_received × η_rectenna  (η = 0.55)
///
/// Range: 1–100 µW
/// Use case: Smart tags, supply chain, 6G ambient IoT
pub struct RfHarvester {
    pub signal_dbm:    f32,
    pub frequency_ghz: f32,
    energy:            EnergyState,
}

impl RfHarvester {
    const RECTENNA_EFF: f32 = 0.55;
    const MIN_UW:       f32 = 1.0;
    const MAX_UW:       f32 = 100.0;

    pub fn new(signal_dbm: f32) -> Self {
        Self { signal_dbm, frequency_ghz: 2.4, energy: EnergyState::default() }
    }

    fn compute_power(&self) -> f32 {
        let p_received = (10.0_f32).powf(self.signal_dbm / 10.0) * 1000.0;
        (p_received * Self::RECTENNA_EFF).clamp(Self::MIN_UW, Self::MAX_UW)
    }
}

impl Harvester for RfHarvester {
    fn update(&mut self, dt_ms: f32) -> f32 {
        let p = self.compute_power();
        self.energy.deposit(p * (dt_ms / 1000.0));
        p
    }
    fn can_fire(&self)         -> bool  { self.energy.can_fire() }
    fn consume_pulse(&mut self) -> bool { self.energy.consume() }
    fn accumulated_uj(&self)   -> f32   { self.energy.accumulated_uj }
    fn total_uj(&self)         -> f32   { self.energy.total_uj }
    fn pulse_count(&self)      -> u64   { self.energy.pulse_count }
    fn name(&self)             -> &'static str { "RF/Rectenna" }
    fn status(&self) -> HarvestStatus {
        HarvestStatus {
            power_uw:       self.compute_power(),
            accumulated_uj: self.energy.accumulated_uj,
            total_uj:       self.energy.total_uj,
            pulse_count:    self.energy.pulse_count,
            can_fire:       self.energy.can_fire(),
            idle_power_w:   IDLE_POWER_W,
        }
    }
}

// ── Vibration Harvester ────────────────────────────────────────────────────

/// Piezoelectric vibration energy harvesting.
///
/// Physics model (simplified cantilever beam):
///   P = a² × f × k² × device_constant
///   k² = 0.12 (PZT coupling coefficient)
///
/// Range: 10–500 µW
/// Use case: Industrial sensors, ships, vehicles, machinery
pub struct VibrationHarvester {
    pub acceleration_g: f32,
    pub frequency_hz:   f32,
    energy:             EnergyState,
}

impl VibrationHarvester {
    const K_SQUARED:     f32 = 0.12;
    const DEVICE_CONST:  f32 = 15.0;
    const MIN_UW:        f32 = 10.0;
    const MAX_UW:        f32 = 500.0;

    pub fn new(acceleration_g: f32, frequency_hz: f32) -> Self {
        Self { acceleration_g, frequency_hz, energy: EnergyState::default() }
    }

    fn compute_power(&self) -> f32 {
        let p = self.acceleration_g.powi(2)
            * self.frequency_hz
            * Self::K_SQUARED
            * Self::DEVICE_CONST;
        p.clamp(Self::MIN_UW, Self::MAX_UW)
    }
}

impl Harvester for VibrationHarvester {
    fn update(&mut self, dt_ms: f32) -> f32 {
        let p = self.compute_power();
        self.energy.deposit(p * (dt_ms / 1000.0));
        p
    }
    fn can_fire(&self)         -> bool  { self.energy.can_fire() }
    fn consume_pulse(&mut self) -> bool { self.energy.consume() }
    fn accumulated_uj(&self)   -> f32   { self.energy.accumulated_uj }
    fn total_uj(&self)         -> f32   { self.energy.total_uj }
    fn pulse_count(&self)      -> u64   { self.energy.pulse_count }
    fn name(&self)             -> &'static str { "Piezoelectric/Vibration" }
    fn status(&self) -> HarvestStatus {
        HarvestStatus {
            power_uw:       self.compute_power(),
            accumulated_uj: self.energy.accumulated_uj,
            total_uj:       self.energy.total_uj,
            pulse_count:    self.energy.pulse_count,
            can_fire:       self.energy.can_fire(),
            idle_power_w:   IDLE_POWER_W,
        }
    }
}

// ── Thermal Harvester ──────────────────────────────────────────────────────

/// Thermoelectric body-heat harvesting (Seebeck effect).
///
/// Physics model:
///   V_open = S × ΔT          (S = 200 µV/K, BiTe module)
///   P      = V_open² × ZT    (ZT = 0.8)
///
/// Range: 5–40 µW
/// Use case: Pacemakers, neural implants, wearables
pub struct ThermalHarvester {
    pub delta_t_celsius: f32,
    energy:              EnergyState,
}

impl ThermalHarvester {
    const SEEBECK_UV_K: f32 = 200.0;
    const ZT:           f32 = 0.8;
    const MIN_UW:       f32 = 5.0;
    const MAX_UW:       f32 = 40.0;

    pub fn new(delta_t_celsius: f32) -> Self {
        Self { delta_t_celsius, energy: EnergyState::default() }
    }

    fn compute_power(&self) -> f32 {
        let v_open = Self::SEEBECK_UV_K * self.delta_t_celsius;
        let p      = (v_open * v_open) * Self::ZT / 1e6;
        p.clamp(Self::MIN_UW, Self::MAX_UW)
    }
}

impl Harvester for ThermalHarvester {
    fn update(&mut self, dt_ms: f32) -> f32 {
        let p = self.compute_power();
        self.energy.deposit(p * (dt_ms / 1000.0));
        p
    }
    fn can_fire(&self)         -> bool  { self.energy.can_fire() }
    fn consume_pulse(&mut self) -> bool { self.energy.consume() }
    fn accumulated_uj(&self)   -> f32   { self.energy.accumulated_uj }
    fn total_uj(&self)         -> f32   { self.energy.total_uj }
    fn pulse_count(&self)      -> u64   { self.energy.pulse_count }
    fn name(&self)             -> &'static str { "Thermoelectric/Body-Heat" }
    fn status(&self) -> HarvestStatus {
        HarvestStatus {
            power_uw:       self.compute_power(),
            accumulated_uj: self.energy.accumulated_uj,
            total_uj:       self.energy.total_uj,
            pulse_count:    self.energy.pulse_count,
            can_fire:       self.energy.can_fire(),
            idle_power_w:   IDLE_POWER_W,
        }
    }
}

// ── Light Harvester ────────────────────────────────────────────────────────

/// Micro-photovoltaic ambient light harvesting.
///
/// Physics model:
///   P = lux × 0.00147 W/m² × area_m² × η
///   η = 0.10 (amorphous silicon indoor efficiency)
///
/// Range: 10–200 µW
/// Use case: Buildings, smart packaging, indoor wearables
pub struct LightHarvester {
    pub illuminance_lux: f32,
    pub cell_area_cm2:   f32,
    energy:              EnergyState,
}

impl LightHarvester {
    const LUX_TO_WM2:  f32 = 0.00147;
    const EFFICIENCY:  f32 = 0.10;
    const MIN_UW:      f32 = 10.0;
    const MAX_UW:      f32 = 200.0;

    pub fn new(illuminance_lux: f32, cell_area_cm2: f32) -> Self {
        Self { illuminance_lux, cell_area_cm2, energy: EnergyState::default() }
    }

    fn compute_power(&self) -> f32 {
        let area_m2 = self.cell_area_cm2 * 1e-4;
        let irr     = self.illuminance_lux * Self::LUX_TO_WM2;
        let p_w     = irr * area_m2 * Self::EFFICIENCY;
        let p_uw    = p_w * 1e6;
        p_uw.clamp(Self::MIN_UW, Self::MAX_UW)
    }
}

impl Harvester for LightHarvester {
    fn update(&mut self, dt_ms: f32) -> f32 {
        let p = self.compute_power();
        self.energy.deposit(p * (dt_ms / 1000.0));
        p
    }
    fn can_fire(&self)         -> bool  { self.energy.can_fire() }
    fn consume_pulse(&mut self) -> bool { self.energy.consume() }
    fn accumulated_uj(&self)   -> f32   { self.energy.accumulated_uj }
    fn total_uj(&self)         -> f32   { self.energy.total_uj }
    fn pulse_count(&self)      -> u64   { self.energy.pulse_count }
    fn name(&self)             -> &'static str { "Photovoltaic/Ambient-Light" }
    fn status(&self) -> HarvestStatus {
        HarvestStatus {
            power_uw:       self.compute_power(),
            accumulated_uj: self.energy.accumulated_uj,
            total_uj:       self.energy.total_uj,
            pulse_count:    self.energy.pulse_count,
            can_fire:       self.energy.can_fire(),
            idle_power_w:   IDLE_POWER_W,
        }
    }
}

// ── Flow Harvester ─────────────────────────────────────────────────────────

/// Fluid flow micro-turbine harvesting.
///
/// Physics model:
///   P = (Q_m3s) × (pressure_Pa) × η_turbine
///   η = 0.25
///
/// Range: 50–1000 µW
/// Use case: Pipelines, water systems, gas infrastructure
pub struct FlowHarvester {
    pub flow_rate_lpm: f32,
    pub pressure_kpa:  f32,
    energy:            EnergyState,
}

impl FlowHarvester {
    const TURBINE_EFF: f32 = 0.25;
    const MIN_UW:      f32 = 50.0;
    const MAX_UW:      f32 = 1000.0;

    pub fn new(flow_rate_lpm: f32, pressure_kpa: f32) -> Self {
        Self { flow_rate_lpm, pressure_kpa, energy: EnergyState::default() }
    }

    fn compute_power(&self) -> f32 {
        let q_m3s   = self.flow_rate_lpm / 60_000.0;
        let p_pa    = self.pressure_kpa * 1000.0;
        let p_w     = q_m3s * p_pa * Self::TURBINE_EFF;
        let p_uw    = p_w * 1e6;
        p_uw.clamp(Self::MIN_UW, Self::MAX_UW)
    }
}

impl Harvester for FlowHarvester {
    fn update(&mut self, dt_ms: f32) -> f32 {
        let p = self.compute_power();
        self.energy.deposit(p * (dt_ms / 1000.0));
        p
    }
    fn can_fire(&self)         -> bool  { self.energy.can_fire() }
    fn consume_pulse(&mut self) -> bool { self.energy.consume() }
    fn accumulated_uj(&self)   -> f32   { self.energy.accumulated_uj }
    fn total_uj(&self)         -> f32   { self.energy.total_uj }
    fn pulse_count(&self)      -> u64   { self.energy.pulse_count }
    fn name(&self)             -> &'static str { "Micro-Turbine/Flow" }
    fn status(&self) -> HarvestStatus {
        HarvestStatus {
            power_uw:       self.compute_power(),
            accumulated_uj: self.energy.accumulated_uj,
            total_uj:       self.energy.total_uj,
            pulse_count:    self.energy.pulse_count,
            can_fire:       self.energy.can_fire(),
            idle_power_w:   IDLE_POWER_W,
        }
    }
}

// ── Harvesting session runner ──────────────────────────────────────────────

/// Simulation result for one harvesting session.
#[derive(Debug, Clone)]
pub struct SessionResult {
    pub source:              &'static str,
    pub duration_s:          f32,
    pub mean_power_uw:       f32,
    pub total_energy_uj:     f32,
    pub pulses_fired:        u64,
    pub inferences_per_sec:  f32,
    pub idle_power_w:        f32,
    pub energy_per_pulse_nj: f32,
}

/// Run a harvesting simulation for `duration_s` seconds
/// with timestep `dt_ms` milliseconds.
pub fn run_session<H: Harvester>(
    harvester: &mut H,
    duration_s: f32,
    dt_ms: f32,
) -> SessionResult {
    let steps = (duration_s * 1000.0 / dt_ms) as usize;
    let mut power_sum = 0.0_f32;

    for _ in 0..steps {
        let p = harvester.update(dt_ms);
        power_sum += p;
        if harvester.can_fire() {
            harvester.consume_pulse();
        }
    }

    let pulses     = harvester.pulse_count();
    let mean_power = if steps > 0 { power_sum / steps as f32 } else { 0.0 };
    let inf_per_s  = if duration_s > 0.0 { pulses as f32 / duration_s } else { 0.0 };

    SessionResult {
        source:              harvester.name(),
        duration_s,
        mean_power_uw:       mean_power,
        total_energy_uj:     harvester.total_uj(),
        pulses_fired:        pulses,
        inferences_per_sec:  inf_per_s,
        idle_power_w:        IDLE_POWER_W,
        energy_per_pulse_nj: crate::BNN_PULSE_NJ,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rf_power_in_range() {
        let h = RfHarvester::new(-20.0);
        let p = h.compute_power();
        assert!(p >= RfHarvester::MIN_UW);
        assert!(p <= RfHarvester::MAX_UW);
    }

    #[test]
    fn vibration_power_in_range() {
        let h = VibrationHarvester::new(1.0, 60.0);
        let p = h.compute_power();
        assert!(p >= VibrationHarvester::MIN_UW);
        assert!(p <= VibrationHarvester::MAX_UW);
    }

    #[test]
    fn thermal_power_in_range() {
        let h = ThermalHarvester::new(3.0);
        let p = h.compute_power();
        assert!(p >= ThermalHarvester::MIN_UW);
        assert!(p <= ThermalHarvester::MAX_UW);
    }

    #[test]
    fn light_power_in_range() {
        let h = LightHarvester::new(500.0, 1.0);
        let p = h.compute_power();
        assert!(p >= LightHarvester::MIN_UW);
        assert!(p <= LightHarvester::MAX_UW);
    }

    #[test]
    fn flow_power_in_range() {
        let h = FlowHarvester::new(5.0, 100.0);
        let p = h.compute_power();
        assert!(p >= FlowHarvester::MIN_UW);
        assert!(p <= FlowHarvester::MAX_UW);
    }

    #[test]
    fn idle_power_always_zero() {
        let h = ThermalHarvester::new(3.0);
        assert_eq!(h.status().idle_power_w, 0.0);
    }

    #[test]
    fn energy_accumulates_over_time() {
        let mut h = VibrationHarvester::new(2.0, 60.0);
        for _ in 0..100 { h.update(10.0); }
        assert!(h.total_uj() > 0.0);
    }

    #[test]
    fn pulse_fires_when_threshold_met() {
        let mut h = VibrationHarvester::new(5.0, 120.0);
        let mut fired = false;
        for _ in 0..10_000 {
            h.update(10.0);
            if h.can_fire() {
                h.consume_pulse();
                fired = true;
                break;
            }
        }
        assert!(fired, "Should fire at least one pulse");
        assert!(h.pulse_count() >= 1);
    }

    #[test]
    fn higher_delta_t_more_power() {
        // Use values clearly above the MIN_UW floor (5.0 µW)
        // ΔT=4 → V=800µV → P=(800²×0.8)/1e6 = 0.512 µW → clamp to 5.0
        // ΔT=10 → V=2000µV → P=(2000²×0.8)/1e6 = 3.2 µW → clamp to 5.0
        // ΔT=16 → V=3200µV → P=(3200²×0.8)/1e6 = 8.19 µW → above floor
        // ΔT=20 → V=4000µV → P=(4000²×0.8)/1e6 = 12.8 µW → above floor
        let h1 = ThermalHarvester::new(16.0);
        let h2 = ThermalHarvester::new(20.0);
        assert!(h2.compute_power() > h1.compute_power(),
            "h2({}) should exceed h1({})", h2.compute_power(), h1.compute_power());
    }

    #[test]
    fn session_result_has_zero_idle_power() {
        let mut h = ThermalHarvester::new(4.0);
        let r = run_session(&mut h, 5.0, 10.0);
        assert_eq!(r.idle_power_w, 0.0);
        assert!(r.pulses_fired > 0);
    }
}
