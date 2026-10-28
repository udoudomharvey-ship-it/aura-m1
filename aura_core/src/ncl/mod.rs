//! aura_core::ncl
//!
//! Null Convention Logic gate primitives.
//!
//! Language: Rust
//! Why Rust: Gate evaluation runs 30× per second across an entire
//!           network. Each gate is a pure function over a 2-bit enum.
//!           Rust compiles this to a handful of CPU instructions with
//!           no heap allocation, no GC, no dynamic dispatch.
//!
//! Every signal is dual-rail encoded:
//!   NULL    (0,0) — no data, physically inert,  power = 0.00 W
//!   DATA0   (1,0) — logic LOW value present,    power = pulse only
//!   DATA1   (0,1) — logic HIGH value present,   power = pulse only
//!   INVALID (1,1) — forbidden

use std::fmt;

// ── Rail state ─────────────────────────────────────────────────────────────

/// Dual-rail encoded signal state.
/// Stored as a single byte — fits in a CPU register, zero overhead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Rail {
    Null    = 0,
    Data0   = 1,
    Data1   = 2,
    Invalid = 3,
}

impl Rail {
    /// Returns true when no data is present (chip is physically inert).
    #[inline(always)]
    pub fn is_null(self) -> bool { self == Rail::Null }

    /// Returns true when data is present (DATA0 or DATA1).
    #[inline(always)]
    pub fn is_data(self) -> bool { matches!(self, Rail::Data0 | Rail::Data1) }

    /// Returns the logical boolean value, or None if NULL/INVALID.
    #[inline(always)]
    pub fn logic_value(self) -> Option<bool> {
        match self {
            Rail::Data1 => Some(true),
            Rail::Data0 => Some(false),
            _           => None,
        }
    }

    /// Construct from a boolean value.
    #[inline(always)]
    pub fn from_bool(v: bool) -> Self {
        if v { Rail::Data1 } else { Rail::Data0 }
    }

    /// Idle power is always exactly zero.
    #[inline(always)]
    pub fn idle_power_w() -> f32 { 0.0 }
}

impl fmt::Display for Rail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rail::Null    => write!(f, "NULL"),
            Rail::Data0   => write!(f, "DATA0"),
            Rail::Data1   => write!(f, "DATA1"),
            Rail::Invalid => write!(f, "INVALID"),
        }
    }
}

// ── NCL gate functions ─────────────────────────────────────────────────────
// All gates are free functions — no allocation, no vtable, inlined.
// The compiler reduces each to a small switch/match expression.

/// TH22 — Threshold-2-of-2 (AND).
/// Fires DATA1 only when BOTH inputs carry DATA1.
/// Returns NULL when either input is NULL.
#[inline(always)]
pub fn th22(a: Rail, b: Rail) -> Rail {
    match (a, b) {
        (Rail::Null,  _          ) => Rail::Null,
        (_,           Rail::Null ) => Rail::Null,
        (Rail::Data1, Rail::Data1) => Rail::Data1,
        _                          => Rail::Data0,
    }
}

/// TH12 — Threshold-1-of-2 (OR).
/// Fires DATA1 when at least one input carries DATA1.
#[inline(always)]
pub fn th12(a: Rail, b: Rail) -> Rail {
    match (a, b) {
        (Rail::Null,  _          ) => Rail::Null,
        (_,           Rail::Null ) => Rail::Null,
        (Rail::Data1, _          ) => Rail::Data1,
        (_,           Rail::Data1) => Rail::Data1,
        _                          => Rail::Data0,
    }
}

/// TH13 — Threshold-1-of-3 (3-input OR).
#[inline(always)]
pub fn th13(a: Rail, b: Rail, c: Rail) -> Rail {
    if a.is_null() || b.is_null() || c.is_null() { return Rail::Null; }
    if a == Rail::Data1 || b == Rail::Data1 || c == Rail::Data1 {
        Rail::Data1
    } else {
        Rail::Data0
    }
}

/// TH23 — Threshold-2-of-3 (Majority gate).
/// Fires DATA1 when at least 2 of 3 inputs carry DATA1.
#[inline(always)]
pub fn th23(a: Rail, b: Rail, c: Rail) -> Rail {
    if a.is_null() || b.is_null() || c.is_null() { return Rail::Null; }
    let count = [a, b, c].iter().filter(|&&r| r == Rail::Data1).count();
    if count >= 2 { Rail::Data1 } else { Rail::Data0 }
}

/// TH33 — Threshold-3-of-3 (3-input AND).
#[inline(always)]
pub fn th33(a: Rail, b: Rail, c: Rail) -> Rail {
    match (a, b, c) {
        (Rail::Null, _, _) | (_, Rail::Null, _) | (_, _, Rail::Null) => Rail::Null,
        (Rail::Data1, Rail::Data1, Rail::Data1) => Rail::Data1,
        _ => Rail::Data0,
    }
}

/// THmn — Generic threshold-m-of-n gate.
/// Fires DATA1 when at least m of n inputs carry DATA1.
/// Uses a fixed-size stack array (no allocation).
#[inline]
pub fn thmn(inputs: &[Rail], m: usize) -> Rail {
    if inputs.iter().any(|r| r.is_null()) { return Rail::Null; }
    let count = inputs.iter().filter(|&&r| r == Rail::Data1).count();
    if count >= m { Rail::Data1 } else { Rail::Data0 }
}

/// NCL Inverter — swaps DATA0 and DATA1, passes NULL unchanged.
/// Implemented by swapping the dual-rail wires — zero additional gates.
#[inline(always)]
pub fn invert(a: Rail) -> Rail {
    match a {
        Rail::Data1 => Rail::Data0,
        Rail::Data0 => Rail::Data1,
        other       => other,
    }
}

/// XOR implemented from NCL primitives.
/// XOR(a,b) = OR( AND(a, NOT(b)), AND(NOT(a), b) )
#[inline]
pub fn xor(a: Rail, b: Rail) -> Rail {
    let na = invert(a);
    let nb = invert(b);
    let t0 = th22(a, nb);
    let t1 = th22(na, b);
    th12(t0, t1)
}

// ── NCL Register ────────────────────────────────────────────────────────────

/// Fixed-width NCL register backed by a stack array.
/// Holds state between pipeline stages.
/// Generic over WIDTH — the compiler generates a specialised
/// version for each width with no dynamic sizing.
#[derive(Clone)]
pub struct NclRegister<const WIDTH: usize> {
    rails: [Rail; WIDTH],
}

impl<const WIDTH: usize> NclRegister<WIDTH> {
    /// Create a new register in the NULL (idle) state.
    pub const fn new() -> Self {
        Self { rails: [Rail::Null; WIDTH] }
    }

    /// Write boolean values into the register.
    pub fn write(&mut self, values: &[bool; WIDTH]) {
        for (i, &v) in values.iter().enumerate() {
            self.rails[i] = Rail::from_bool(v);
        }
    }

    /// Read the current logic values (None if any rail is NULL).
    pub fn read(&self) -> Option<[bool; WIDTH]> {
        let mut out = [false; WIDTH];
        for (i, &r) in self.rails.iter().enumerate() {
            out[i] = r.logic_value()?;
        }
        Some(out)
    }

    /// Reset all rails to NULL (return to idle — 0.00 W).
    pub fn reset(&mut self) {
        self.rails = [Rail::Null; WIDTH];
    }

    /// Returns true when all rails are NULL.
    pub fn is_null(&self) -> bool {
        self.rails.iter().all(|r| r.is_null())
    }

    /// Returns true when all rails carry data.
    pub fn is_complete(&self) -> bool {
        self.rails.iter().all(|r| r.is_data())
    }
}

// ── Quality state encoder ──────────────────────────────────────────────────

/// Quality state boundaries (paper's 20th–80th percentile binning).
pub const Q_BOUNDARIES: [f32; 4] = [0.25, 0.45, 0.63, 0.80];

/// Maps a Q_HCI float value to a quality state (1–5).
/// Pure function — no allocation, deterministic.
#[inline]
pub fn encode_quality_state(q: f32) -> u8 {
    debug_assert!((0.0..=1.0).contains(&q), "Q_HCI must be in [0,1]");
    for (i, &boundary) in Q_BOUNDARIES.iter().enumerate() {
        if q < boundary { return (i + 1) as u8; }
    }
    5
}

/// Returns a dual-rail signal for the quality state output.
/// State is active (DATA1) on the corresponding state rail.
#[inline]
pub fn quality_state_rail(state: u8) -> Rail {
    if state >= 1 && state <= 5 { Rail::Data1 } else { Rail::Null }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_has_zero_power() {
        assert_eq!(Rail::idle_power_w(), 0.0);
        assert!(Rail::Null.is_null());
        assert!(!Rail::Null.is_data());
    }

    #[test]
    fn th22_truth_table() {
        assert_eq!(th22(Rail::Data0, Rail::Data0), Rail::Data0);
        assert_eq!(th22(Rail::Data0, Rail::Data1), Rail::Data0);
        assert_eq!(th22(Rail::Data1, Rail::Data0), Rail::Data0);
        assert_eq!(th22(Rail::Data1, Rail::Data1), Rail::Data1);
        assert_eq!(th22(Rail::Null,  Rail::Data1), Rail::Null);
    }

    #[test]
    fn th12_truth_table() {
        assert_eq!(th12(Rail::Data0, Rail::Data0), Rail::Data0);
        assert_eq!(th12(Rail::Data1, Rail::Data0), Rail::Data1);
        assert_eq!(th12(Rail::Data0, Rail::Data1), Rail::Data1);
        assert_eq!(th12(Rail::Data1, Rail::Data1), Rail::Data1);
        assert_eq!(th12(Rail::Null,  Rail::Data1), Rail::Null);
    }

    #[test]
    fn th23_majority() {
        assert_eq!(th23(Rail::Data0, Rail::Data0, Rail::Data1), Rail::Data0);
        assert_eq!(th23(Rail::Data1, Rail::Data1, Rail::Data0), Rail::Data1);
        assert_eq!(th23(Rail::Data1, Rail::Data1, Rail::Data1), Rail::Data1);
    }

    #[test]
    fn inverter_swaps() {
        assert_eq!(invert(Rail::Data1), Rail::Data0);
        assert_eq!(invert(Rail::Data0), Rail::Data1);
        assert_eq!(invert(Rail::Null),  Rail::Null);
    }

    #[test]
    fn xor_truth_table() {
        assert_eq!(xor(Rail::Data0, Rail::Data0), Rail::Data0);
        assert_eq!(xor(Rail::Data1, Rail::Data0), Rail::Data1);
        assert_eq!(xor(Rail::Data0, Rail::Data1), Rail::Data1);
        assert_eq!(xor(Rail::Data1, Rail::Data1), Rail::Data0);
    }

    #[test]
    fn register_write_read_reset() {
        let mut reg = NclRegister::<4>::new();
        assert!(reg.is_null());
        reg.write(&[true, false, true, true]);
        assert!(reg.is_complete());
        assert_eq!(reg.read(), Some([true, false, true, true]));
        reg.reset();
        assert!(reg.is_null());
    }

    #[test]
    fn quality_state_boundaries() {
        assert_eq!(encode_quality_state(0.10), 1);
        assert_eq!(encode_quality_state(0.24), 1);
        assert_eq!(encode_quality_state(0.25), 2);
        assert_eq!(encode_quality_state(0.44), 2);
        assert_eq!(encode_quality_state(0.45), 3);
        assert_eq!(encode_quality_state(0.62), 3);
        assert_eq!(encode_quality_state(0.63), 4);
        assert_eq!(encode_quality_state(0.79), 4);
        assert_eq!(encode_quality_state(0.80), 5);
        assert_eq!(encode_quality_state(0.99), 5);
    }

    #[test]
    fn thmn_generic() {
        let inputs = [Rail::Data1, Rail::Data1, Rail::Data0, Rail::Data0];
        assert_eq!(thmn(&inputs, 2), Rail::Data1);
        assert_eq!(thmn(&inputs, 3), Rail::Data0);
    }
}
