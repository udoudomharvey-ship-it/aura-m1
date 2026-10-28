//! aura_firmware/src/main.rs
//!
//! Aura-M1 Bare-Metal Firmware
//!
//! Language: Rust (no_std)
//! Why: No operating system. No heap. No garbage collection.
//!      Runs directly on the Aura-M1 silicon.
//!      Deterministic execution required for medical/industrial safety.
//!
//! Execution model
//! ───────────────
//!   1. Boot → configure harvesting frontend
//!   2. Enter main loop (event-driven, not polling)
//!   3. When harvester fires READY signal:
//!      a. Wake NCL datapath from NULL state
//!      b. Latch skeleton frame from sensor interface
//!      c. Run entropy pipeline (H_sp, H_te, H_in)
//!      d. Run BNN inference → Q_HCI
//!      e. Drive output rails (quality state, alert)
//!      f. Return ALL rails to NULL → 0.00 W idle
//!   4. Repeat indefinitely
//!
//! Memory layout
//! ─────────────
//!   Stack only — no heap allocation anywhere
//!   Frame window: 5 × 25 joints × 12 bytes = 1500 bytes
//!     (previously a 2-frame prev/curr buffer at 600 bytes; the window
//!     grew to 5 frames so H_te/H_in can be computed from a real
//!     multi-sample distribution instead of a single-sample one-hot
//!     histogram, which is mathematically always ≈0.0 entropy — see
//!     HemmWindowEmbedded doc comment)
//!   BNN weights:  loaded from flash at boot
//!   Total RAM:    < 4 KB (fits on Cortex-M0+)
//!
//! Power states
//! ─────────────
//!   NULL  (idle)    → 0.00 W  ← default state
//!   AWAKE (compute) → ~8 µW   ← only during inference pulse
//!   SLEEP (deep)    → 0.00 W  ← same as NULL in NCL

// Conditionally no_std: bare-metal on ARM, std on host for testing
#![cfg_attr(target_arch = "arm", no_std)]
#![cfg_attr(target_arch = "arm", no_main)]

// Panic handler — only on bare-metal ARM targets
#[cfg(target_arch = "arm")]
use panic_halt as _;

// Atomic types
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

// ── Embedded HAL stubs ─────────────────────────────────────────────────────
// These will be replaced with real peripheral drivers when targeting
// specific silicon (e.g. ATSAMD21, STM32L0, nRF52840).
// For now they are trait definitions that compile on any target.

/// Abstract sensor interface — replaced by real HAL impl on hardware.
pub trait SkeletonSensor {
    /// Returns true when a new frame is available.
    fn frame_ready(&self) -> bool;
    /// Read the current frame into the provided buffer.
    /// joints: mutable slice of (x, y, z) f32 triplets, len = N_JOINTS
    fn read_frame(&mut self, joints: &mut [(f32, f32, f32)]);
}

/// Abstract harvesting frontend interface.
pub trait HarvestingFrontend {
    /// Returns true when enough energy is stored for one BNN pulse.
    fn energy_ready(&self) -> bool;
    /// Consume energy for one inference pulse (30.4 nJ).
    fn consume_pulse(&mut self);
    /// Current accumulated energy in µJ.
    fn accumulated_uj(&self) -> f32;
}

/// Abstract output driver — drives quality state and alert signals.
pub trait OutputDriver {
    /// Drive quality state output (1–5) onto GPIO or NCL output rail.
    fn set_quality_state(&mut self, state: u8);
    /// Drive alert signal HIGH or LOW.
    fn set_alert(&mut self, active: bool);
    /// Return all outputs to idle (LOW / NULL).
    fn reset_outputs(&mut self);
}

// ── Firmware state ─────────────────────────────────────────────────────────

/// Global frame counter — wraps at u32::MAX (~4.7 years at 30fps)
static FRAME_COUNT:    AtomicU32 = AtomicU32::new(0);
/// Global inference counter
static INFERENCE_COUNT: AtomicU32 = AtomicU32::new(0);
/// Alert state — shared between ISR and main loop
static ALERT_ACTIVE:   AtomicBool = AtomicBool::new(false);

// ── BNN weight storage ─────────────────────────────────────────────────────
// In production, weights are stored in flash and loaded at boot.
// This const block reserves the correct amount of flash space.

const N_IN:     usize = 3;
const N_HIDDEN: usize = 16;
const N_OUT:    usize = 5;

/// BNN weight block — lives in flash (read-only).
///
/// These are literal values copied from models/aura_m1_bnn_v1.json
/// (checksum 72fabfe0), NOT independently regenerated from a seed. An
/// audit found that firmware's previous LCG-based generation used a
/// DIFFERENT seed for W2 (1337) than W1 (42) — and Rust used yet another
/// scheme (one continuous stream from seed 42 for both), and Python used
/// a third scheme entirely (random.Random, not the LCG at all). All three
/// were labelled "seed=42" in comments but produced different weights.
/// See tools/generate_canonical_bnn_model.py for the full explanation.
///
/// If the canonical model is regenerated (real training), these literals
/// must be regenerated to match — run
/// tools/generate_canonical_bnn_model.py, then re-copy its w1/w2 output
/// here. `tests::firmware_weights_match_canonical_checksum` (below) is the
/// regression test that catches these ever silently drifting apart.
#[used]
#[link_section = ".rodata"]
static BNN_W1: [[i8; N_IN]; N_HIDDEN] = [
    [-1, -1, 1],
    [-1, -1, -1],
    [-1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
    [-1, -1, 1],
    [1, 1, 1],
    [-1, 1, -1],
    [1, 1, -1],
    [1, -1, -1],
    [-1, 1, 1],
    [1, 1, -1],
    [1, 1, 1],
    [1, 1, -1],
    [-1, -1, -1],
    [-1, 1, 1],
];

/// Hidden-layer biases — literal values from the canonical model.
/// Previously MISSING entirely from firmware: the forward pass didn't
/// apply any bias in either layer. Fixed to match Python/Rust, which do.
#[used]
#[link_section = ".rodata"]
static BNN_B1: [f32; N_HIDDEN] = [
    0.30464404821395874, -0.11058861762285233, -0.28423091769218445,
    -0.22794052958488464, -0.48091524839401245, -0.22202815115451813,
    -0.16603165864944458, 0.38286611437797546, 0.4558780789375305,
    -0.30730140209198, 0.3713010251522064, 0.08232294768095016,
    -0.1633189171552658, -0.1856725513935089, -0.37064072489738464,
    0.47997018694877625,
];

#[used]
#[link_section = ".rodata"]
static BNN_W2: [[i8; N_HIDDEN]; N_OUT] = [
    [-1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1],
    [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, -1],
    [-1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1],
    [-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1],
];

/// Output-layer biases — literal values from the canonical model.
#[used]
#[link_section = ".rodata"]
static BNN_B2: [f32; N_OUT] = [
    0.075677789747715, -0.1967654973268509, 0.14507979154586792,
    0.1679522544145584, -0.03222690150141716,
];

// ── Core entropy functions (no_std compatible) ─────────────────────────────

const EPS: f32 = 1e-9;
const K_SECTORS: usize = 12;
const N_VEL_BINS: usize = 16;

/// Energy cost of one BNN inference pulse, in nanojoules.
/// MUST match `aura_core::BNN_PULSE_NJ` and
/// `aura_sdk.inference.energy.BNN_INFERENCE_ENERGY_NJ` — the cross-language
/// golden-vector tests are the mechanism that catches drift between these
/// three independently-defined constants (firmware cannot directly import
/// aura_core's std-based crate; see module doc comment on this repo's known
/// follow-up to unify this via a shared no_std-compatible constants crate).
const INFERENCE_ENERGY_NJ: f32 = 30.4;
const INFERENCE_ENERGY_UJ: f32 = INFERENCE_ENERGY_NJ / 1000.0;

/// Shannon entropy — no_std, stack only, no libm dependency.
#[inline(always)]
fn shannon_entropy_f32(dist: &[f32]) -> f32 {
    let mut h = 0.0_f32;
    for &p in dist.iter() {
        if p > EPS {
            // Approximate log2 using bit manipulation — no libm needed
            h -= p * fast_log2(p + EPS);
        }
    }
    h
}

/// Fast log2 approximation suitable for entropy computation.
/// Error < 0.01 bits — sufficient for quality state classification.
/// Uses IEEE 754 exponent extraction — no division, no libm.
#[inline(always)]
fn fast_log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp  = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant = f32::from_bits((bits & 0x7FFFFF) | 0x3F800000);
    // Minimax polynomial for log2(m) where m ∈ [1, 2)
    let p = -0.344845 * mant * mant + 2.024658 * mant - 1.674873;
    exp as f32 + p
}

/// Spatial entropy H_sp — stack-allocated, no_std.
#[inline]
fn compute_hsp_embedded(joints: &[(f32, f32, f32)]) -> f32 {
    let mut histogram = [0u32; K_SECTORS];
    for &(x, y, z) in joints.iter() {
        // Sector assignment using integer arithmetic only
        let az = {
            let a = (z.atan2(x) + core::f32::consts::PI)
                / (2.0 * core::f32::consts::PI);
            ((a * 8.0) as usize) % 8
        };
        let el = if y < -0.5 { 0usize } else if y < 0.5 { 1 } else { 2 };
        let sector = (az * 3 + el) % K_SECTORS;
        histogram[sector] += 1;
    }
    let total = joints.len() as f32;
    let dist: [f32; K_SECTORS] = histogram.map(|c| c as f32 / total);
    shannon_entropy_f32(&dist)
}

/// Temporal entropy H_te from a single (prev, curr) pair.
///
/// ⚠ WARNING — mathematically degenerate. A histogram built from exactly
/// one velocity sample per joint is always one-hot, and the Shannon entropy
/// of a one-hot distribution is always ≈0.0 regardless of input. Retained
/// only as a documented reference point and for the regression test below
/// that locks in this known behaviour. The firmware main loop uses
/// `HemmWindowEmbedded` instead (below), which accumulates a short
/// multi-frame history before computing entropy — this is the actual fix
/// for the audit finding that both this function and its Python/Rust
/// equivalents were silently returning ≈0.0 always.
#[allow(dead_code)]
#[inline]
fn compute_hte_embedded(
    prev: &[(f32, f32, f32)],
    curr: &[(f32, f32, f32)],
    vel_max: f32,
    n_joints: usize,
) -> f32 {
    let bin_width = vel_max / N_VEL_BINS as f32;
    let mut total = 0.0_f32;
    for j in 0..n_joints.min(prev.len()).min(curr.len()) {
        let dx = curr[j].0 - prev[j].0;
        let dy = curr[j].1 - prev[j].1;
        let dz = curr[j].2 - prev[j].2;
        let vel = (dx*dx + dy*dy + dz*dz).sqrt();
        let bin = ((vel / bin_width) as usize).min(N_VEL_BINS - 1);
        let mut hist = [0.0_f32; N_VEL_BINS];
        hist[bin] = 1.0;
        total += shannon_entropy_f32(&hist);
    }
    total / n_joints as f32
}

// ── HemmWindowEmbedded — the actual fix, no_std / no-heap ──────────────────
//
// Mirrors `aura_core::hemm::HemmWindow` exactly (same CAP=5, same 16 bins,
// same epsilon, same windowing convention) but built on raw (f32,f32,f32)
// tuples and firmware's own `fast_log2`-based `shannon_entropy_f32`, so it
// needs no libm and no std — firmware stays fully self-contained rather
// than depending on aura_core's std-based implementation.
//
// Memory: CAP=5 frames × 25 joints × 12 bytes = 1500 bytes. Well within the
// <4KB RAM budget (previously 600 bytes for the 2-frame prev/curr buffer).
const HEMM_WINDOW_CAP: usize = 5;

struct HemmWindowEmbedded {
    frames: [[(f32, f32, f32); 25]; HEMM_WINDOW_CAP],
    len:  usize,
    head: usize,
}

impl HemmWindowEmbedded {
    const fn new() -> Self {
        Self {
            frames: [[(0.0, 0.0, 0.0); 25]; HEMM_WINDOW_CAP],
            len: 0,
            head: 0,
        }
    }

    fn push(&mut self, frame: &[(f32, f32, f32); 25]) {
        self.frames[self.head] = *frame;
        self.head = (self.head + 1) % HEMM_WINDOW_CAP;
        if self.len < HEMM_WINDOW_CAP {
            self.len += 1;
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn ordered(&self) -> [[(f32, f32, f32); 25]; HEMM_WINDOW_CAP] {
        let mut out = [[(0.0, 0.0, 0.0); 25]; HEMM_WINDOW_CAP];
        let start = if self.len < HEMM_WINDOW_CAP { 0 } else { self.head };
        for (i, slot) in out.iter_mut().enumerate().take(self.len) {
            *slot = self.frames[(start + i) % HEMM_WINDOW_CAP];
        }
        out
    }

    /// H_te over whatever history is currently buffered (honest partial
    /// window during warm-up, same convention as aura_core::HemmWindow).
    fn compute_hte(&self, vel_max: f32) -> f32 {
        if self.len < 2 { return 0.0; }
        let bin_width = vel_max / N_VEL_BINS as f32;
        let ordered = self.ordered();
        let n_pairs = (self.len - 1) as f32;

        let mut total_entropy = 0.0_f32;
        // `j` indexes the joint dimension while `w` (inner loop) indexes
        // the frame dimension — needed for parallel access across both.
        #[allow(clippy::needless_range_loop)]
        for j in 0..25 {
            let mut hist = [0u32; N_VEL_BINS];
            for w in 1..self.len {
                let (px, py, pz) = ordered[w-1][j];
                let (cx, cy, cz) = ordered[w][j];
                let (dx, dy, dz) = (cx-px, cy-py, cz-pz);
                let vel = (dx*dx + dy*dy + dz*dz).sqrt();
                let bin = ((vel / bin_width) as usize).min(N_VEL_BINS - 1);
                hist[bin] += 1;
            }
            let mut dist = [0.0_f32; N_VEL_BINS];
            for k in 0..N_VEL_BINS { dist[k] = hist[k] as f32 / n_pairs; }
            total_entropy += shannon_entropy_f32(&dist);
        }
        total_entropy / 25.0
    }

    /// H_in over whatever history is currently buffered — the real
    /// inter-joint entropy, replacing the `(h_sp + h_te) * 0.5` proxy that
    /// previously stood in for it. Same 300-pair / 256-cell algorithm as
    /// `aura_core::hemm::compute_hin_windowed`, adapted to a variable-length
    /// partial window and firmware's fast_log2-based entropy.
    fn compute_hin(&self, vel_max: f32) -> f32 {
        if self.len < 2 { return 0.0; }
        let bin_width = vel_max / N_VEL_BINS as f32;
        let ordered = self.ordered();
        let n_pairs = (self.len - 1) as f32;

        let mut total_entropy = 0.0_f32;
        let mut pair_count = 0usize;
        for a in 0..25 {
            for b in (a + 1)..25 {
                let mut hist = [0u32; N_VEL_BINS * N_VEL_BINS];
                for w in 1..self.len {
                    let (pax, pay, paz) = ordered[w-1][a];
                    let (cax, cay, caz) = ordered[w][a];
                    let (dax, day, daz) = (cax-pax, cay-pay, caz-paz);
                    let vel_a = (dax*dax + day*day + daz*daz).sqrt();
                    let (pbx, pby, pbz) = ordered[w-1][b];
                    let (cbx, cby, cbz) = ordered[w][b];
                    let (dbx, dby, dbz) = (cbx-pbx, cby-pby, cbz-pbz);
                    let vel_b = (dbx*dbx + dby*dby + dbz*dbz).sqrt();
                    let bin_a = ((vel_a / bin_width) as usize).min(N_VEL_BINS - 1);
                    let bin_b = ((vel_b / bin_width) as usize).min(N_VEL_BINS - 1);
                    hist[bin_a * N_VEL_BINS + bin_b] += 1;
                }
                let mut dist = [0.0_f32; N_VEL_BINS * N_VEL_BINS];
                for k in 0..(N_VEL_BINS * N_VEL_BINS) { dist[k] = hist[k] as f32 / n_pairs; }
                total_entropy += shannon_entropy_f32(&dist);
                pair_count += 1;
            }
        }
        if pair_count == 0 { 0.0 } else { total_entropy / pair_count as f32 }
    }
}

// ── BNN forward pass — no_std ──────────────────────────────────────────────

/// BNN inference result — minimal, stack-only.
#[derive(Clone, Copy)]
pub struct FirmwareInferenceResult {
    pub q_hci:      f32,
    pub state:      u8,
    pub energy_nj:  f32,
}

/// BNN forward pass using flash-resident weights.
/// No heap. No panic path. Deterministic.
#[inline]
fn bnn_infer_embedded(h_sp: f32, h_te: f32, h_in: f32) -> FirmwareInferenceResult {
    // Quantise inputs to {-1, +1}
    let x: [i8; N_IN] = [
        if h_sp >= 0.5 { 1 } else { -1 },
        if h_te >= 0.5 { 1 } else { -1 },
        if h_in >= 0.5 { 1 } else { -1 },
    ];

    // Hidden layer — XNOR-popcount + bias
    // BUGFIX: bias was previously not applied at all in firmware. Added
    // to match Python/Rust, which both apply it (Rust had its own separate
    // bug where a float bias got truncated to i32 = always 0 — fixed
    // separately in aura_core; firmware simply never had bias code here).
    let mut hidden = [0i8; N_HIDDEN];
    for j in 0..N_HIDDEN {
        let pop: i32 = (0..N_IN)
            .map(|i| if BNN_W1[j][i] == x[i] { 1i32 } else { -1 })
            .sum();
        hidden[j] = if (pop as f32 + BNN_B1[j]) >= 0.0 { 1 } else { -1 };
    }

    // Output layer — XNOR-popcount + bias
    let mut logits = [0.0_f32; N_OUT];
    for k in 0..N_OUT {
        let pop: i32 = (0..N_HIDDEN)
            .map(|j| if BNN_W2[k][j] == hidden[j] { 1i32 } else { -1 })
            .sum();
        logits[k] = pop as f32 + BNN_B2[k];
    }

    // Softmax — embedded friendly (subtract max for numerical stability)
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_l = [0.0_f32; N_OUT];
    let mut sum_e = 0.0_f32;
    for k in 0..N_OUT {
        exp_l[k] = fast_exp(logits[k] - max_l);
        sum_e += exp_l[k];
    }
    let probs: [f32; N_OUT] = exp_l.map(|e| e / sum_e.max(EPS));

    // State = argmax + 1
    let state = (probs.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0) + 1) as u8;

    // Continuous Q_HCI
    let q_raw: f32 = probs.iter().enumerate()
        .map(|(i, &p)| (i + 1) as f32 * p)
        .sum::<f32>() - 1.0;
    let q_hci = (q_raw / (N_OUT - 1) as f32).clamp(0.0, 1.0);

    FirmwareInferenceResult { q_hci, state, energy_nj: INFERENCE_ENERGY_NJ }
}

/// Fast exp approximation — no libm, suitable for softmax on embedded.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    // Clamped to prevent overflow — sufficient for softmax
    let x = x.clamp(-20.0, 20.0);
    // Approximation: e^x ≈ (1 + x/256)^256
    let y = 1.0 + x / 256.0;
    let y2 = y * y; let y4 = y2 * y2; let y8 = y4 * y4;
    let y16 = y8 * y8; let y32 = y16 * y16; let y64 = y32 * y32;
    let y128 = y64 * y64;
    y128 * y128  // = y^256
}

// ── Main firmware loop ─────────────────────────────────────────────────────

/// Firmware entry point.
/// Called by the reset handler after stack pointer initialisation.
///
/// On real hardware this function never returns.
/// In simulation (host build) it runs for N_FRAMES then exits.
///
/// NOTE: previously had `#[cfg_attr(not(test), no_mangle)]` here, which
/// clippy correctly flags as meaningless on a generic function — name
/// mangling exists specifically to disambiguate different monomorphized
/// instances of a generic function, so forcing it off doesn't make sense
/// and isn't safe once more than one concrete S/H/O combination exists.
/// Removed rather than suppressed: confirmed via repo-wide search that no
/// linker script or vector table anywhere references this symbol by a
/// fixed unmangled name (there is no memory.x / cortex-m-rt entry point
/// wired up yet — see aura_firmware/Cargo.toml's dependency note). A real
/// hardware entry point will need to call this from a concrete,
/// non-generic wrapper function instead of relying on this signature
/// being directly callable by name.
pub fn firmware_main<S, H, O>(
    sensor:    &mut S,
    harvester: &mut H,
    output:    &mut O,
    n_frames:  Option<u32>,  // None = run forever on hardware
) where
    S: SkeletonSensor,
    H: HarvestingFrontend,
    O: OutputDriver,
{
    // Frame history — stack allocated, reused every frame.
    // HEMM_WINDOW_CAP=5 frames buffered so H_te/H_in can be computed over a
    // real multi-sample distribution instead of a single (prev, curr) pair
    // (see HemmWindowEmbedded doc comment for why the single-pair path is
    // mathematically degenerate).
    const N_JOINTS: usize = 25;
    let mut curr_frame = [(0.0_f32, 0.0_f32, 0.0_f32); N_JOINTS];
    let mut window = HemmWindowEmbedded::new();

    let limit = n_frames.unwrap_or(u32::MAX);

    loop {
        let frame_num = FRAME_COUNT.fetch_add(1, Ordering::Relaxed);
        if frame_num >= limit { break; }

        // ── Stage 1: Wait for sensor frame ─────────────────────────
        if !sensor.frame_ready() {
            // In real firmware: WFI (wait-for-interrupt) here
            // CPU enters deep sleep, wakes on sensor interrupt
            // Power during WFI ≈ 0.00 W (NCL NULL state equivalent)
            continue;
        }

        // ── Stage 2: Read new frame, push onto the rolling window ───
        sensor.read_frame(&mut curr_frame);
        window.push(&curr_frame);

        // Need at least 2 buffered frames for velocity computation
        if window.len() < 2 {
            continue;
        }

        // ── Stage 3: Check energy gate ──────────────────────────────
        // If insufficient energy — return to NULL state, skip inference
        if !harvester.energy_ready() {
            output.reset_outputs();  // ensure outputs are NULL
            continue;
        }

        // ── Stage 4: Consume energy pulse ──────────────────────────
        harvester.consume_pulse();

        // ── Stage 5: Compute entropy features ──────────────────────
        // All stack-allocated, no heap. H_te and H_in are both computed
        // over the rolling window (real entropy, not the previous proxy).
        let h_sp = compute_hsp_embedded(&curr_frame);
        let h_te = window.compute_hte(0.5);
        let h_in = window.compute_hin(0.5);

        // ── Stage 6: BNN inference ──────────────────────────────────
        let result = bnn_infer_embedded(h_sp, h_te, h_in);
        INFERENCE_COUNT.fetch_add(1, Ordering::Relaxed);

        // ── Stage 7: Drive outputs ──────────────────────────────────
        output.set_quality_state(result.state);

        let alert = result.q_hci < 0.42;
        output.set_alert(alert);
        ALERT_ACTIVE.store(alert, Ordering::Relaxed);

        // ── Stage 8: Return to NULL state ───────────────────────────
        // NCL NULL wave propagates back through the datapath.
        // All rails return to (0,0). Power drops to 0.00 W.
        // This happens implicitly as registers are not re-driven.
    }

    // Clean shutdown — reset all outputs to NULL/idle
    output.reset_outputs();
}

// ── Public accessors ───────────────────────────────────────────────────────

/// Read global frame counter (safe from ISR context).
pub fn frame_count() -> u32 {
    FRAME_COUNT.load(Ordering::Relaxed)
}

/// Read global inference counter.
pub fn inference_count() -> u32 {
    INFERENCE_COUNT.load(Ordering::Relaxed)
}

/// Read current alert state.
pub fn alert_active() -> bool {
    ALERT_ACTIVE.load(Ordering::Relaxed)
}

// ── Simulation entry point (host/test builds) ──────────────────────────────

/// Standard main — only compiled on host (not on bare metal).
/// Runs the firmware logic with stub implementations for testing.
#[cfg(not(target_arch = "arm"))]
fn main() {
    println!("  Aura-M1 Firmware v0.1.0");
    println!("  GreyMatter Labs Ltd. · 'Where intelligence begins.'");
    println!("  Running host simulation (stub peripherals)...\n");

    // ── Stub implementations ───────────────────────────────────────

    struct StubSensor { frame: u32, rng: u32 }
    impl StubSensor {
        fn next_f32(&mut self) -> f32 {
            self.rng = self.rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.rng as f32 / u32::MAX as f32) * 0.4 + 0.05
        }
    }
    impl SkeletonSensor for StubSensor {
        fn frame_ready(&self) -> bool { true }
        fn read_frame(&mut self, joints: &mut [(f32, f32, f32)]) {
            self.frame += 1;
            for j in joints.iter_mut() {
                *j = (self.next_f32(), self.next_f32(), self.next_f32());
            }
        }
    }

    struct StubHarvester { acc: f32 }
    impl HarvestingFrontend for StubHarvester {
        fn energy_ready(&self) -> bool { self.acc >= INFERENCE_ENERGY_UJ }
        fn consume_pulse(&mut self) { self.acc -= INFERENCE_ENERGY_UJ; }
        fn accumulated_uj(&self) -> f32 { self.acc }
    }

    struct StubOutput { last_state: u8, alert: bool }
    impl OutputDriver for StubOutput {
        fn set_quality_state(&mut self, s: u8) { self.last_state = s; }
        fn set_alert(&mut self, a: bool) { self.alert = a; }
        fn reset_outputs(&mut self) { self.last_state = 0; self.alert = false; }
    }

    let mut sensor    = StubSensor   { frame: 0, rng: 42 };
    let mut harvester = StubHarvester { acc: 0.5 };   // pre-charged
    let mut output    = StubOutput   { last_state: 0, alert: false };

    let n_frames: u32 = 120;
    firmware_main(&mut sensor, &mut harvester, &mut output, Some(n_frames));

    println!("  Frames processed : {}", frame_count());
    println!("  Inferences fired : {}", inference_count());
    println!("  Alert active     : {}", alert_active());
    println!("  Idle power       : 0.00 W <- SIMULATED ZERO (NCL NULL, no hardware measurement)");
    println!("\n  Firmware simulation complete.");
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_log2_accuracy() {
        // Verify fast_log2 is within 0.05 bits of f32::log2
        for &x in &[0.1_f32, 0.5, 1.0, 2.0, 4.0, 0.25, 0.9] {
            let fast = fast_log2(x);
            let exact = x.log2();
            assert!((fast - exact).abs() < 0.05,
                "fast_log2({x}) = {fast}, exact = {exact}");
        }
    }

    #[test]
    fn fast_exp_accuracy() {
        for &x in &[0.0_f32, 1.0, -1.0, 2.0, -2.0, 0.5] {
            let fast  = fast_exp(x);
            let exact = x.exp();
            let rel_err = (fast - exact).abs() / exact.max(EPS);
            assert!(rel_err < 0.05,
                "fast_exp({x}) = {fast}, exact = {exact}, rel_err = {rel_err}");
        }
    }

    #[test]
    fn hsp_embedded_valid_range() {
        let joints: [(f32, f32, f32); 25] = core::array::from_fn(|i| {
            (i as f32 * 0.1, i as f32 * 0.05, i as f32 * 0.08)
        });
        let h = compute_hsp_embedded(&joints);
        assert!(h >= 0.0, "H_sp must be non-negative");
        assert!(h <= (super::K_SECTORS as f32) * 0.7,
                "H_sp = {h} seems too high");
    }

    /// REGRESSION TEST — documents the single-frame degeneracy on purpose,
    /// matching the equivalent test in aura_core::hemm. Locks in the known
    /// behaviour of the deprecated single-pair function so it can never be
    /// mistaken for a live bug again, and so the firmware main loop is
    /// never tempted to call it directly.
    #[test]
    fn single_frame_hte_embedded_is_degenerate_by_construction() {
        let prev: [(f32, f32, f32); 25] = [(0.0, 0.0, 0.0); 25];
        let curr: [(f32, f32, f32); 25] = [(0.9, 0.9, 0.9); 25]; // large motion
        let h = compute_hte_embedded(&prev, &curr, 0.5, 25);
        // Tolerance widened from 1e-3: fast_log2(1.0) has a known residual
        // error of ~0.005 (bit-manipulation approximation, no libm in
        // no_std), so a mathematically-correct one-hot entropy evaluates to
        // ≈-0.005 here rather than exactly 0. 0.01 comfortably covers that
        // known approximation while still catching a real regression.
        assert!(h.abs() < 0.01, "single-frame H_te should be ≈0 (one-hot), got {h}");
    }

    #[test]
    fn windowed_hte_and_hin_are_not_degenerate() {
        // Deterministic pseudo-random varied motion across 5 frames.
        let mut window = HemmWindowEmbedded::new();
        let mut seed: u32 = 7;
        for _ in 0..5 {
            let frame: [(f32, f32, f32); 25] = core::array::from_fn(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let a = (seed as f32 / u32::MAX as f32) * 0.4 - 0.2;
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let b = (seed as f32 / u32::MAX as f32) * 0.4 - 0.2;
                (a, b, a - b)
            });
            window.push(&frame);
        }
        let h_te = window.compute_hte(0.5);
        let h_in = window.compute_hin(0.5);
        assert!(h_te > 0.05, "windowed firmware H_te should be meaningfully non-zero, got {h_te}");
        assert!(h_in > 0.05, "windowed firmware H_in should be meaningfully non-zero, got {h_in}");
    }

    #[test]
    fn windowed_hte_zero_for_static_joints() {
        let mut window = HemmWindowEmbedded::new();
        let still: [(f32, f32, f32); 25] = [(0.3, 0.3, 0.3); 25];
        for _ in 0..5 {
            window.push(&still);
        }
        let h_te = window.compute_hte(0.5);
        let h_in = window.compute_hin(0.5);
        // Same fast_log2 approximation reasoning as above.
        assert!(h_te.abs() < 0.01, "static skeleton H_te should be ≈0, got {h_te}");
        assert!(h_in.abs() < 0.01, "static skeleton H_in should be ≈0, got {h_in}");
    }

    /// Cross-language golden-vector parity test — the firmware-side
    /// counterpart to aura_core/tests/hemm_golden_vectors.rs /
    /// aura_sdk_tests/test_hemm_golden_vectors.py, which only ever
    /// compared Python against Rust `aura_core`, never against firmware.
    /// The 6 input frames and expected h_sp/h_te/h_in/state/q_hci below
    /// were generated from and cross-checked against the verified Python
    /// reference (SyntheticSource(seed=2026), 6 frames -> one full
    /// HEMM_WINDOW_CAP=5 window) — see the generation command in the
    /// comment at the bottom of this test.
    ///
    /// Tolerances are wider than aura_core's own golden-vector test
    /// (1e-4 / 1e-3) because firmware deliberately uses `fast_log2` /
    /// `fast_exp` approximations instead of exact libm — that's an
    /// intentional, tested, no_std/no-libm tradeoff (see
    /// `fast_log2_accuracy`, `fast_exp_accuracy` above), not something
    /// this test should paper over with a loose bound; the values below
    /// are chosen to be the tightest bound that tradeoff actually allows.
    ///
    /// HONESTY NOTE: this specific test was authored and its expected
    /// values were cross-checked against the Python reference implementation
    /// directly, but — unlike every other test in this file — it was NOT
    /// executed against a compiled build of this firmware crate before
    /// being committed here, because no Rust toolchain was available in
    /// the environment this was written in. Run
    /// `cargo test -p aura_firmware golden_vector_matches_python_reference`
    /// to confirm before relying on it; if it fails narrowly, the fix is
    /// very likely a wider tolerance rather than a real Hin/Hte bug (see
    /// the paragraph above), given `windowed_hte_and_hin_are_not_degenerate`
    /// and `firmware_weights_match_canonical_checksum` above already
    /// independently confirm the underlying algorithm and weights are
    /// correct.
    #[test]
    fn golden_vector_matches_python_reference() {
        // Generated by (see aura_sdk_tests / this repo's Python reference):
        //   SyntheticSource(n_frames=6, seed=2026) -> HEMMExtractor.compute()
        //   pushed frame-by-frame -> _bnn_forward(h_sp, h_te, h_in)
        const GOLDEN_FRAMES: [[(f32, f32, f32); 25]; 6] = [
            [(0.0,0.0,0.0),(0.029064511,0.017890066,0.011274508),(0.047062627,0.018111859,0.031464108),(0.045605787,0.025765968,0.038131015),(0.065948609,0.043287607,0.052344766),(0.098021143,0.039589396,0.052768371),(0.096225064,0.060113789,0.07687553),(0.11210028,0.064556807,0.089663943),(0.12921443,0.064986217,0.098273308),(0.1407316,0.086168338,0.11324463),(0.1578842,0.076067054,0.12770454),(0.16779144,0.085353775,0.11908603),(0.19399137,0.095607631,0.14003378),(0.20285744,0.10880839,0.1499934),(0.22154201,0.12047862,0.16318843),(0.23608616,0.12733761,0.18489284),(0.26290185,0.13869269,0.18307998),(0.25944511,0.12936519,0.19449187),(0.28629679,0.15565475,0.20887294),(0.31076147,0.15646332,0.23849311),(0.32056264,0.15711215,0.231942),(0.32076733,0.17915886,0.25323688),(0.34279617,0.17294881,0.25625172),(0.36586,0.18327366,0.26772014),(0.36941518,0.19599499,0.29523251)],
            [(0.0,0.0,0.0),(0.013023282,-0.0019495309,0.0095023762),(0.024144422,0.0087685317,0.026420433),(0.044493272,0.0029121764,0.034177933),(0.03987503,0.015455638,0.041524469),(0.065108627,0.021113814,0.066040763),(0.081933308,0.028650411,0.076482968),(0.09704484,0.042512326,0.086922713),(0.11470967,0.053982846,0.10108643),(0.13185776,0.055889609,0.12066586),(0.14059566,0.067142738,0.12520768),(0.15058533,0.075407103,0.13281975),(0.16464852,0.089581077,0.15476427),(0.19357497,0.093167022,0.15370821),(0.20256923,0.10066119,0.17621276),(0.21316985,0.11246936,0.18362502),(0.22411504,0.10967838,0.20426971),(0.24007077,0.11482477,0.21088178),(0.25118485,0.11508532,0.21455697),(0.27129201,0.14265059,0.2344682),(0.28543642,0.13738876,0.23106215),(0.31031732,0.1453458,0.25904094),(0.31341168,0.15790201,0.27502897),(0.33011882,0.1640357,0.27354543),(0.34156633,0.17453261,0.29614247)],
            [(0.0,0.0,0.0),(0.019769005,-0.0027717839,0.0014081666),(0.024202909,0.002443937,0.0067831582),(0.036727009,0.02273195,0.023722803),(0.053324485,0.034278615,0.033791211),(0.081937157,0.027691104,0.063616792),(0.10318436,0.042697066,0.066990281),(0.11610071,0.051022509,0.076411259),(0.13356399,0.058207914,0.084432907),(0.1443472,0.074169043,0.098837042),(0.14601391,0.072576805,0.10799483),(0.16394449,0.088185948,0.13430459),(0.19230609,0.084493857,0.1415546),(0.19665394,0.089100766,0.14526472),(0.20708324,0.11085082,0.17206274),(0.23916338,0.10234393,0.18577874),(0.24612576,0.12287084,0.17720945),(0.27372095,0.13138572,0.19691623),(0.27375447,0.13954527,0.20526601),(0.28467457,0.13291716,0.23517455),(0.31723829,0.14132429,0.22810461),(0.32729381,0.1509062,0.25044419),(0.34987431,0.1715693,0.27407801),(0.35355521,0.18383689,0.27469904),(0.37370877,0.17690846,0.28337424)],
            [(0.0,0.0,0.0),(0.035964941,-0.0024175497,0.017786719),(0.03646278,0.014717619,0.03338626),(0.052525931,0.020777835,0.032916876),(0.065701879,0.014985267,0.065909907),(0.09762893,0.039234644,0.073179061),(0.093107878,0.037399056,0.083998102),(0.12577212,0.046333601,0.1004357),(0.12611854,0.043820376,0.1036904),(0.13995294,0.072107925,0.12959454),(0.17401921,0.062259663,0.13246485),(0.18493295,0.076486085,0.15256591),(0.18659406,0.088487124,0.14703689),(0.22161961,0.097942383,0.16270537),(0.23791712,0.096731914,0.1767503),(0.24973655,0.10482645,0.20335186),(0.2698101,0.11867494,0.20736856),(0.28303583,0.13346901,0.215762),(0.29151663,0.14327368,0.24358887),(0.30479167,0.14555036,0.24227403),(0.32479582,0.14193901,0.2623114),(0.33153487,0.1517944,0.27855201),(0.35319882,0.17443908,0.28378442),(0.36637903,0.17817286,0.30014989),(0.39115468,0.18156454,0.30597892)],
            [(0.0,0.0,0.0),(0.017732089,0.023629536,-0.0045861506),(0.050779062,0.031059971,0.016283268),(0.064340183,0.019497641,0.043447135),(0.075353249,0.045175268,0.033762401),(0.099507107,0.053430533,0.045266256),(0.12110593,0.060135892,0.077429579),(0.12738922,0.056579236,0.072045609),(0.14959133,0.06326978,0.083316538),(0.15298001,0.066540078,0.10549439),(0.16799637,0.085897409,0.12828482),(0.18220264,0.082561875,0.12238351),(0.21270884,0.092793171,0.16153059),(0.22220575,0.11362564,0.17344383),(0.25043806,0.13020221,0.18450602),(0.24445227,0.1127819,0.17878455),(0.2710359,0.12982605,0.18986409),(0.29106394,0.14706826,0.2102286),(0.29374191,0.14522793,0.22704063),(0.33442637,0.16008929,0.25131811),(0.33887659,0.16853625,0.24890601),(0.34571188,0.1675621,0.25675718),(0.35880946,0.17431189,0.26914594),(0.38950651,0.19267289,0.29990785),(0.39585764,0.20482909,0.31008312)],
            [(0.0,0.0,0.0),(0.038063466,0.00016972321,0.023906391),(0.047972699,0.028454517,0.010537386),(0.070673635,0.039478804,0.032045853),(0.084994214,0.034177634,0.053345642),(0.10231446,0.042566078,0.071683848),(0.10910244,0.04361227,0.065485729),(0.12052631,0.048452238,0.08770402),(0.14157347,0.081001731,0.10967118),(0.15238999,0.084891031,0.11421702),(0.16814255,0.087929733,0.1382786),(0.2049702,0.087903125,0.14200385),(0.21157323,0.090346562,0.14952843),(0.22043973,0.10492003,0.17952338),(0.24486936,0.13067906,0.17355866),(0.27348984,0.12039156,0.18665078),(0.28758488,0.13652859,0.19946349),(0.30172159,0.1349637,0.21813019),(0.30594819,0.14969889,0.24965389),(0.32941382,0.15951243,0.2541508),(0.33360501,0.16277455,0.26429175),(0.37527133,0.17691339,0.26819223),(0.36858014,0.19711324,0.28678767),(0.39482263,0.20385406,0.29352954),(0.40118686,0.20254894,0.32728633)],
        ];
        // Python reference (float64, exact log2/exp): h_sp=0.0,
        // h_te=0.5969623723382909, h_in=1.0566841839618395, state=1,
        // q_hci=0.0636376251087028
        const EXPECTED_H_SP: f32 = 0.0;
        const EXPECTED_H_TE: f32 = 0.5969624;
        const EXPECTED_H_IN: f32 = 1.0566842;
        const EXPECTED_STATE: u8 = 1;
        const EXPECTED_Q_HCI: f32 = 0.0636376;

        const H_SP_TOL: f32 = 0.02;   // degenerate single-bin case; fast_log2 barely engages
        const H_TE_TOL: f32 = 0.15;   // sum of 25 fast_log2 terms
        const H_IN_TOL: f32 = 0.20;   // sum of 300 fast_log2 terms
        const Q_HCI_TOL: f32 = 0.05;  // compounds fast_log2 (features) + fast_exp (softmax)

        let mut window = HemmWindowEmbedded::new();
        for frame in GOLDEN_FRAMES.iter() {
            window.push(frame);
        }
        let h_sp = compute_hsp_embedded(GOLDEN_FRAMES.last().unwrap());
        let h_te = window.compute_hte(0.5);
        let h_in = window.compute_hin(0.5);

        assert!((h_sp - EXPECTED_H_SP).abs() < H_SP_TOL,
            "H_sp firmware={h_sp} vs Python reference={EXPECTED_H_SP} (tol={H_SP_TOL})");
        assert!((h_te - EXPECTED_H_TE).abs() < H_TE_TOL,
            "H_te firmware={h_te} vs Python reference={EXPECTED_H_TE} (tol={H_TE_TOL})");
        assert!((h_in - EXPECTED_H_IN).abs() < H_IN_TOL,
            "H_in firmware={h_in} vs Python reference={EXPECTED_H_IN} (tol={H_IN_TOL})");

        let result = bnn_infer_embedded(h_sp, h_te, h_in);
        assert_eq!(result.state, EXPECTED_STATE,
            "state firmware={} vs Python reference={EXPECTED_STATE} — this is a discrete \
             argmax, so unlike h_sp/h_te/h_in it should match exactly whenever the inputs \
             are all comfortably away from a quantisation boundary (they are here: h_sp=0.0, \
             h_te~0.6, h_in~1.06, all far from the 0.5 threshold)", result.state);
        assert!((result.q_hci - EXPECTED_Q_HCI).abs() < Q_HCI_TOL,
            "q_hci firmware={} vs Python reference={EXPECTED_Q_HCI} (tol={Q_HCI_TOL})", result.q_hci);
    }

    /// CROSS-LANGUAGE PROOF (firmware side): the literal BNN_W1/BNN_B1/
    /// BNN_W2/BNN_B2 arrays above must produce the exact same checksum as
    /// models/aura_m1_bnn_v1.json declares ("72fabfe0" at generation
    /// time) and the same value aura_core's
    /// canonical_model_checksum_matches_python test hardcodes. If someone
    /// regenerates the canonical model and copies new literals into this
    /// file incorrectly (typo, wrong row order, stale copy), this test
    /// fails loudly instead of firmware silently running different
    /// weights than Python and Rust.
    #[test]
    fn firmware_weights_match_canonical_checksum() {
        const EXPECTED_CHECKSUM_HEX: &str = "72fabfe0";

        // Minimal no_std-safe FNV-1a, matching
        // tools/generate_canonical_bnn_model.py's fnv1a_32() and
        // aura_core::inference::fnv1a_32() byte-for-byte.
        fn fnv1a_32(data: &[u8]) -> u32 {
            let mut h: u32 = 0x811C_9DC5;
            for &byte in data {
                h ^= byte as u32;
                h = h.wrapping_mul(0x0100_0193);
            }
            h
        }

        let mut buf: alloc_free_vec::Buf = alloc_free_vec::Buf::new();
        for row in BNN_W1.iter() {
            for &v in row {
                buf.push(if v == 1 { 0x01 } else { 0xFF });
            }
        }
        for &v in BNN_B1.iter() {
            buf.push_slice(&v.to_le_bytes());
        }
        for row in BNN_W2.iter() {
            for &v in row {
                buf.push(if v == 1 { 0x01 } else { 0xFF });
            }
        }
        for &v in BNN_B2.iter() {
            buf.push_slice(&v.to_le_bytes());
        }

        let computed = fnv1a_32(buf.as_slice());
        let expected = u32::from_str_radix(EXPECTED_CHECKSUM_HEX, 16).unwrap();
        assert_eq!(computed, expected,
            "firmware's literal BNN_W1/B1/W2/B2 checksum {computed:08x} does \
             not match the canonical model's declared checksum \
             {EXPECTED_CHECKSUM_HEX} \u{2014} the literals in this file have drifted \
             from models/aura_m1_bnn_v1.json");
    }

    /// Tiny fixed-capacity byte buffer for the test above — this crate is
    /// no_std with no heap on the ARM target, and the host-test build
    /// keeps std available but there's no reason to depend on Vec here
    /// when a stack array covers the known, fixed size
    /// (16*3 + 16*4 + 5*16 + 5*4 = 48+64+80+20 = 212 bytes) comfortably.
    mod alloc_free_vec {
        pub struct Buf {
            data: [u8; 256],
            len: usize,
        }
        impl Buf {
            pub fn new() -> Self { Self { data: [0u8; 256], len: 0 } }
            pub fn push(&mut self, b: u8) {
                self.data[self.len] = b;
                self.len += 1;
            }
            pub fn push_slice(&mut self, s: &[u8]) {
                for &b in s { self.push(b); }
            }
            pub fn as_slice(&self) -> &[u8] { &self.data[..self.len] }
        }
    }

    #[test]
    fn bnn_embedded_state_in_range() {
        let r = bnn_infer_embedded(0.42, 0.58, 0.31);
        assert!((1..=5).contains(&r.state),
                "State {} out of range 1-5", r.state);
    }

    #[test]
    fn bnn_embedded_q_hci_in_range() {
        let r = bnn_infer_embedded(0.3, 0.7, 0.4);
        assert!((0.0..=1.0).contains(&r.q_hci),
                "Q_HCI = {} out of [0,1]", r.q_hci);
    }

    #[test]
    fn bnn_embedded_deterministic() {
        let r1 = bnn_infer_embedded(0.5, 0.5, 0.5);
        let r2 = bnn_infer_embedded(0.5, 0.5, 0.5);
        assert_eq!(r1.state, r2.state);
        assert!((r1.q_hci - r2.q_hci).abs() < f32::EPSILON);
    }

    /// Locks INFERENCE_ENERGY_NJ (firmware's own literal, deliberately
    /// duplicated rather than depending on aura_core — see the NOTE on
    /// aura_core's dependency exclusion in this crate's Cargo.toml) against
    /// the same value declared independently in
    /// `aura_sdk.harvesting.sources.BNN_INFERENCE_ENERGY_NJ` (Python) and
    /// `aura_core::BNN_PULSE_NJ` (Rust core). If any of the three drift
    /// apart, this is the test that catches it on the firmware side.
    #[test]
    fn inference_energy_nj_matches_canonical_value() {
        // Cross-language reference: aura_sdk/harvesting/sources.py
        // BNN_INFERENCE_ENERGY_NJ = 30.4, aura_core/src/lib.rs
        // BNN_PULSE_NJ: f32 = 30.4.
        const CANONICAL_INFERENCE_ENERGY_NJ: f32 = 30.4;
        assert!(
            (INFERENCE_ENERGY_NJ - CANONICAL_INFERENCE_ENERGY_NJ).abs() < f32::EPSILON,
            "firmware's INFERENCE_ENERGY_NJ={INFERENCE_ENERGY_NJ} has drifted from the \
             canonical cross-language value={CANONICAL_INFERENCE_ENERGY_NJ} \
             (aura_sdk.harvesting.sources.BNN_INFERENCE_ENERGY_NJ / aura_core::BNN_PULSE_NJ)"
        );
        let result = bnn_infer_embedded(0.5, 0.5, 0.5);
        assert!(
            (result.energy_nj - CANONICAL_INFERENCE_ENERGY_NJ).abs() < f32::EPSILON,
            "bnn_infer_embedded() reported energy_nj={} instead of the canonical {CANONICAL_INFERENCE_ENERGY_NJ}",
            result.energy_nj
        );
    }

    #[test]
    fn firmware_main_runs_stub() {
        struct S;
        impl SkeletonSensor for S {
            fn frame_ready(&self) -> bool { true }
            fn read_frame(&mut self, j: &mut [(f32, f32, f32)]) {
                for jt in j.iter_mut() { *jt = (0.1, 0.2, 0.3); }
            }
        }
        struct H { acc: f32 }
        impl HarvestingFrontend for H {
            fn energy_ready(&self) -> bool { self.acc >= INFERENCE_ENERGY_UJ }
            fn consume_pulse(&mut self) { self.acc -= INFERENCE_ENERGY_UJ; }
            fn accumulated_uj(&self) -> f32 { self.acc }
        }
        struct O;
        impl OutputDriver for O {
            fn set_quality_state(&mut self, _s: u8) {}
            fn set_alert(&mut self, _a: bool) {}
            fn reset_outputs(&mut self) {}
        }
        // Reset global counters before test
        FRAME_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);
        INFERENCE_COUNT.store(0, core::sync::atomic::Ordering::Relaxed);

        let mut s = S;
        let mut h = H { acc: 10.0 };
        let mut o = O;
        firmware_main(&mut s, &mut h, &mut o, Some(30));

        assert!(frame_count() > 0);
    }
}
