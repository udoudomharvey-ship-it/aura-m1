//! aura_core::hemm
//!
//! HEMM Hybrid Entropy Descriptors — Rust implementation.
//!
//! Language: Rust
//! Why Rust: H_in alone requires 300 pairwise histograms × 256 cells
//!           × 30fps = 2.3M operations/second. Python with interpreter
//!           overhead cannot hit the 3.8ms/frame target. Rust compiles
//!           this to tight SIMD-eligible loops with zero overhead.
//!
//! Three descriptors per frame:
//!   H_sp — Spatial entropy    (HOJ3D, K=12 spherical sectors)
//!   H_te — Temporal entropy   (velocity histograms, 16 bins/joint)
//!   H_in — Inter-joint entropy (16×16 co-movement histograms)
//!
//! All computation is stack-allocated. No heap allocation in hot paths.

use std::f32;

// ── Constants ──────────────────────────────────────────────────────────────

/// Number of spherical sectors for HOJ3D (paper optimal K=12)
pub const K_SECTORS: usize = 12;

/// Number of velocity bins per joint for H_te
pub const N_VEL_BINS: usize = 16;

/// Number of joints — Kinect v2 (HCI-Lab-30)
pub const N_JOINTS_V2: usize = 25;

/// Number of joints — Kinect v1 (MSRC-12)
pub const N_JOINTS_V1: usize = 20;

/// Small epsilon to avoid log2(0)
const EPS: f32 = 1e-9;

/// Shannon entropy of a normalised distribution.
/// H = -Σ p × log2(p + ε)
/// Pure function — no allocation.
#[inline]
fn shannon_entropy(distribution: &[f32]) -> f32 {
    -distribution
        .iter()
        .map(|&p| p * (p + EPS).log2())
        .sum::<f32>()
}

// ── 3D Joint position ──────────────────────────────────────────────────────

/// A single 3D joint position in metres.
#[derive(Clone, Copy, Debug, Default)]
pub struct Joint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Joint {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }

    /// Euclidean distance to another joint.
    #[inline]
    pub fn distance_to(self, other: Joint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx*dx + dy*dy + dz*dz).sqrt()
    }

    /// Subtract another joint (for spine-base normalisation).
    #[inline]
    pub fn sub(self, other: Joint) -> Joint {
        Joint::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Velocity magnitude to next frame joint.
    #[inline]
    pub fn velocity_to(self, next: Joint) -> f32 {
        self.distance_to(next)
    }

    /// Azimuthal sector index (0..K_SECTORS) for HOJ3D.
    ///
    /// BUGFIX (found via cross-language golden-vector test): this used to
    /// compute elevation as an atan2-derived ANGLE in radians
    /// (`self.y.atan2(...)`) and threshold that angle against 0.5 — but the
    /// thesis specification (and the Python reference implementation) both
    /// use the raw Cartesian Y-COORDINATE in metres, thresholded at ±0.5 m.
    /// Radians and metres are different units; thresholding one against a
    /// constant meant for the other produced a systematically different
    /// (and wrong) sector assignment, which showed up as H_sp being ~1.2-1.4
    /// bits higher in Rust than Python on every golden-vector case. Fixed to
    /// match the documented spec exactly.
    #[inline]
    fn sector_index(self) -> usize {
        // Divide space into K_SECTORS sectors using azimuth + elevation.
        let azimuth = self.z.atan2(self.x); // -π to π
        // Map azimuth [-π, π] → [0, 8) sectors.
        let az_sector = (((azimuth + f32::consts::PI) / (2.0 * f32::consts::PI) * 8.0) as usize) % 8;
        // Elevation category from the raw Y position (metres), matching the
        // thesis spec exactly: below / level / above the horizontal plane
        // at y = ±0.5 m. NOT an angle — do not reintroduce atan2 here.
        let el_sector = if self.y < -0.5 { 0usize }
                        else if self.y < 0.5 { 1 }
                        else { 2 };
        // 8 azimuth × 3 elevation = 24 combinations, folded into K_SECTORS=12
        // via modulo (matches the Python reference exactly).
        (az_sector * 3 + el_sector) % K_SECTORS
    }
}

// ── Preprocessing ──────────────────────────────────────────────────────────

/// Normalise a frame of joints relative to the spine-base joint.
/// Spine-base is joint index 0 by convention.
/// Returns normalised joints on the stack.
#[inline]
pub fn normalise_frame<const N: usize>(joints: &[Joint; N]) -> [Joint; N] {
    let spine = joints[0];
    let mut out = [Joint::default(); N];
    for (i, &j) in joints.iter().enumerate() {
        out[i] = j.sub(spine);
    }
    out
}

/// Gaussian temporal smoothing — σ=1.5 frames, 5-tap kernel.
/// Applied per-joint across a sliding window of 5 consecutive frames.
/// Stack-allocated output — no heap.
pub fn gaussian_smooth<const N: usize>(
    frames: &[[Joint; N]; 5],
) -> [Joint; N] {
    // Gaussian kernel σ=1.5, 5 taps, normalised
    const KERNEL: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let mut out = [Joint::default(); N];
    for j in 0..N {
        let mut sx = 0.0_f32;
        let mut sy = 0.0_f32;
        let mut sz = 0.0_f32;
        for (k, &w) in KERNEL.iter().enumerate() {
            sx += frames[k][j].x * w;
            sy += frames[k][j].y * w;
            sz += frames[k][j].z * w;
        }
        out[j] = Joint::new(sx, sy, sz);
    }
    out
}

// ── H_sp — Spatial Entropy ─────────────────────────────────────────────────
//
// Language: Rust
// Why: HOJ3D sector assignment runs for every joint every frame.
//      The sector_index() computation uses atan2 — expensive in Python,
//      optimised by LLVM in Rust to near-SIMD throughput.

/// Compute H_sp (spatial entropy) from a normalised frame.
///
/// Algorithm:
///   1. Assign each joint to one of K=12 spherical sectors
///   2. Count joints per sector → occupancy histogram
///   3. Normalise histogram → probability distribution
///   4. Compute Shannon entropy
///
/// Returns H_sp in [0, log2(K)].
#[inline]
pub fn compute_hsp<const N: usize>(normalised_joints: &[Joint; N]) -> f32 {
    // Stack-allocated histogram — no heap
    let mut histogram = [0u32; K_SECTORS];
    for &joint in normalised_joints.iter() {
        let sector = joint.sector_index();
        histogram[sector] += 1;
    }
    let total = N as f32;
    // Normalise to probability distribution
    let distribution: [f32; K_SECTORS] = histogram.map(|c| c as f32 / total);
    shannon_entropy(&distribution)
}

// ── H_te — Temporal Entropy ────────────────────────────────────────────────
//
// Language: Rust
// Why: Velocity histogram construction for 25 joints × 16 bins × 30fps.
//      Each bin boundary lookup and count increment must be cache-friendly.
//      Rust's [f32; N] arrays sit contiguously on the stack.

/// Compute H_te (temporal entropy) from two consecutive normalised frames.
///
/// Algorithm:
///   1. Compute velocity magnitude for each joint (frame-to-frame distance)
///   2. Build 16-bin histogram of velocity magnitudes per joint
///   3. Normalise each histogram
///   4. Compute Shannon entropy per joint, average across all joints
///
/// `vel_max` is the 99th-percentile velocity from the training split
/// (used to set histogram bin boundaries). Default: 0.5 m/frame.
#[inline]
pub fn compute_hte<const N: usize>(
    prev_frame: &[Joint; N],
    curr_frame: &[Joint; N],
    vel_max: f32,
) -> f32 {
    let vel_max = if vel_max <= 0.0 { 0.5 } else { vel_max };
    let bin_width = vel_max / N_VEL_BINS as f32;

    let mut total_entropy = 0.0_f32;

    for j in 0..N {
        let vel = prev_frame[j].velocity_to(curr_frame[j]);
        let bin = ((vel / bin_width) as usize).min(N_VEL_BINS - 1);

        // Single-sample histogram — this joint contributes to bin `bin`
        // For a proper histogram, accumulate across a window.
        // Here we use a per-frame approximation: one-hot encoding.
        let mut hist = [0.0_f32; N_VEL_BINS];
        hist[bin] = 1.0;
        total_entropy += shannon_entropy(&hist);
    }

    total_entropy / N as f32
}

/// Compute H_te from a window of frames for richer histograms.
/// Uses the last `WINDOW` frames to build velocity distribution per joint.
/// This is closer to the paper's specification.
pub fn compute_hte_windowed<const N: usize, const WINDOW: usize>(
    frames: &[[Joint; N]; WINDOW],
    vel_max: f32,
) -> f32 {
    let vel_max = if vel_max <= 0.0 { 0.5 } else { vel_max };
    let bin_width = vel_max / N_VEL_BINS as f32;
    let mut total_entropy = 0.0_f32;

    // `j` indexes the joint dimension while iterating the frame (WINDOW)
    // dimension in the inner loop — needed to read frames[w-1][j] and
    // frames[w][j] together, which a single-array iterator can't express.
    #[allow(clippy::needless_range_loop)]
    for j in 0..N {
        let mut hist = [0u32; N_VEL_BINS];
        for w in 1..WINDOW {
            let vel = frames[w-1][j].velocity_to(frames[w][j]);
            let bin = ((vel / bin_width) as usize).min(N_VEL_BINS - 1);
            hist[bin] += 1;
        }
        let total = (WINDOW - 1) as f32;
        let distribution: [f32; N_VEL_BINS] = hist.map(|c| c as f32 / total);
        total_entropy += shannon_entropy(&distribution);
    }

    total_entropy / N as f32
}

// ── H_in — Inter-joint Entropy ─────────────────────────────────────────────
//
// Language: Rust  ← THIS is why we needed Rust.
// Why: N=25 joints → C(25,2) = 300 pairs.
//      Each pair requires a 16×16=256-cell histogram.
//      300 × 256 = 76,800 cells to update per frame × 30fps = 2.3M ops/s.
//      Python interpreter overhead makes this impossible at 3.8ms target.
//      Rust compiles the inner loop to ~4 instructions per cell update.

/// Compute H_in (inter-joint entropy) from two consecutive normalised frames.
///
/// Algorithm:
///   1. For every pair of joints (a, b):
///      a. Compute velocity of joint a → bin_a (0..16)
///      b. Compute velocity of joint b → bin_b (0..16)
///      c. Increment co-movement histogram[bin_a][bin_b]
///   2. L1-normalise each 16×16 histogram
///   3. Compute Shannon entropy of each histogram
///   4. Average across all C(N,2) pairs
///
/// Stack allocates one [u32; 256] histogram at a time — total stack usage
/// is constant regardless of N.
pub fn compute_hin<const N: usize>(
    prev_frame: &[Joint; N],
    curr_frame: &[Joint; N],
    vel_max: f32,
) -> f32 {
    let vel_max = if vel_max <= 0.0 { 0.5 } else { vel_max };
    let bin_width = vel_max / N_VEL_BINS as f32;

    // Precompute velocity bins for all joints — one pass over joints
    let mut bins = [0usize; N];
    for j in 0..N {
        let vel = prev_frame[j].velocity_to(curr_frame[j]);
        bins[j] = ((vel / bin_width) as usize).min(N_VEL_BINS - 1);
    }

    let mut total_entropy = 0.0_f32;
    let mut pair_count = 0usize;

    // C(N,2) pairs — tight inner loop, LLVM can auto-vectorise
    for a in 0..N {
        for b in (a + 1)..N {
            // 16×16 co-movement histogram — stack allocated
            let mut hist = [0u32; N_VEL_BINS * N_VEL_BINS];
            let idx = bins[a] * N_VEL_BINS + bins[b];
            hist[idx] += 1;

            // L1-normalise (sum is always 1 for single sample)
            let distribution: [f32; N_VEL_BINS * N_VEL_BINS] =
                hist.map(|c| c as f32);
            total_entropy += shannon_entropy(&distribution);
            pair_count += 1;
        }
    }

    if pair_count == 0 { 0.0 } else { total_entropy / pair_count as f32 }
}

/// Windowed H_in — accumulates histogram across W consecutive frame pairs.
/// More statistically robust, closer to the paper's specification.
pub fn compute_hin_windowed<const N: usize, const WINDOW: usize>(
    frames: &[[Joint; N]; WINDOW],
    vel_max: f32,
) -> f32 {
    let vel_max = if vel_max <= 0.0 { 0.5 } else { vel_max };
    let bin_width = vel_max / N_VEL_BINS as f32;

    let mut total_entropy = 0.0_f32;
    let mut pair_count    = 0usize;

    for a in 0..N {
        for b in (a + 1)..N {
            let mut hist = [0u32; N_VEL_BINS * N_VEL_BINS];

            for w in 1..WINDOW {
                let vel_a = frames[w-1][a].velocity_to(frames[w][a]);
                let vel_b = frames[w-1][b].velocity_to(frames[w][b]);
                let bin_a = ((vel_a / bin_width) as usize).min(N_VEL_BINS - 1);
                let bin_b = ((vel_b / bin_width) as usize).min(N_VEL_BINS - 1);
                hist[bin_a * N_VEL_BINS + bin_b] += 1;
            }

            let total = (WINDOW - 1) as f32;
            let distribution: [f32; N_VEL_BINS * N_VEL_BINS] =
                hist.map(|c| c as f32 / total);
            total_entropy += shannon_entropy(&distribution);
            pair_count += 1;
        }
    }

    if pair_count == 0 { 0.0 } else { total_entropy / pair_count as f32 }
}

// ── HemmWindow — stateful sliding-history buffer ────────────────────────────
//
// Language: Rust
// Why this exists: `compute_hte` and `compute_hin` above build a histogram
// from a SINGLE prev/curr velocity sample. A histogram built from exactly one
// sample is always one-hot (one cell = 1, the rest = 0), and the Shannon
// entropy of a one-hot distribution is mathematically always 0, regardless of
// the input data. This was discovered during a full-repository audit: both
// the Rust and Python single-frame H_te/H_in paths were returning ≈0.0 for
// every input, silently, because the math is degenerate — not because the
// implementation was "wrong" in the sense of a typo, but because a
// one-sample histogram cannot carry entropy.
//
// `HemmWindow` fixes this the same way `compute_hte_windowed` /
// `compute_hin_windowed` already did above (those functions were already
// correct but were never wired into the live pipeline): it accumulates a
// short rolling history of CAP frames and computes entropy over the
// resulting multi-sample distribution. Unlike the fixed-size `_windowed`
// helpers, it works with a *partial* window too (2..CAP frames), so the very
// first few frames after start-up still produce a real, if noisier,
// estimate rather than requiring CAP full frames before any output exists.
//
// CAP = 5 matches the windowed H_te design already documented as a specific,
// named commitment in the thesis (Chapter 5, recommended milestone 3).
// Applying the same CAP to H_in as well is the correction this audit found
// necessary: H_in had the identical one-hot degeneracy as H_te, not just an
// approximation gap in firmware.
#[derive(Clone, Copy)]
pub struct HemmWindow<const N: usize, const CAP: usize> {
    frames: [[Joint; N]; CAP],
    /// Number of valid frames currently held (0..=CAP).
    len: usize,
    /// Ring index of the next slot to write.
    head: usize,
}

impl<const N: usize, const CAP: usize> Default for HemmWindow<N, CAP> {
    fn default() -> Self {
        Self {
            frames: [[Joint::default(); N]; CAP],
            len: 0,
            head: 0,
        }
    }
}

impl<const N: usize, const CAP: usize> HemmWindow<N, CAP> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new frame into the window, evicting the oldest frame once full.
    pub fn push(&mut self, frame: &[Joint; N]) {
        self.frames[self.head] = *frame;
        self.head = (self.head + 1) % CAP;
        if self.len < CAP {
            self.len += 1;
        }
    }

    /// True once the window holds a full CAP frames of history.
    pub fn is_warm(&self) -> bool {
        self.len == CAP
    }

    /// Number of frames currently buffered (0..=CAP).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return buffered frames in chronological order (oldest → newest).
    fn ordered(&self) -> [[Joint; N]; CAP] {
        let mut out = [[Joint::default(); N]; CAP];
        let start = if self.len < CAP { 0 } else { self.head };
        for (i, slot) in out.iter_mut().enumerate().take(self.len) {
            *slot = self.frames[(start + i) % CAP];
        }
        out
    }

    /// Most recently pushed frame, if any.
    pub fn latest(&self) -> Option<[Joint; N]> {
        if self.len == 0 {
            return None;
        }
        let idx = (self.head + CAP - 1) % CAP;
        Some(self.frames[idx])
    }

    /// H_te computed over whatever history is currently buffered
    /// (2..=CAP frames → 1..=CAP-1 velocity samples per joint).
    /// Returns 0.0 only when fewer than 2 frames have been pushed —
    /// an honest "not enough data yet" state, not a hidden math bug.
    pub fn compute_hte(&self, vel_max: f32) -> f32 {
        if self.len < 2 {
            return 0.0;
        }
        let vel_max = if vel_max <= 0.0 { 0.5 } else { vel_max };
        let bin_width = vel_max / N_VEL_BINS as f32;
        let ordered = self.ordered();
        let n_pairs = self.len - 1;

        let mut total_entropy = 0.0_f32;
        // Same reasoning as compute_hte_windowed above: `j` indexes the
        // joint dimension while `w` (inner loop) indexes the frame
        // dimension — needed for parallel access, not a plain single-array
        // iteration clippy's suggested rewrite would simplify correctly.
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            let mut hist = [0u32; N_VEL_BINS];
            for w in 1..self.len {
                let vel = ordered[w - 1][j].velocity_to(ordered[w][j]);
                let bin = ((vel / bin_width) as usize).min(N_VEL_BINS - 1);
                hist[bin] += 1;
            }
            let distribution: [f32; N_VEL_BINS] =
                hist.map(|c| c as f32 / n_pairs as f32);
            total_entropy += shannon_entropy(&distribution);
        }
        total_entropy / N as f32
    }

    /// H_in computed over whatever history is currently buffered.
    /// Same "honest partial window" behaviour as `compute_hte`.
    pub fn compute_hin(&self, vel_max: f32) -> f32 {
        if self.len < 2 {
            return 0.0;
        }
        let vel_max = if vel_max <= 0.0 { 0.5 } else { vel_max };
        let bin_width = vel_max / N_VEL_BINS as f32;
        let ordered = self.ordered();
        let n_pairs = self.len - 1;

        let mut total_entropy = 0.0_f32;
        let mut pair_count = 0usize;
        for a in 0..N {
            for b in (a + 1)..N {
                let mut hist = [0u32; N_VEL_BINS * N_VEL_BINS];
                for w in 1..self.len {
                    let vel_a = ordered[w - 1][a].velocity_to(ordered[w][a]);
                    let vel_b = ordered[w - 1][b].velocity_to(ordered[w][b]);
                    let bin_a = ((vel_a / bin_width) as usize).min(N_VEL_BINS - 1);
                    let bin_b = ((vel_b / bin_width) as usize).min(N_VEL_BINS - 1);
                    hist[bin_a * N_VEL_BINS + bin_b] += 1;
                }
                let distribution: [f32; N_VEL_BINS * N_VEL_BINS] =
                    hist.map(|c| c as f32 / n_pairs as f32);
                total_entropy += shannon_entropy(&distribution);
                pair_count += 1;
            }
        }
        if pair_count == 0 { 0.0 } else { total_entropy / pair_count as f32 }
    }

    /// Full feature vector using the latest frame for H_sp and the
    /// buffered window for H_te / H_in. Returns None until at least
    /// 2 frames have been pushed.
    pub fn compute_features(&self, vel_max: f32) -> Option<EntropyFeatures> {
        let latest = self.latest()?;
        if self.len < 2 {
            return None;
        }
        Some(EntropyFeatures {
            h_sp: compute_hsp(&latest),
            h_te: self.compute_hte(vel_max),
            h_in: self.compute_hin(vel_max),
        })
    }
}

// ── Combined feature vector ────────────────────────────────────────────────

/// The HEMM feature vector φ_t = [H_sp, H_te, H_in].
/// Three f32 values — 12 bytes total. Passed by value, lives on stack.
#[derive(Clone, Copy, Debug, Default)]
pub struct EntropyFeatures {
    pub h_sp: f32,
    pub h_te: f32,
    pub h_in: f32,
}

impl EntropyFeatures {
    pub fn new(h_sp: f32, h_te: f32, h_in: f32) -> Self {
        Self { h_sp, h_te, h_in }
    }

    /// Compute all three descriptors from two consecutive frames.
    ///
    /// ⚠ WARNING — single-sample H_te / H_in are mathematically degenerate.
    /// A histogram built from exactly one velocity sample per (joint /
    /// joint-pair) is always one-hot, and the Shannon entropy of a one-hot
    /// distribution is always ≈0.0 regardless of input. This function is
    /// retained ONLY for backward API compatibility with callers that
    /// still pass a bare (prev, curr) pair. H_sp is unaffected (it
    /// aggregates over 25 joints within a single frame, not over time) and
    /// remains meaningful here.
    ///
    /// For a real H_te / H_in estimate, use `HemmWindow` (or
    /// `compute_hte_windowed` / `compute_hin_windowed` directly), which
    /// accumulate a short multi-frame history before computing entropy —
    /// this is what `Pipeline25` uses internally.
    #[deprecated(
        note = "H_te and H_in from a single (prev, curr) pair are always ≈0.0 by construction (one-hot histogram ⇒ zero entropy). Use HemmWindow::compute_features instead."
    )]
    pub fn compute<const N: usize>(
        prev: &[Joint; N],
        curr: &[Joint; N],
        vel_max: f32,
    ) -> Self {
        Self {
            h_sp: compute_hsp(curr),
            h_te: compute_hte(prev, curr, vel_max),
            h_in: compute_hin(prev, curr, vel_max),
        }
    }

    /// Returns features as a [f32; 3] array for BNN input.
    #[inline]
    pub fn as_array(&self) -> [f32; 3] {
        [self.h_sp, self.h_te, self.h_in]
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    fn make_joints<const N: usize>(val: f32) -> [Joint; N] {
        [Joint::new(val, val, val); N]
    }

    #[test]
    fn shannon_entropy_uniform_max() {
        // Uniform distribution → maximum entropy = log2(K)
        let uniform = [1.0_f32 / K_SECTORS as f32; K_SECTORS];
        let h = shannon_entropy(&uniform);
        let expected = (K_SECTORS as f32).log2();
        assert!((h - expected).abs() < 0.001, "uniform entropy = {h}, expected {expected}");
    }

    #[test]
    fn shannon_entropy_concentrated_zero() {
        // All mass on one bin → entropy near 0
        let mut concentrated = [0.0_f32; K_SECTORS];
        concentrated[0] = 1.0;
        let h = shannon_entropy(&concentrated);
        assert!(h.abs() < 0.01, "concentrated entropy should be ~0, got {h}");
    }

    #[test]
    fn hsp_returns_valid_range() {
        let joints = make_joints::<25>(0.3);
        let h = compute_hsp(&joints);
        assert!(h >= 0.0, "H_sp must be non-negative");
        assert!(h <= (K_SECTORS as f32).log2() + 0.001, "H_sp must be <= log2(K)");
    }

    /// REGRESSION TEST — documents the single-frame degeneracy on purpose.
    /// `compute_hte` / `compute_hin` build a histogram from exactly one
    /// velocity sample, which is always one-hot, whose Shannon entropy is
    /// always ≈0.0 by definition — regardless of how different `prev` and
    /// `curr` are. This test locks in that *known, documented* behaviour of
    /// the deprecated single-frame path so nobody mistakes it for a live
    /// bug again, and so nobody accidentally starts relying on this path
    /// for a real entropy value. Real callers must use `HemmWindow`.
    #[test]
    fn single_frame_hte_and_hin_are_degenerate_by_construction() {
        let prev = make_joints::<25>(0.0);
        let curr = make_joints::<25>(0.9); // large, obvious motion
        let hte = compute_hte(&prev, &curr, 0.5);
        let hin = compute_hin(&prev, &curr, 0.5);
        assert!(hte.abs() < 1e-6, "single-frame H_te should be ≈0 (one-hot), got {hte}");
        assert!(hin.abs() < 1e-6, "single-frame H_in should be ≈0 (one-hot), got {hin}");
    }

    #[test]
    fn hte_returns_valid_range() {
        let prev = make_joints::<25>(0.1);
        let curr = make_joints::<25>(0.2);
        let h = compute_hte(&prev, &curr, 0.5);
        assert!(h >= 0.0);
    }

    #[test]
    fn hin_returns_valid_range() {
        let prev = make_joints::<25>(0.0);
        let curr = make_joints::<25>(0.1);
        let h = compute_hin(&prev, &curr, 0.5);
        assert!(h >= 0.0);
    }

    #[test]
    fn entropy_features_compute() {
        let prev = make_joints::<25>(0.0);
        let curr = make_joints::<25>(0.15);
        let f = EntropyFeatures::compute(&prev, &curr, 0.5);
        let arr = f.as_array();
        assert_eq!(arr.len(), 3);
        assert!(arr.iter().all(|&v| v >= 0.0));
    }

    // ── HemmWindow — the actual fix ─────────────────────────────────────

    fn varied_frame<const N: usize>(seed: u32) -> [Joint; N] {
        let mut s = seed;
        core::array::from_fn(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let a = (s as f32 / u32::MAX as f32) * 0.4 - 0.2;
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let b = (s as f32 / u32::MAX as f32) * 0.4 - 0.2;
            let _ = i;
            Joint::new(a, b, a - b)
        })
    }

    #[test]
    fn windowed_hte_is_not_degenerate() {
        let mut win: HemmWindow<25, 5> = HemmWindow::new();
        for i in 0..5u32 {
            win.push(&varied_frame(100 + i));
        }
        assert!(win.is_warm());
        let hte = win.compute_hte(0.5);
        assert!(hte > 0.05, "windowed H_te should be meaningfully non-zero for varied motion, got {hte}");
    }

    #[test]
    fn windowed_hin_is_not_degenerate() {
        let mut win: HemmWindow<25, 5> = HemmWindow::new();
        for i in 0..5u32 {
            win.push(&varied_frame(200 + i));
        }
        assert!(win.is_warm());
        let hin = win.compute_hin(0.5);
        assert!(hin > 0.05, "windowed H_in should be meaningfully non-zero for varied motion, got {hin}");
    }

    #[test]
    fn windowed_hte_zero_for_static_joints() {
        // Sanity check: a genuinely static skeleton (no motion at all)
        // should still legitimately produce near-zero H_te — this confirms
        // the windowed path tracks real motion rather than always
        // returning a nonzero "floor" value irrespective of input.
        let mut win: HemmWindow<25, 5> = HemmWindow::new();
        let still = make_joints::<25>(0.3);
        for _ in 0..5 {
            win.push(&still);
        }
        let hte = win.compute_hte(0.5);
        assert!(hte.abs() < 1e-6, "H_te for a static skeleton should be ≈0, got {hte}");
    }

    #[test]
    fn hemm_window_partial_warmup_still_produces_estimate() {
        // Only 2 frames pushed (not yet full CAP=5) — should still return
        // a real, if noisier, estimate rather than requiring a full window.
        let mut win: HemmWindow<25, 5> = HemmWindow::new();
        assert!(win.compute_features(0.5).is_none(), "no frames yet -> None");
        win.push(&varied_frame(1));
        assert!(win.compute_features(0.5).is_none(), "1 frame -> not enough for velocity yet");
        win.push(&varied_frame(2));
        assert!(!win.is_warm(), "2 frames is not yet a full 5-frame window");
        let features = win.compute_features(0.5).expect("2 frames should be enough for a partial estimate");
        assert!(features.h_te >= 0.0);
        assert!(features.h_in >= 0.0);
    }

    #[test]
    fn hemm_window_evicts_oldest_frame() {
        let mut win: HemmWindow<25, 3> = HemmWindow::new();
        for i in 0..3u32 {
            win.push(&varied_frame(i));
        }
        assert_eq!(win.len(), 3);
        win.push(&varied_frame(99));
        assert_eq!(win.len(), 3, "CAP=3 window should stay at 3, not grow");
    }

    #[test]
    fn windowed_hte_matches_fixed_size_helper() {
        // HemmWindow's variable-length path should agree with the original
        // fixed-size compute_hte_windowed when the window is fully warm.
        let frames: [[Joint; 25]; 5] = core::array::from_fn(|i| varied_frame(500 + i as u32));
        let expected = compute_hte_windowed(&frames, 0.5);

        let mut win: HemmWindow<25, 5> = HemmWindow::new();
        for f in &frames {
            win.push(f);
        }
        let actual = win.compute_hte(0.5);
        assert!((actual - expected).abs() < 1e-5,
            "HemmWindow.compute_hte = {actual}, compute_hte_windowed = {expected}");
    }

    #[test]
    fn windowed_hin_matches_fixed_size_helper() {
        let frames: [[Joint; 25]; 5] = core::array::from_fn(|i| varied_frame(600 + i as u32));
        let expected = compute_hin_windowed(&frames, 0.5);

        let mut win: HemmWindow<25, 5> = HemmWindow::new();
        for f in &frames {
            win.push(f);
        }
        let actual = win.compute_hin(0.5);
        assert!((actual - expected).abs() < 1e-5,
            "HemmWindow.compute_hin = {actual}, compute_hin_windowed = {expected}");
    }

    #[test]
    fn normalise_removes_spine_offset() {
        let mut joints = make_joints::<25>(1.0);
        joints[0] = Joint::new(1.0, 2.0, 3.0); // spine base
        let norm = normalise_frame(&joints);
        // spine base should now be at origin
        assert!((norm[0].x).abs() < 1e-6);
        assert!((norm[0].y).abs() < 1e-6);
        assert!((norm[0].z).abs() < 1e-6);
    }

    #[test]
    fn joint_distance() {
        let a = Joint::new(0.0, 0.0, 0.0);
        let b = Joint::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-5);
    }
}
