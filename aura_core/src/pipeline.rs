//! aura_core::pipeline
//!
//! Real-time pipeline orchestrator — Rust implementation.
//!
//! Language: Rust
//! Why Rust: The pipeline runs at 30fps continuously.
//!           Frame timing, energy gating, and state management
//!           must be deterministic with zero GC pauses.
//!
//! Pipeline stages per frame:
//!   1. Receive skeleton joints from sensor
//!   2. Normalise + preprocess
//!   3. Compute entropy features [H_sp, H_te, H_in]
//!   4. Check harvester — enough energy to fire?
//!   5. If yes: run BNN inference → Q_HCI
//!   6. Encode quality state onto NCL dual-rail
//!   7. Check alert threshold
//!   8. Return telemetry

use crate::{
    hemm::{Joint, EntropyFeatures, HemmWindow},
    inference::{AuraBnn, InferenceResult},
    harvesting::Harvester,
    ncl::{Rail, encode_quality_state},
    IDLE_POWER_W,
};

/// Window size for H_te / H_in history accumulation.
/// Matches the 5-frame window documented in the thesis (Chapter 5,
/// recommended milestone 3), applied to both H_te and H_in — see
/// `hemm::HemmWindow` for why both descriptors need it, not just H_te.
pub const HEMM_WINDOW_SIZE: usize = 5;

// ── Pipeline configuration ──────────────────────────────────────────────────

/// Configuration for the Aura-M1 real-time pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Alert fires when Q_HCI drops below this threshold
    pub alert_threshold: f32,
    /// Frame timestep in milliseconds (default 33.3 = 30fps)
    pub frame_dt_ms:     f32,
    /// Maximum velocity for entropy bin boundaries (m/frame)
    pub vel_max:         f32,
    /// Number of joints expected per frame
    pub n_joints:        usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            alert_threshold: 0.42,
            frame_dt_ms:     33.3,
            vel_max:         0.5,
            n_joints:        25,
        }
    }
}

// ── Frame telemetry ─────────────────────────────────────────────────────────

/// Full telemetry output from one pipeline frame.
#[derive(Debug, Clone)]
pub struct FrameTelemetry {
    pub frame_idx:      u64,
    pub fired:          bool,
    pub power_uw:       f32,
    pub accumulated_uj: f32,
    pub features:       Option<EntropyFeatures>,
    pub inference:      Option<InferenceResult>,
    pub quality_state:  u8,
    pub quality_rail:   Rail,
    pub alert_rail:     Rail,
    pub alert:          bool,
    pub idle_power_w:   f32,
}

impl FrameTelemetry {
    /// Q_HCI value or 0.0 if no inference this frame.
    pub fn q_hci(&self) -> f32 {
        self.inference.map(|r| r.q_hci).unwrap_or(0.0)
    }
}

// ── Session statistics ──────────────────────────────────────────────────────

/// Aggregate statistics for a completed session.
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    pub total_frames:     u64,
    pub fired_frames:     u64,
    pub skipped_frames:   u64,
    pub fire_rate_pct:    f32,
    pub mean_q:           f32,
    pub total_energy_nj:  f32,
    pub idle_power_w:     f32,
    pub alert_count:      u64,
}

// ── Static pipeline (fixed N joints) ───────────────────────────────────────

/// Aura-M1 real-time pipeline for N=25 joints (Kinect v2 / HCI-Lab-30).
///
/// Usage
/// ─────
///   let mut pipeline = Pipeline25::new(
///       AuraBnn::seeded(),
///       VibrationHarvester::new(1.0, 60.0),
///       PipelineConfig::default(),
///   );
///
///   for (prev, curr) in frame_pairs {
///       let tel = pipeline.process_frame(&prev, &curr);
///       println!("Q_HCI = {:?}", tel.q_hci());
///   }
pub struct Pipeline25<H: Harvester> {
    bnn:         AuraBnn,
    harvester:   H,
    config:      PipelineConfig,
    frame_idx:   u64,
    // Rolling H_te / H_in history — see HemmWindow doc for why this exists.
    hemm_window: HemmWindow<25, HEMM_WINDOW_SIZE>,
    window_seeded: bool,
    // Session accumulators
    q_sum:       f32,
    fired_count: u64,
    alert_count: u64,
    total_e_nj:  f32,
}

impl<H: Harvester> Pipeline25<H> {
    /// Create a new pipeline.
    pub fn new(bnn: AuraBnn, harvester: H, config: PipelineConfig) -> Self {
        Self {
            bnn, harvester, config,
            frame_idx:   0,
            hemm_window: HemmWindow::new(),
            window_seeded: false,
            q_sum:       0.0,
            fired_count: 0,
            alert_count: 0,
            total_e_nj:  0.0,
        }
    }

    /// Process one frame through the full pipeline.
    ///
    /// Parameters
    /// ───────────
    ///   prev — normalised joints from previous frame
    ///   curr — normalised joints from current frame
    ///
    /// H_te / H_in are computed from a rolling `HEMM_WINDOW_SIZE`-frame
    /// history maintained internally (see `hemm::HemmWindow`), not from
    /// `prev`/`curr` alone — a single-sample estimate is mathematically
    /// degenerate (always ≈0.0). `prev` is only used to seed that history
    /// on the very first call; every call after that pushes `curr` onto
    /// the rolling window, since `curr` here is `prev` on the next call
    /// and pushing it twice would double-count that frame.
    ///
    /// Returns FrameTelemetry with all outputs.
    pub fn process_frame(
        &mut self,
        prev: &[Joint; 25],
        curr: &[Joint; 25],
    ) -> FrameTelemetry {
        self.frame_idx += 1;

        // Seed the rolling HEMM window on the very first call so we don't
        // throw away the caller-supplied `prev` frame; every later call
        // only pushes `curr` (see doc comment above).
        if !self.window_seeded {
            self.hemm_window.push(prev);
            self.window_seeded = true;
        }
        self.hemm_window.push(curr);

        // Stage 1: Harvest ambient energy for this frame period
        let power_uw      = self.harvester.update(self.config.frame_dt_ms);
        let accumulated_uj = self.harvester.accumulated_uj();

        // Stage 2: Check energy gate — can we fire?
        if !self.harvester.can_fire() {
            return FrameTelemetry {
                frame_idx:      self.frame_idx,
                fired:          false,
                power_uw,
                accumulated_uj,
                features:       None,
                inference:      None,
                quality_state:  0,
                quality_rail:   Rail::Null,
                alert_rail:     Rail::Null,
                alert:          false,
                idle_power_w:   IDLE_POWER_W,
            };
        }

        // Stage 3: Consume energy pulse
        self.harvester.consume_pulse();

        // Stage 4: Compute entropy features from the rolling window.
        // `compute_features` returns None only before any history exists,
        // which cannot happen here since we always seed+push above first.
        let features = self.hemm_window
            .compute_features(self.config.vel_max)
            .expect("hemm_window seeded above; at least 2 frames always buffered here");

        // Stage 5: BNN inference
        let inference = self.bnn.infer(&features.as_array());

        // Stage 6: Encode quality state onto NCL dual-rail
        let quality_state = encode_quality_state(inference.q_hci);
        let quality_rail  = Rail::Data1;  // output rail is active

        // Stage 7: Alert threshold check
        let alert       = inference.q_hci < self.config.alert_threshold;
        let alert_rail  = Rail::from_bool(alert);

        // Stage 8: Update session accumulators
        self.q_sum      += inference.q_hci;
        self.fired_count += 1;
        self.total_e_nj  += inference.energy_nj;
        if alert { self.alert_count += 1; }

        FrameTelemetry {
            frame_idx: self.frame_idx,
            fired: true,
            power_uw,
            accumulated_uj: self.harvester.accumulated_uj(),
            features:       Some(features),
            inference:      Some(inference),
            quality_state,
            quality_rail,
            alert_rail,
            alert,
            idle_power_w:   IDLE_POWER_W,
        }
    }

    /// Session statistics accumulated across all processed frames.
    pub fn session_stats(&self) -> SessionStats {
        let total = self.frame_idx;
        let fired = self.fired_count;
        SessionStats {
            total_frames:    total,
            fired_frames:    fired,
            skipped_frames:  total - fired,
            fire_rate_pct:   if total > 0 { fired as f32 / total as f32 * 100.0 } else { 0.0 },
            mean_q:          if fired > 0 { self.q_sum / fired as f32 } else { 0.0 },
            total_energy_nj: self.total_e_nj,
            idle_power_w:    IDLE_POWER_W,
            alert_count:     self.alert_count,
        }
    }

    /// Reset session accumulators (keep weights and config).
    pub fn reset_session(&mut self) {
        self.frame_idx   = 0;
        self.q_sum       = 0.0;
        self.fired_count = 0;
        self.alert_count = 0;
        self.total_e_nj  = 0.0;
        self.hemm_window = HemmWindow::new();
        self.window_seeded = false;
        self.bnn.reset_session();
    }
}

// ── Synthetic frame generator (for testing) ────────────────────────────────

/// Generate a synthetic skeleton frame sequence for testing.
/// Produces `n_frames` pairs of (prev, curr) joint arrays.
/// Workload rises across three task phases matching HCI-Lab-30.
pub fn synthetic_session(n_frames: usize) -> Vec<([Joint; 25], [Joint; 25])> {
    let mut frames = Vec::with_capacity(n_frames);
    let mut q: f32 = 0.68;
    // Simple LCG for deterministic pseudo-random offsets
    let mut rng_state: u32 = 42;
    let mut rng = move || -> f32 {
        rng_state = rng_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng_state as f32 / u32::MAX as f32) * 0.06 - 0.03
    };

    for i in 0..n_frames {
        let target = if i < n_frames / 3 { 0.72_f32 }
                     else if i < 2 * n_frames / 3 { 0.50 }
                     else { 0.28 };
        q = (q + (target - q) * 0.10 + rng()).clamp(0.05, 0.95);

        // Build skeleton from quality value: lower q → more spread joints
        let spread = (1.0 - q) * 0.3 + 0.05;
        let noise  = rng() * spread;
        let prev_joints: [Joint; 25] = std::array::from_fn(|j| {
            Joint::new(
                (j as f32 * 0.1 + noise) * spread,
                (j as f32 * 0.05)        * spread,
                (j as f32 * 0.08 + noise) * spread,
            )
        });
        let curr_joints: [Joint; 25] = std::array::from_fn(|j| {
            Joint::new(
                prev_joints[j].x + rng() * 0.02,
                prev_joints[j].y + rng() * 0.02,
                prev_joints[j].z + rng() * 0.02,
            )
        });
        frames.push((prev_joints, curr_joints));
    }
    frames
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inference::AuraBnn,
        harvesting::VibrationHarvester,
    };

    fn make_pipeline() -> Pipeline25<VibrationHarvester> {
        Pipeline25::new(
            AuraBnn::seeded(),
            VibrationHarvester::new(2.0, 60.0),
            PipelineConfig::default(),
        )
    }

    #[test]
    fn pipeline_runs_120_frames() {
        let mut pipeline = make_pipeline();
        let session      = synthetic_session(120);
        for (prev, curr) in &session {
            let tel = pipeline.process_frame(prev, curr);
            assert_eq!(tel.idle_power_w, 0.0);
        }
        let stats = pipeline.session_stats();
        assert_eq!(stats.total_frames, 120);
        assert!(stats.fired_frames > 0, "Should fire at least some frames");
    }

    #[test]
    fn idle_power_always_zero() {
        let mut pipeline = make_pipeline();
        let session      = synthetic_session(10);
        for (prev, curr) in &session {
            let tel = pipeline.process_frame(prev, curr);
            assert_eq!(tel.idle_power_w, 0.0);
        }
    }

    #[test]
    fn q_hci_in_range_when_fired() {
        let mut pipeline = make_pipeline();
        let session      = synthetic_session(120);
        for (prev, curr) in &session {
            let tel = pipeline.process_frame(prev, curr);
            if tel.fired {
                let q = tel.q_hci();
                assert!((0.0..=1.0).contains(&q), "Q_HCI={q} out of range");
            }
        }
    }

    #[test]
    fn quality_state_matches_q_hci() {
        let mut pipeline = make_pipeline();
        let session      = synthetic_session(60);
        for (prev, curr) in &session {
            let tel = pipeline.process_frame(prev, curr);
            if tel.fired {
                let expected_state = encode_quality_state(tel.q_hci());
                assert_eq!(tel.quality_state, expected_state);
            }
        }
    }

    #[test]
    fn alert_fires_below_threshold() {
        let mut pipeline = Pipeline25::new(
            AuraBnn::seeded(),
            VibrationHarvester::new(5.0, 120.0),
            PipelineConfig { alert_threshold: 0.99, ..Default::default() },
        );
        let session = synthetic_session(60);
        let mut any_alert = false;
        for (prev, curr) in &session {
            let tel = pipeline.process_frame(prev, curr);
            if tel.alert { any_alert = true; }
        }
        // With threshold=0.99, nearly everything should alert
        assert!(any_alert, "Expected alerts with threshold=0.99");
    }

    #[test]
    fn session_reset_clears_stats() {
        let mut pipeline = make_pipeline();
        let session      = synthetic_session(30);
        for (prev, curr) in &session {
            pipeline.process_frame(prev, curr);
        }
        pipeline.reset_session();
        let stats = pipeline.session_stats();
        assert_eq!(stats.total_frames,  0);
        assert_eq!(stats.fired_frames,  0);
        assert_eq!(stats.mean_q,        0.0);
        assert_eq!(stats.total_energy_nj, 0.0);
    }

    #[test]
    fn session_stats_fire_rate_valid() {
        let mut pipeline = make_pipeline();
        let session      = synthetic_session(120);
        for (prev, curr) in &session {
            pipeline.process_frame(prev, curr);
        }
        let stats = pipeline.session_stats();
        assert!(stats.fire_rate_pct >= 0.0);
        assert!(stats.fire_rate_pct <= 100.0);
    }

    #[test]
    fn synthetic_session_length() {
        let session = synthetic_session(120);
        assert_eq!(session.len(), 120);
    }

    /// REGRESSION TEST for the audit finding: prior to the HemmWindow fix,
    /// `Pipeline25` computed H_te/H_in from a single (prev, curr) pair via
    /// the now-deprecated `EntropyFeatures::compute`, which is mathematically
    /// guaranteed to return ≈0.0 (one-hot histogram ⇒ zero entropy) no
    /// matter how much the skeleton actually moved. This test fails loudly
    /// if that degenerate path is ever reintroduced.
    #[test]
    fn pipeline_hte_and_hin_are_not_always_zero() {
        let mut pipeline = make_pipeline();
        let session = synthetic_session(120);
        let mut any_nonzero_hte = false;
        let mut any_nonzero_hin = false;
        for (prev, curr) in &session {
            let tel = pipeline.process_frame(prev, curr);
            if let Some(f) = tel.features {
                if f.h_te > 0.01 { any_nonzero_hte = true; }
                if f.h_in > 0.01 { any_nonzero_hin = true; }
            }
        }
        assert!(any_nonzero_hte, "H_te was ≈0 across the entire session — windowed fix regressed");
        assert!(any_nonzero_hin, "H_in was ≈0 across the entire session — windowed fix regressed");
    }
}
