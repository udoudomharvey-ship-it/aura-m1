//! aura_core — Aura-M1 Rust Performance Core
//!
//! GreyMatter Labs Ltd. · "Where intelligence begins."
//!
//! This crate implements every performance-critical hot path:
//!   - NCL dual-rail gate simulation
//!   - HEMM entropy descriptors (H_sp, H_te, H_in)
//!   - BNN forward pass (XNOR-popcount)
//!   - Ambient energy harvesting models
//!
//! Design rules
//! ─────────────
//!   - No heap allocation in hot paths (prefer arrays, stack)
//!   - No garbage collection (Rust ownership handles everything)
//!   - Deterministic execution (same input → same output, always)
//!   - Zero-cost abstractions (enum dispatch compiles to nothing)
//!   - All public items are #[inline] where appropriate

#![allow(dead_code)]

pub mod ncl;
pub mod hemm;
pub mod inference;
pub mod harvesting;
pub mod pipeline;

/// Aura-M1 SDK version
pub const VERSION: &str = "0.1.0";

/// BNN inference energy per pulse in nanojoules
pub const BNN_PULSE_NJ: f32 = 30.4;

/// BNN inference energy per pulse in microjoules
pub const BNN_PULSE_UJ: f32 = BNN_PULSE_NJ / 1000.0;

/// Target latency per frame in milliseconds (matching HEMM paper)
pub const TARGET_LATENCY_MS: f32 = 3.8;

/// Target active power during inference in microwatts
pub const BNN_ACTIVE_UW: f32 = 8.0;

/// Idle power — always exactly zero (NCL NULL state)
pub const IDLE_POWER_W: f32 = 0.0;
