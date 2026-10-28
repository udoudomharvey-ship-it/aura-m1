//! aura_core/tests/hemm_golden_vectors.rs
//!
//! Language: Rust (integration test)
//! Why: Cross-language parity regression protection for H_sp / H_te / H_in.
//!
//! Loads the SAME `tests/golden_vectors/hemm_golden_vectors.json` file that
//! `aura_sdk_tests/test_hemm_golden_vectors.py` checks against, and verifies
//! `aura_core::hemm::HemmWindow` reproduces the Python reference
//! implementation's values within 1e-4. If both this test and the Python
//! test pass, Python and Rust are verified numerically at parity for
//! H_sp/H_te/H_in specifically — not just for final QHCI, which is what the
//! original audit found had silently masked the bug.
//!
//! Run: cargo test --package aura_core --test hemm_golden_vectors

use aura_core::hemm::{Joint, HemmWindow, compute_hsp};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

const TOLERANCE: f32 = 1e-4;

fn golden_path() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../aura_core, so go up one level to the repo root.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.push("tests");
    p.push("golden_vectors");
    p.push("hemm_golden_vectors.json");
    p
}

fn load_golden() -> Value {
    let text = fs::read_to_string(golden_path())
        .expect("hemm_golden_vectors.json not found — run \
                 tests/golden_vectors/generate_hemm_golden_vectors.py first");
    serde_json::from_str(&text).expect("golden vector file is not valid JSON")
}

fn parse_frame(frame_json: &Value) -> [Joint; 25] {
    let arr = frame_json.as_array().expect("frame should be a JSON array");
    assert_eq!(arr.len(), 25, "expected 25 joints per frame");
    let mut joints = [Joint::new(0.0, 0.0, 0.0); 25];
    for (i, j) in arr.iter().enumerate() {
        let triple = j.as_array().expect("joint should be [x,y,z]");
        let x = triple[0].as_f64().unwrap() as f32;
        let y = triple[1].as_f64().unwrap() as f32;
        let z = triple[2].as_f64().unwrap() as f32;
        joints[i] = Joint::new(x, y, z);
    }
    joints
}

#[test]
fn hemm_golden_vectors_match_python_reference() {
    let golden = load_golden();
    let cases = golden["cases"].as_array().expect("cases should be an array");
    assert!(!cases.is_empty(), "golden vector file has no cases");

    let mut failures = Vec::new();

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let vel_max = case["vel_max"].as_f64().unwrap() as f32;
        let frames_json = case["frames"].as_array().expect("frames should be an array");

        let expected_h_sp = case["expected"]["h_sp"].as_f64().unwrap() as f32;
        let expected_h_te = case["expected"]["h_te"].as_f64().unwrap() as f32;
        let expected_h_in = case["expected"]["h_in"].as_f64().unwrap() as f32;

        // Push every frame into a fresh window, in order — mirrors exactly
        // how the Python side builds `frames` for _hte_windowed_py /
        // _hin_windowed_py (chronological order, no seeding tricks needed
        // here since we're calling the windowed functions directly with
        // the full frame list, same as the Python golden vector generator
        // does with _hte_windowed_py(frames, vel_max)).
        let mut window: HemmWindow<25, 5> = HemmWindow::new();
        let mut last_frame: Option<[Joint; 25]> = None;
        for f in frames_json {
            let joints = parse_frame(f);
            window.push(&joints);
            last_frame = Some(joints);
        }

        let actual_h_sp = compute_hsp(&last_frame.expect("at least one frame per case"));
        let actual_h_te = window.compute_hte(vel_max);
        let actual_h_in = window.compute_hin(vel_max);

        let d_sp = (actual_h_sp - expected_h_sp).abs();
        let d_te = (actual_h_te - expected_h_te).abs();
        let d_in = (actual_h_in - expected_h_in).abs();

        if d_sp > TOLERANCE || d_te > TOLERANCE || d_in > TOLERANCE {
            failures.push(format!(
                "[{name}] H_sp: rust={actual_h_sp:.6} py={expected_h_sp:.6} diff={d_sp:.6} | \
                 H_te: rust={actual_h_te:.6} py={expected_h_te:.6} diff={d_te:.6} | \
                 H_in: rust={actual_h_in:.6} py={expected_h_in:.6} diff={d_in:.6}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "Python/Rust HEMM parity FAILED for {} case(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Explicit regression guard mirroring the Python side: the audit's original
/// bug made H_te/H_in silently ≈0.0 for every input. The full-window golden
/// case must not reproduce that degeneracy in Rust either.
#[test]
fn golden_full_window_case_is_not_degenerate_in_rust() {
    let golden = load_golden();
    let cases = golden["cases"].as_array().unwrap();
    let case = cases.iter()
        .find(|c| c["name"] == "full_window_seed_100")
        .expect("full_window_seed_100 case must exist in golden vectors");

    let vel_max = case["vel_max"].as_f64().unwrap() as f32;
    let frames_json = case["frames"].as_array().unwrap();

    let mut window: HemmWindow<25, 5> = HemmWindow::new();
    for f in frames_json {
        window.push(&parse_frame(f));
    }

    let h_te = window.compute_hte(vel_max);
    let h_in = window.compute_hin(vel_max);
    assert!(h_te > 0.05, "Rust H_te on full-window golden case should be meaningfully non-zero, got {h_te}");
    assert!(h_in > 0.05, "Rust H_in on full-window golden case should be meaningfully non-zero, got {h_in}");
}
