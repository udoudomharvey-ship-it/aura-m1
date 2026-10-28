//! aura_core/tests/bnn_golden_vectors.rs
//!
//! Language: Rust (integration test)
//! Why: cross-language parity regression protection for the BNN forward
//! pass, using the canonical model (models/aura_m1_bnn_v1.json) as the
//! fixed weight source. Companion to
//! aura_sdk_tests/test_bnn_golden_vectors.py, which checks the same JSON
//! against Python's _bnn_forward().
//!
//! Run: cargo test --package aura_core --test bnn_golden_vectors

use aura_core::inference::AuraBnn;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

const Q_TOLERANCE: f32 = 1e-4;
const LOGIT_TOLERANCE: f32 = 1e-3;

fn golden_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.push("tests");
    p.push("golden_vectors");
    p.push("bnn_golden_vectors.json");
    p
}

fn load_golden() -> Value {
    let text = fs::read_to_string(golden_path())
        .expect("bnn_golden_vectors.json not found — run \
                 tests/golden_vectors/generate_bnn_golden_vectors.py first");
    serde_json::from_str(&text).expect("golden vector file is not valid JSON")
}

#[test]
fn bnn_golden_file_matches_canonical_model_checksum() {
    // If the canonical model gets regenerated without regenerating the
    // golden vectors, this catches it before the (now stale) expected
    // values get compared against anything.
    let golden = load_golden();
    let golden_checksum = golden["model_checksum"].as_str().unwrap();

    let weights = aura_core::inference::BnnWeights::canonical();
    let computed = aura_core::inference::fnv1a_32(
        &aura_core::inference::canonical_byte_sequence(
            &weights.w1, &weights.b1, &weights.w2, &weights.b2,
        )
    );
    assert_eq!(format!("{computed:08x}"), golden_checksum,
        "bnn_golden_vectors.json was generated against a different \
         canonical model than aura_core currently loads");
}

#[test]
fn bnn_golden_vectors_match_python_reference() {
    let golden = load_golden();
    let cases = golden["cases"].as_array().expect("cases should be an array");
    assert!(!cases.is_empty(), "golden vector file has no cases");
    let alert_threshold = golden["alert_threshold"].as_f64().unwrap() as f32;

    let mut bnn = AuraBnn::from_canonical();
    let mut failures = Vec::new();

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let h_sp = case["input"]["h_sp"].as_f64().unwrap() as f32;
        let h_te = case["input"]["h_te"].as_f64().unwrap() as f32;
        let h_in = case["input"]["h_in"].as_f64().unwrap() as f32;

        let expected_state = case["expected"]["state"].as_u64().unwrap() as u8;
        let expected_q_hci = case["expected"]["q_hci"].as_f64().unwrap() as f32;
        let expected_logits: Vec<f32> = case["expected"]["logits"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let expected_alert = case["expected"]["alert"].as_bool().unwrap();

        // InferenceResult exposes softmax probabilities (`probs`), not raw
        // logits — compute the expected softmax from the golden vector's
        // raw logits here so both sides are compared on the same basis.
        let max_logit = expected_logits.iter().cloned().fold(f32::MIN, f32::max);
        let exp_logits: Vec<f32> = expected_logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let expected_probs: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

        let result = bnn.infer(&[h_sp, h_te, h_in]);

        let mut case_failures = Vec::new();
        if result.state != expected_state {
            case_failures.push(format!("state: rust={} py={}", result.state, expected_state));
        }
        if (result.q_hci - expected_q_hci).abs() > Q_TOLERANCE {
            case_failures.push(format!("q_hci: rust={:.6} py={:.6}", result.q_hci, expected_q_hci));
        }
        for (i, (&actual_p, &expected_p)) in result.probs.iter().zip(expected_probs.iter()).enumerate() {
            if (actual_p - expected_p).abs() > LOGIT_TOLERANCE {
                case_failures.push(format!("probs[{i}]: rust={actual_p:.6} py={expected_p:.6}"));
            }
        }
        let actual_alert = result.q_hci < alert_threshold;
        if actual_alert != expected_alert {
            case_failures.push(format!("alert: rust={actual_alert} py={expected_alert}"));
        }

        if !case_failures.is_empty() {
            failures.push(format!("[{name}] {}", case_failures.join(" | ")));
        }
    }

    assert!(
        failures.is_empty(),
        "Python/Rust BNN parity FAILED for {} case(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn boundary_0_5_quantises_same_as_clearly_above() {
    let golden = load_golden();
    let cases = golden["cases"].as_array().unwrap();
    let boundary = cases.iter().find(|c| c["name"] == "boundary_all_exactly_0.5").unwrap();
    let hi_cell = cases.iter().find(|c| c["name"] == "quantised_cell_7_hihihi").unwrap();

    assert_eq!(boundary["expected"]["state"], hi_cell["expected"]["state"]);
}
