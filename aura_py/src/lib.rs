//! aura_py/src/lib.rs
//!
//! PyO3 bindings — exposes the Rust aura_core to Python.
//!
//! Language: Rust (binding layer)
//! Why: PyO3 generates a native .so that Python imports directly.
//!      The Python SDK calls Rust as if it were native Python code.
//!      Zero logic duplication — Python never reimplements what Rust owns.
//!
//! After building with `maturin develop` or `cargo build`:
//!   from aura_core_rs import RustBnn, RustEntropy, RustHarvester
//!
//! All function signatures are identical to the Python SDK equivalents,
//! ensuring the Python layer requires zero changes to use the Rust backend.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use aura_core::{
    hemm::{Joint, HemmWindow, compute_hsp, compute_hte, compute_hin},
    inference::{AuraBnn, BnnWeights},
    harvesting::{
        Harvester, RfHarvester, VibrationHarvester,
        ThermalHarvester, LightHarvester, FlowHarvester,
        run_session,
    },
    ncl::{Rail, th22, th12, th23, invert, encode_quality_state},
    pipeline::{Pipeline25, PipelineConfig, synthetic_session},
};

// ── Helper: Python list → [Joint; 25] ─────────────────────────────────────

fn list_to_joints(py_joints: Vec<(f32, f32, f32)>) -> PyResult<[Joint; 25]> {
    if py_joints.len() != 25 {
        return Err(PyValueError::new_err(format!(
            "Expected 25 joints, got {}", py_joints.len()
        )));
    }
    Ok(std::array::from_fn(|i| {
        Joint::new(py_joints[i].0, py_joints[i].1, py_joints[i].2)
    }))
}

// ── NCL gate bindings ──────────────────────────────────────────────────────

/// Python wrapper for NCL dual-rail state.
/// Mirrors aura_sdk.ncl.RailState API exactly.
#[pyclass(name = "RailState")]
#[derive(Clone)]
pub struct PyRailState {
    inner: Rail,
}

#[pymethods]
impl PyRailState {
    #[staticmethod]
    fn null()  -> Self { Self { inner: Rail::Null  } }
    #[staticmethod]
    fn data0() -> Self { Self { inner: Rail::Data0 } }
    #[staticmethod]
    fn data1() -> Self { Self { inner: Rail::Data1 } }

    fn is_null(&self)  -> bool { self.inner.is_null() }
    fn is_data(&self)  -> bool { self.inner.is_data() }

    fn logic_value(&self) -> Option<bool> { self.inner.logic_value() }

    #[staticmethod]
    fn from_bool(v: bool) -> Self { Self { inner: Rail::from_bool(v) } }

    #[staticmethod]
    fn idle_power_w() -> f32 { Rail::idle_power_w() }

    fn __repr__(&self) -> String { format!("{}", self.inner) }
    fn __str__(&self)  -> String { format!("{}", self.inner) }

    fn __eq__(&self, other: &Self) -> bool { self.inner == other.inner }
}

/// Python-callable NCL gate functions.
/// Usage: from aura_core_rs import ncl_th22, ncl_th12, ncl_invert
#[pyfunction]
fn ncl_th22(a: &PyRailState, b: &PyRailState) -> PyRailState {
    PyRailState { inner: th22(a.inner, b.inner) }
}

#[pyfunction]
fn ncl_th12(a: &PyRailState, b: &PyRailState) -> PyRailState {
    PyRailState { inner: th12(a.inner, b.inner) }
}

#[pyfunction]
fn ncl_th23(a: &PyRailState, b: &PyRailState, c: &PyRailState) -> PyRailState {
    PyRailState { inner: th23(a.inner, b.inner, c.inner) }
}

#[pyfunction]
fn ncl_invert(a: &PyRailState) -> PyRailState {
    PyRailState { inner: invert(a.inner) }
}

#[pyfunction]
fn ncl_encode_quality_state(q: f32) -> PyResult<u8> {
    if !(0.0..=1.0).contains(&q) {
        return Err(PyValueError::new_err("Q_HCI must be in [0.0, 1.0]"));
    }
    Ok(encode_quality_state(q))
}

// ── Entropy descriptor bindings ────────────────────────────────────────────

/// Rust-accelerated HEMM entropy computation.
/// Drop-in replacement for aura_sdk.hemm Python module.
///
/// Python usage:
///   from aura_core_rs import RustEntropy
///   e = RustEntropy()
///   features = e.compute(prev_joints, curr_joints, vel_max=0.5)
///   print(features["h_sp"], features["h_te"], features["h_in"])
///
/// NOTE ON STATE: `compute()` maintains an internal rolling 5-frame window
/// (see `aura_core::hemm::HemmWindow`) so that H_te / H_in are computed from
/// a real multi-sample distribution rather than a single (prev, curr) pair,
/// which is mathematically degenerate (always ≈0.0 entropy — see HemmWindow
/// doc comment in aura_core). This means a `RustEntropy` instance is
/// stateful across calls: create one instance per session/stream, the same
/// way `aura_sdk.hemm.HEMMExtractor` does, rather than a fresh instance per
/// frame. `compute_hte` / `compute_hin` below remain single-frame passthroughs
/// to the raw (deprecated, degenerate) functions for callers that explicitly
/// want that — most callers should use `compute()` instead.
#[pyclass(name = "RustEntropy")]
pub struct PyEntropy {
    window: HemmWindow<25, 5>,
    seeded: bool,
}

#[pymethods]
impl PyEntropy {
    #[new]
    fn new() -> Self {
        Self { window: HemmWindow::new(), seeded: false }
    }

    /// Compute H_sp (spatial entropy) from normalised joint positions.
    /// joints: list of 25 (x, y, z) tuples in metres
    fn compute_hsp(&self, joints: Vec<(f32, f32, f32)>) -> PyResult<f32> {
        let j = list_to_joints(joints)?;
        Ok(compute_hsp(&j))
    }

    /// Compute H_te (temporal entropy) from exactly two consecutive frames.
    /// ⚠ Single-frame passthrough — mathematically degenerate (always ≈0.0,
    /// see `RustEntropy` doc comment). Prefer `compute()` for real values.
    #[allow(deprecated)]
    fn compute_hte(
        &self,
        prev: Vec<(f32, f32, f32)>,
        curr: Vec<(f32, f32, f32)>,
        vel_max: f32,
    ) -> PyResult<f32> {
        let p = list_to_joints(prev)?;
        let c = list_to_joints(curr)?;
        Ok(compute_hte(&p, &c, vel_max))
    }

    /// Compute H_in (inter-joint entropy) from exactly two consecutive frames.
    /// ⚠ Single-frame passthrough — mathematically degenerate (always ≈0.0,
    /// see `RustEntropy` doc comment). Prefer `compute()` for real values.
    #[allow(deprecated)]
    fn compute_hin(
        &self,
        prev: Vec<(f32, f32, f32)>,
        curr: Vec<(f32, f32, f32)>,
        vel_max: f32,
    ) -> PyResult<f32> {
        let p = list_to_joints(prev)?;
        let c = list_to_joints(curr)?;
        Ok(compute_hin(&p, &c, vel_max))
    }

    /// Compute all three descriptors in one call, using the internal
    /// rolling window for H_te / H_in (see class doc comment).
    /// On the very first call for a new instance, both `prev` and `curr`
    /// are pushed into the window to seed it; every call after that only
    /// pushes `curr` (since `curr` here is `prev` on the next call, and
    /// pushing it twice would double-count that frame).
    /// Returns (h_sp, h_te, h_in).
    fn compute(
        &mut self,
        prev: Vec<(f32, f32, f32)>,
        curr: Vec<(f32, f32, f32)>,
        vel_max: f32,
    ) -> PyResult<(f32, f32, f32)> {
        let p = list_to_joints(prev)?;
        let c = list_to_joints(curr)?;
        if !self.seeded {
            self.window.push(&p);
            self.seeded = true;
        }
        self.window.push(&c);
        let f = self.window
            .compute_features(vel_max)
            .expect("window seeded above; at least 2 frames always buffered here");
        Ok((f.h_sp, f.h_te, f.h_in))
    }

    /// Reset the internal rolling window (start a fresh session/stream).
    fn reset(&mut self) {
        self.window = HemmWindow::new();
        self.seeded = false;
    }
}

// ── BNN inference bindings ─────────────────────────────────────────────────

/// Rust-accelerated BNN inference engine.
/// Drop-in replacement for aura_sdk.inference.AuraBNN.
///
/// Python usage:
///   from aura_core_rs import RustBnn
///   bnn = RustBnn()
///   result = bnn.infer(h_sp=0.42, h_te=0.58, h_in=0.31)
///   print(result["q_hci"], result["state"], result["idle_power_w"])
#[pyclass(name = "RustBnn")]
pub struct PyBnn {
    inner: AuraBnn,
}

#[pymethods]
impl PyBnn {
    /// Create BNN with seeded random weights (development/testing).
    #[new]
    fn new() -> Self {
        Self { inner: AuraBnn::seeded() }
    }

    /// Create BNN from trained weights JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        AuraBnn::from_json(json)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Run one BNN inference on entropy feature values.
    /// Returns dict matching aura_sdk.inference.InferenceResult fields.
    fn infer(&mut self, h_sp: f32, h_te: f32, h_in: f32) -> PyResult<PyObject> {
        let r = self.inner.infer(&[h_sp, h_te, h_in]);
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("q_hci",         r.q_hci)?;
            dict.set_item("state",          r.state)?;
            dict.set_item("state_label",    r.state_label())?;
            dict.set_item("energy_nj",      r.energy_nj)?;
            dict.set_item("latency_ms",     r.latency_ms)?;
            dict.set_item("idle_power_w",   r.idle_power_w)?;
            dict.set_item("workload_pct",   r.workload_pct())?;
            dict.set_item("probs",
                r.probs.iter().copied().collect::<Vec<f32>>())?;
            Ok(dict.into())
        })
    }

    /// Run inference on a sequence of (h_sp, h_te, h_in) tuples.
    /// Returns list of result dicts.
    fn infer_sequence(
        &mut self,
        sequence: Vec<(f32, f32, f32)>,
    ) -> PyResult<Vec<PyObject>> {
        let features: Vec<[f32; 3]> = sequence
            .iter()
            .map(|&(a, b, c)| [a, b, c])
            .collect();
        let mut results = Vec::new();
        self.inner.infer_sequence(&features, &mut results);
        results.iter().map(|r| {
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("q_hci",       r.q_hci)?;
                dict.set_item("state",        r.state)?;
                dict.set_item("energy_nj",    r.energy_nj)?;
                dict.set_item("idle_power_w", r.idle_power_w)?;
                Ok(dict.into())
            })
        }).collect()
    }

    fn total_energy_nj(&self) -> f32 { self.inner.total_energy_nj() }
    fn inference_count(&self) -> u64  { self.inner.inference_count() }

    fn reset_session(&mut self) { self.inner.reset_session(); }

    fn __repr__(&self) -> String {
        format!(
            "RustBnn(inferences={}, energy={:.2}nJ, idle=0.00W)",
            self.inner.inference_count(),
            self.inner.total_energy_nj()
        )
    }
}

// ── Harvester bindings ─────────────────────────────────────────────────────

/// Rust-accelerated harvesting session runner.
/// Mirrors aura_sdk.harvesting.HarvestingSession API.
///
/// Python usage:
///   from aura_core_rs import run_harvesting_session
///   result = run_harvesting_session("thermal", delta_t=3.0, duration_s=60.0)
///   print(result["pulses_fired"], result["idle_power_w"])
#[pyfunction]
fn run_harvesting_session(
    source:     &str,
    duration_s: f32,
    dt_ms:      Option<f32>,
    // Source-specific parameters
    signal_dbm:     Option<f32>,
    acceleration_g: Option<f32>,
    frequency_hz:   Option<f32>,
    delta_t:        Option<f32>,
    illuminance_lux: Option<f32>,
    cell_area_cm2:  Option<f32>,
    flow_rate_lpm:  Option<f32>,
    pressure_kpa:   Option<f32>,
) -> PyResult<PyObject> {
    let dt = dt_ms.unwrap_or(10.0);

    macro_rules! make_result {
        ($harvester:expr) => {{
            let mut h = $harvester;
            let r = run_session(&mut h, duration_s, dt);
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("source",              r.source)?;
                dict.set_item("duration_s",          r.duration_s)?;
                dict.set_item("mean_power_uw",       r.mean_power_uw)?;
                dict.set_item("total_energy_uj",     r.total_energy_uj)?;
                dict.set_item("pulses_fired",        r.pulses_fired)?;
                dict.set_item("inferences_per_sec",  r.inferences_per_sec)?;
                dict.set_item("idle_power_w",        r.idle_power_w)?;
                dict.set_item("energy_per_pulse_nj", r.energy_per_pulse_nj)?;
                Ok(dict.into())
            })
        }};
    }

    match source {
        "rf"        => make_result!(RfHarvester::new(signal_dbm.unwrap_or(-20.0))),
        "vibration" => make_result!(VibrationHarvester::new(
            acceleration_g.unwrap_or(1.0),
            frequency_hz.unwrap_or(60.0),
        )),
        "thermal"   => make_result!(ThermalHarvester::new(delta_t.unwrap_or(3.0))),
        "light"     => make_result!(LightHarvester::new(
            illuminance_lux.unwrap_or(500.0),
            cell_area_cm2.unwrap_or(1.0),
        )),
        "flow"      => make_result!(FlowHarvester::new(
            flow_rate_lpm.unwrap_or(5.0),
            pressure_kpa.unwrap_or(100.0),
        )),
        other => Err(PyValueError::new_err(format!("Unknown source: {other}"))),
    }
}

// ── Full pipeline binding ──────────────────────────────────────────────────

/// Run a complete synthetic pipeline session from Python.
/// Returns list of frame telemetry dicts.
///
/// Python usage:
///   from aura_core_rs import run_pipeline_session
///   frames = run_pipeline_session(n_frames=120, source="vibration")
///   for f in frames:
///       print(f["q_hci"], f["state"], f["idle_power_w"])
#[pyfunction]
fn run_pipeline_session(
    n_frames:       usize,
    source:         Option<&str>,
    alert_threshold: Option<f32>,
) -> PyResult<Vec<PyObject>> {
    let config = PipelineConfig {
        alert_threshold: alert_threshold.unwrap_or(0.42),
        ..Default::default()
    };

    let session = synthetic_session(n_frames);
    let mut results: Vec<PyObject> = Vec::with_capacity(n_frames);

    // Use vibration as default source for pipeline demo
    let _ = source; // extensible — currently uses vibration
    let mut pipeline = Pipeline25::new(
        AuraBnn::seeded(),
        VibrationHarvester::new(2.0, 60.0),
        config,
    );

    for (prev, curr) in &session {
        let tel = pipeline.process_frame(prev, curr);
        let obj = Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("frame",          tel.frame_idx)?;
            dict.set_item("fired",          tel.fired)?;
            dict.set_item("power_uw",       tel.power_uw)?;
            dict.set_item("accumulated_uj", tel.accumulated_uj)?;
            dict.set_item("q_hci",          tel.q_hci())?;
            dict.set_item("state",          tel.quality_state)?;
            dict.set_item("alert",          tel.alert)?;
            dict.set_item("idle_power_w",   tel.idle_power_w)?;
            Ok::<PyObject, PyErr>(dict.into())
        })?;
        results.push(obj);
    }

    Ok(results)
}

// ── Version + constants ────────────────────────────────────────────────────

#[pyfunction]
fn version() -> &'static str { aura_core::VERSION }

#[pyfunction]
fn bnn_pulse_nj() -> f32 { aura_core::BNN_PULSE_NJ }

#[pyfunction]
fn idle_power_w() -> f32 { aura_core::IDLE_POWER_W }

#[pyfunction]
fn target_latency_ms() -> f32 { aura_core::TARGET_LATENCY_MS }

// ── Module registration ────────────────────────────────────────────────────

/// Register all bindings into the Python module `aura_core_rs`.
///
/// PYO3 VERSION NOTE: this uses the `&Bound<'_, PyModule>` signature
/// required by PyO3's "Bound API" (the older `&PyModule` GIL-refs style
/// this function previously used still compiles under a `gil-refs`
/// compatibility feature in some PyO3 versions, but is not guaranteed
/// going forward). This migration was needed as part of bumping pyo3
/// 0.20 -> 0.22 to fix a hard build failure: 0.20's maximum supported
/// Python version is 3.12, and this environment's interpreter is 3.13.
/// This is the ONE change in this whole project I could not verify
/// compiles myself (no Rust toolchain in this sandbox) — if `cargo
/// check -p aura_py` fails here, paste the exact error back and it can
/// be fixed from the real compiler output rather than guessed at again.
#[pymodule]
fn aura_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyRailState>()?;
    m.add_class::<PyEntropy>()?;
    m.add_class::<PyBnn>()?;

    // NCL gate functions
    m.add_function(wrap_pyfunction!(ncl_th22, m)?)?;
    m.add_function(wrap_pyfunction!(ncl_th12, m)?)?;
    m.add_function(wrap_pyfunction!(ncl_th23, m)?)?;
    m.add_function(wrap_pyfunction!(ncl_invert, m)?)?;
    m.add_function(wrap_pyfunction!(ncl_encode_quality_state, m)?)?;

    // Harvesting
    m.add_function(wrap_pyfunction!(run_harvesting_session, m)?)?;

    // Pipeline
    m.add_function(wrap_pyfunction!(run_pipeline_session, m)?)?;

    // Constants / metadata
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(bnn_pulse_nj, m)?)?;
    m.add_function(wrap_pyfunction!(idle_power_w, m)?)?;
    m.add_function(wrap_pyfunction!(target_latency_ms, m)?)?;

    // Module-level constants
    m.add("__version__", aura_core::VERSION)?;
    m.add("IDLE_POWER_W", aura_core::IDLE_POWER_W)?;
    m.add("BNN_PULSE_NJ", aura_core::BNN_PULSE_NJ)?;

    Ok(())
}
