/// aura_core/benches/entropy_bench.rs
///
/// Criterion benchmarks for all three entropy descriptors
/// and the full pipeline. Run with:
///   cargo bench --package aura_core
///
/// Compares Rust vs estimated Python baseline timings.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use aura_core::{
    hemm::{
        Joint, EntropyFeatures, normalise_frame,
        compute_hsp, compute_hte, compute_hin,
    },
    inference::{AuraBnn, BnnWeights},
    harvesting::VibrationHarvester,
    pipeline::{Pipeline25, PipelineConfig, synthetic_session},
};

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_joints(val: f32) -> [Joint; 25] {
    std::array::from_fn(|i| Joint::new(
        val + i as f32 * 0.01,
        val + i as f32 * 0.02,
        val + i as f32 * 0.015,
    ))
}

// ── H_sp benchmark ─────────────────────────────────────────────────────────

fn bench_hsp(c: &mut Criterion) {
    let joints = make_joints(0.3);
    c.bench_function("H_sp (spatial entropy, N=25 joints)", |b| {
        b.iter(|| compute_hsp(black_box(&joints)))
    });
}

// ── H_te benchmark ─────────────────────────────────────────────────────────

fn bench_hte(c: &mut Criterion) {
    let prev = make_joints(0.1);
    let curr = make_joints(0.2);
    c.bench_function("H_te (temporal entropy, N=25 joints)", |b| {
        b.iter(|| compute_hte(black_box(&prev), black_box(&curr), 0.5))
    });
}

// ── H_in benchmark ─────────────────────────────────────────────────────────

fn bench_hin(c: &mut Criterion) {
    let prev = make_joints(0.0);
    let curr = make_joints(0.15);
    c.bench_function("H_in (inter-joint entropy, 300 pairs)", |b| {
        b.iter(|| compute_hin(black_box(&prev), black_box(&curr), 0.5))
    });
}

// ── Combined EntropyFeatures::compute ──────────────────────────────────────

fn bench_entropy_combined(c: &mut Criterion) {
    let prev = make_joints(0.1);
    let curr = make_joints(0.2);
    c.bench_function("EntropyFeatures::compute (H_sp + H_te + H_in)", |b| {
        b.iter(|| EntropyFeatures::compute(black_box(&prev), black_box(&curr), 0.5))
    });
}

// ── BNN inference benchmark ─────────────────────────────────────────────────

fn bench_bnn(c: &mut Criterion) {
    let mut bnn = AuraBnn::seeded();
    let features = [0.42_f32, 0.58, 0.31];
    c.bench_function("BNN inference (XNOR-popcount forward pass)", |b| {
        b.iter(|| bnn.infer(black_box(&features)))
    });
}

// ── Full pipeline frame benchmark ───────────────────────────────────────────

fn bench_pipeline_frame(c: &mut Criterion) {
    let mut pipeline = Pipeline25::new(
        AuraBnn::seeded(),
        VibrationHarvester::new(2.0, 60.0),
        PipelineConfig::default(),
    );
    let session = synthetic_session(2);
    let (prev, curr) = &session[0];

    c.bench_function("Full pipeline frame (entropy + BNN + NCL)", |b| {
        b.iter(|| pipeline.process_frame(black_box(prev), black_box(curr)))
    });
}

// ── N joints scaling ────────────────────────────────────────────────────────

fn bench_hin_scaling(c: &mut Criterion) {
    // Show how H_in cost scales — informational for hardware sizing
    let prev = make_joints(0.0);
    let curr = make_joints(0.1);
    let mut group = c.benchmark_group("H_in scaling");
    for label in ["N=25 (Kinect v2)"] {
        group.bench_with_input(
            BenchmarkId::new("H_in", label),
            &0.5_f32,
            |b, &vel_max| {
                b.iter(|| compute_hin(black_box(&prev), black_box(&curr), vel_max))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_hsp,
    bench_hte,
    bench_hin,
    bench_entropy_combined,
    bench_bnn,
    bench_pipeline_frame,
    bench_hin_scaling,
);
criterion_main!(benches);
