"""
aura_sdk.__main__
──────────────────
CLI self-test for the hybrid Aura-M1 SDK.

    python -m aura_sdk

Verifies all four layers: NCL, compiler, BNN inference, harvesting.
Reports which backend is active (Rust or Python fallback).
"""
import sys


def main():
    import aura_sdk as sdk

    print()
    print("  GreyMatter Labs Ltd.  ·  Aura-M1 SDK  ·  v" + sdk.__version__)
    print("  'Where intelligence begins.'")
    print()
    print(f"  Backend : {'Rust (aura_core_rs)' if sdk.RUST_BACKEND else 'Python fallback (Rust not compiled)'}")
    print()

    # ── 1. NCL gate ───────────────────────────────────────────────────
    print("  [1/5] NCL gate smoke test ...")
    a   = sdk.DualRail("a", sdk.RailState.DATA1)
    b   = sdk.DualRail("b", sdk.RailState.DATA1)
    out = sdk.TH22()(a, b)
    assert out == sdk.RailState.DATA1
    a.set_null(); b.set_null()
    assert sdk.TH22()(a, b) == sdk.RailState.NULL
    print("      ✓  TH22: (1,1)→DATA1  (null,null)→NULL  idle=0.00W")

    # ── 2. Compiler ────────────────────────────────────────────────────
    print("  [2/5] Compiler smoke test ...")
    cr = sdk.compile("AND(OR(a, b), NOT(c))")
    r  = cr.run(a=True, b=False, c=False)
    assert r.output_values.get("out") is True
    print(f"      ✓  AND(OR(a,b),NOT(c)) — {cr.gate_count} gates  "
          f"{r.total_energy_pJ:.3f} pJ/event")

    # ── 3. HEMM extractor ──────────────────────────────────────────────
    print("  [3/5] HEMM entropy extractor ...")
    extractor = sdk.HEMMExtractor()
    prev = [(i * 0.01, i * 0.02, i * 0.015) for i in range(25)]
    curr = [(i * 0.01 + 0.005, i * 0.02, i * 0.015) for i in range(25)]
    f    = extractor.compute(prev, curr)
    assert f.h_sp >= 0.0 and f.h_te >= 0.0 and f.h_in >= 0.0
    print(f"      ✓  H_sp={f.h_sp:.3f}  H_te={f.h_te:.3f}  H_in={f.h_in:.3f}  "
          f"backend={extractor.backend}")

    # ── 4. BNN inference ──────────────────────────────────────────────
    print("  [4/5] BNN inference smoke test ...")
    bnn = sdk.AuraBNN()
    res = bnn.infer(H_sp=f.h_sp, H_te=f.h_te, H_in=f.h_in)
    assert 0.0 <= res.q_hci <= 1.0
    assert res.idle_power_uW == 0.0
    assert res.state in [1, 2, 3, 4, 5]
    print(f"      ✓  Q_HCI={res.q_hci:.4f}  state={res.state}  "
          f"E={res.energy_nJ:.2f}nJ  idle=0.00µW  backend={bnn.backend}")

    # ── 5. Harvesting ─────────────────────────────────────────────────
    print("  [5/5] Harvesting + pipeline smoke test ...")
    h       = sdk.ThermalHarvester(delta_T_C=4.0)
    session = sdk.HarvestingSession(h, duration_s=5.0)
    stats   = session.run()
    assert stats["idle_power_W"] == 0.0
    assert stats["total_energy_uJ"] > 0

    pipeline = sdk.make_demo_pipeline(n_frames=30)
    pipeline.sinks = []
    p_stats = pipeline.run()
    assert p_stats.total_frames == 30
    assert p_stats.idle_power_w == 0.0
    print(f"      ✓  Harvester: {stats['total_energy_uJ']:.2f}µJ  "
          f"{stats['pulses_fired']} pulses  idle=0.00W")
    print(f"      ✓  Pipeline: {p_stats.total_frames} frames  "
          f"{p_stats.fired_frames} inferences  idle=0.00W")

    print()
    print("  ✓  All checks passed. Aura-M1 SDK is operational.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
