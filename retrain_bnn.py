"""retrain_bnn.py -- fit the BNN to real labeled webcam data, with per-feature
median rescaling (written to models/entropy_scale.json). The production
_quantise() threshold (>= 0.5) is shared by Python/Rust/firmware and is NOT
changed; callers (run_live.py) apply the same rescaling before bnn.infer()."""
import csv, json, math, random, statistics, struct
from collections import Counter, defaultdict
from pathlib import Path

DATASET   = Path("live_sessions/labeled_dataset.csv")
MODEL_OUT = Path("models/aura_m1_bnn_v1.json")
SCALE_OUT = Path("models/entropy_scale.json")
N_IN, N_HIDDEN, N_OUT = 3, 16, 5
LABEL_TO_STATE = {"focused": 5, "spatial_restlessness": 1,
                  "cognitive_overload": 1, "sustained_low_quality": 1}

def quantise(x): return 1 if x >= 0.5 else -1

def forward(x_bin, w1, b1, w2, b2):
    hidden = [1 if (sum(1 if w1[j][i] == x_bin[i] else -1 for i in range(N_IN)) + b1[j]) >= 0 else -1
              for j in range(N_HIDDEN)]
    logits = [sum(1 if w2[k][j] == hidden[j] else -1 for j in range(N_HIDDEN)) + b2[k]
              for k in range(N_OUT)]
    m = max(logits); ex = [math.exp(l - m) for l in logits]; s = sum(ex)
    probs = [e / s for e in ex]
    q_raw = sum((i + 1) * p for i, p in enumerate(probs)) - 1
    return probs.index(max(probs)) + 1, max(0.0, min(1.0, q_raw / (N_OUT - 1)))

def fnv1a_32(data):
    h = 0x811C9DC5
    for b in data:
        h ^= b; h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def canon_bytes(w1, b1, w2, b2):
    buf = bytearray()
    for row in w1:
        for v in row: buf.append(0x01 if v == 1 else 0xFF)
    for v in b1: buf += struct.pack("<f", v)
    for row in w2:
        for v in row: buf.append(0x01 if v == 1 else 0xFF)
    for v in b2: buf += struct.pack("<f", v)
    return bytes(buf)

def rescale(sp, te, inn, sc):
    return (sp / (2 * sc["h_sp_median"]), te / (2 * sc["h_te_median"]),
            inn / (2 * sc["h_in_median"]))

def main():
    if not DATASET.exists():
        print(f"No dataset at {DATASET}. Run export_training_data.py first."); return
    rows = list(csv.DictReader(open(DATASET, newline="")))
    if not rows:
        print("Dataset empty."); return

    sc = {f"{k}_median": statistics.median(float(r[k]) for r in rows) or 1.0
          for k in ("h_sp", "h_te", "h_in")}
    print(f"Rescaling from {len(rows)} real frames: " +
          "  ".join(f"{k}={v:.3f}" for k, v in sc.items()) + "  (each -> 0.5)\n")

    def code_of(r):
        return tuple(quantise(x) for x in rescale(float(r["h_sp"]), float(r["h_te"]), float(r["h_in"]), sc))

    counts = defaultdict(Counter)
    for r in rows: counts[code_of(r)][r["label"]] += 1
    print(f"{len(counts)} distinct patterns after rescaling:\n")
    targets = {}
    for code, c in counts.items():
        lab, n = c.most_common(1)[0]
        targets[code] = LABEL_TO_STATE.get(lab, 3)
        flag = "" if n == sum(c.values()) else "  <- mixed, used majority"
        print(f"  {code}: {dict(c)} -> state {targets[code]}{flag}")

    def loss(w1, b1, w2, b2):
        return sum((forward(c, w1, b1, w2, b2)[1] - (t - 1) / (N_OUT - 1)) ** 2
                   for c, t in targets.items())

    rng = random.Random(42); best, best_loss = None, float("inf")
    for _ in range(600):
        w1 = [[rng.choice([-1, 1]) for _ in range(N_IN)] for _ in range(N_HIDDEN)]
        b1 = [rng.uniform(-2, 2) for _ in range(N_HIDDEN)]
        w2 = [[rng.choice([-1, 1]) for _ in range(N_HIDDEN)] for _ in range(N_OUT)]
        b2 = [rng.uniform(-2, 2) for _ in range(N_OUT)]
        cur = loss(w1, b1, w2, b2)
        for _ in range(3000):
            ch = rng.random()
            n1, nb1, n2, nb2 = [r[:] for r in w1], b1[:], [r[:] for r in w2], b2[:]
            if ch < 0.3:   n1[rng.randrange(N_HIDDEN)][rng.randrange(N_IN)] *= -1
            elif ch < 0.5: nb1[rng.randrange(N_HIDDEN)] += rng.gauss(0, 0.3)
            elif ch < 0.8: n2[rng.randrange(N_OUT)][rng.randrange(N_HIDDEN)] *= -1
            else:          nb2[rng.randrange(N_OUT)] += rng.gauss(0, 0.5)
            nl = loss(n1, nb1, n2, nb2)
            if nl <= cur: w1, b1, w2, b2, cur = n1, nb1, n2, nb2, nl
            if cur == 0: break
        if cur < best_loss: best_loss, best = cur, (w1, b1, w2, b2)
        if best_loss < 1e-6: break
    w1, b1, w2, b2 = best
    print(f"\nSearch done. Final loss {best_loss:.6f}")

    ok, tot = Counter(), Counter()
    for r in rows:
        _, q = forward(code_of(r), w1, b1, w2, b2)
        tot[r["label"]] += 1
        if (q < 0.42) == (r["label"] != "focused"): ok[r["label"]] += 1
    print("\nIN-SAMPLE accuracy (alert vs no-alert; optimistic, nothing held out):")
    for l, n in tot.items(): print(f"  {l:24s} {ok[l]:5d}/{n:<5d} ({100 * ok[l] / n:.1f}%)")
    allc = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    print("Reachable states over all 8 patterns:", sorted({forward(c, w1, b1, w2, b2)[0] for c in allc}))

    ck = fnv1a_32(canon_bytes(w1, b1, w2, b2))
    model = {"model_version": "1.3.0", "architecture_version": "3x16x5-xnor-popcount-v1",
        "hemm_version": "1.0.0",
        "training_dataset_version": f"real-labeled-session-data-v2-rescaled ({len(rows)} frames, {len(counts)} patterns)",
        "qhci_mapping_version": "softmax-weighted-mean-v1", "joint_schema": "25-JOINT-V1",
        "weight_format": "i8_pm1", "bias_format": "f32",
        "n_in": N_IN, "n_hidden": N_HIDDEN, "n_out": N_OUT, "generation_seed": 42,
        "checksum_algorithm": "fnv1a_32", "checksum": f"{ck:08x}",
        "w1": w1, "b1": b1, "w2": w2, "b2": b2,
        "note": ("Fit by discrete binary-weight search on real self-labeled webcam data AFTER "
                 "per-feature median rescaling (models/entropy_scale.json). MUST be used with "
                 "that rescaling applied before bnn.infer(); single-user, small dataset, "
                 "in-sample validation only.")}
    MODEL_OUT.parent.mkdir(exist_ok=True)
    MODEL_OUT.write_text(json.dumps(model, indent=2))
    SCALE_OUT.write_text(json.dumps(sc, indent=2))
    print(f"\nWrote {MODEL_OUT} (checksum {ck:08x}) and {SCALE_OUT}")

if __name__ == "__main__":
    main()
