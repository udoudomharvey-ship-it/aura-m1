"""eval_session.py -- honest held-out validation of focus detection on your labeled sessions.

Usage (from the aura-m1 folder):
    python3 eval_session.py                 # uses live_sessions/
    python3 eval_session.py --dir PATH --perms 200 --no-qc

What it does
  * Cleans tracking dropouts (h_sp collapsing below 60% of the session median, +/-5 s around them).
  * Treats each contiguous label run as ONE window (frames inside a window are not independent samples).
  * Leave-one-SESSION-out: models never see the session they are scored on.
  * Compares: M1 deployed BNN (as run_live uses it), M2 pattern lookup (what retrain_bnn.py fits),
    M3 per-session-baselined 20 s rolling features + logistic regression.
  * Reports balanced accuracy (frames and windows), a window-bootstrap 95% CI, and a label-shuffle null.
  * Refuses to give a verdict when the data cannot support one, and says exactly what is missing.
Positive class = NOT focused (an alert should fire).
"""
import argparse, csv, sys
from bisect import bisect_left, bisect_right
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FEATS = ("h_sp", "h_te", "h_in")
ALERT_THRESHOLD = 0.45      # same as run_live.py
ROLL_S = 20.0               # rolling-feature window
MIN_ROLL_FRAMES = 40
MIN_GOOD_FRAMES = 30        # per label window
MAX_BAD_FRACTION = 0.20     # per label window
DROPOUT_PAD_S = 5.0


def load_sessions(d):
    sessions = {}
    for p in sorted(Path(d).glob("session_*_frames.csv")):
        rows = []
        try:
            with open(p, newline="") as f:
                rd = csv.reader(f)
                header = next(rd, None)
                if not header or "label" not in header:
                    continue
                for row in rd:
                    if len(row) != len(header):
                        continue
                    rec = dict(zip(header, row))
                    try:
                        t = float(rec["wall_time_s"])
                        x = [float(rec[k]) for k in FEATS]
                    except (KeyError, ValueError):
                        continue
                    rows.append((t, x, rec["label"]))
        except OSError:
            continue
        if rows:
            rows.sort(key=lambda r: r[0])
            sessions[p.name.replace("_frames.csv", "")] = rows
    return sessions


def mad_sigma(v):
    m = np.median(v)
    return 1.4826 * np.median(np.abs(v - m)) or 1.0


def flag_dropouts(t, hsp, pad=DROPOUT_PAD_S):
    """bad = h_sp collapsed below 60% of the session median (tracking loss drives it toward 0),
    padded in time. Calibrated on real sessions: flags 0-9% of frames, not ordinary variation."""
    med = np.median(hsp)
    raw = hsp < 0.6 * med
    bad_t = t[raw]
    if len(bad_t) == 0:
        return np.zeros(len(t), bool)
    lo = np.searchsorted(bad_t, t - pad, side="left")
    hi = np.searchsorted(bad_t, t + pad, side="right")
    return hi > lo


def rolling_features(t, X, good):
    """Per-session baseline z-score (unsupervised, no labels), then trailing 20 s mean/std. -> (n,6)"""
    n = len(t)
    Z = np.zeros_like(X)
    for j in range(3):
        v = X[good, j]
        Z[:, j] = (X[:, j] - np.median(v)) / mad_sigma(v)
    g = good.astype(float)
    out = np.full((n, 6), np.nan)
    c = np.concatenate([[0], np.cumsum(g)])
    start = np.searchsorted(t, t - ROLL_S, side="right")
    cnt = c[np.arange(1, n + 1)] - c[start]
    ok = cnt >= MIN_ROLL_FRAMES
    for j in range(3):
        s1 = np.concatenate([[0], np.cumsum(Z[:, j] * g)])
        s2 = np.concatenate([[0], np.cumsum(Z[:, j] ** 2 * g)])
        a = s1[np.arange(1, n + 1)] - s1[start]
        b = s2[np.arange(1, n + 1)] - s2[start]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = a / cnt
            var = np.maximum(b / cnt - mean ** 2, 0)
        out[ok, 2 * j] = mean[ok]
        out[ok, 2 * j + 1] = np.sqrt(var[ok])
    return out


def build(sessions, use_qc):
    """Frame table restricted to usable label windows. Returns dict of arrays + a report list."""
    F = {k: [] for k in ("sess", "win", "y", "raw", "roll")}
    report, wid = [], 0
    for name, rows in sessions.items():
        t = np.array([r[0] for r in rows])
        X = np.array([r[1] for r in rows])
        labels = [r[2] for r in rows]
        bad = flag_dropouts(t, X[:, 0]) if use_qc else np.zeros(len(t), bool)
        roll = rolling_features(t, X, ~bad)
        # contiguous label runs
        runs, i = [], 0
        while i < len(rows):
            if labels[i] == "":
                i += 1
                continue
            j = i
            while j + 1 < len(rows) and labels[j + 1] == labels[i] and t[j + 1] - t[j] < 2.0:
                j += 1
            runs.append((i, j, labels[i]))
            i = j + 1
        info = {"session": name, "frames": len(rows), "bad": int(bad.sum()), "kept": [], "dropped": []}
        for i, j, lab in runs:
            idx = np.arange(i, j + 1)
            good = ~bad[idx]
            frac_bad = 1 - good.mean()
            usable = good.sum() >= MIN_GOOD_FRAMES and frac_bad <= MAX_BAD_FRACTION
            (info["kept"] if usable else info["dropped"]).append((lab, len(idx), round(float(frac_bad), 2)))
            if not usable:
                continue
            idx = idx[good & np.isfinite(roll[idx]).all(axis=1)]
            if len(idx) < MIN_GOOD_FRAMES:
                info["kept"].pop(); info["dropped"].append((lab, len(idx), "warmup"))
                continue
            F["sess"] += [name] * len(idx)
            F["win"] += [wid] * len(idx)
            F["y"] += [0 if lab == "focused" else 1] * len(idx)
            F["raw"].append(X[idx]); F["roll"].append(roll[idx])
            wid += 1
        report.append(info)
    if not F["y"]:
        return None, report
    return {"sess": np.array(F["sess"]), "win": np.array(F["win"]), "y": np.array(F["y"]),
            "raw": np.vstack(F["raw"]), "roll": np.vstack(F["roll"])}, report


def bal_acc(y, p):
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return 0.5 * ((p[pos] == 1).mean() + (p[neg] == 0).mean())


def fit_logistic(X, y, l2=1.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    pos = max(y.mean(), 1e-9)
    sw = np.where(y == 1, 0.5 / pos, 0.5 / max(1 - pos, 1e-9))
    w = np.zeros(Z.shape[1]); R = np.eye(len(w)) * l2; R[0, 0] = 0
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (sw * (p - y)) + R @ w
        H = (Z * (sw * p * (1 - p))[:, None]).T @ Z + R + 1e-6 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-7:
            break
    return mu, sd, w


def predict_logistic(m, X):
    mu, sd, w = m
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return (1 / (1 + np.exp(-np.clip(Z @ w, -30, 30))) > 0.5).astype(int)


def code_of(raw, med):
    return ((raw / (2 * med)) >= 0.5).astype(int) @ np.array([4, 2, 1])


def fit_lookup(raw, y):
    med = np.median(raw, axis=0)
    codes = code_of(raw, med)
    table = {int(c): int(y[codes == c].mean() > 0.5) for c in np.unique(codes)}
    return med, table, int(y.mean() > 0.5)


def predict_lookup(m, raw):
    med, table, default = m
    return np.array([table.get(int(c), default) for c in code_of(raw, med)])


def deployed_predictions(raw):
    try:
        sys.path.insert(0, str(HERE))
        import json
        from aura_sdk.inference.bnn import AuraBNN
        sc = json.loads((HERE / "models" / "entropy_scale.json").read_text())
        bnn = AuraBNN()
        out = []
        for sp, te, inn in raw:
            r = bnn.infer(sp / (2 * sc["h_sp_median"]), te / (2 * sc["h_te_median"]), inn / (2 * sc["h_in_median"]))
            out.append(int(r.q_hci < ALERT_THRESHOLD))
        return np.array(out)
    except Exception as e:
        print(f"  (M1 deployed model skipped: {type(e).__name__}: {e})")
        return None


def loso(D, rng=None, models=("M2", "M3")):
    """Leave-one-session-out. If rng is given, training window labels are shuffled (null)."""
    sessions = sorted(set(D["sess"]))
    pooled = {m: [] for m in models}
    idx_test_all, folds = [], 0
    for s in sessions:
        te = D["sess"] == s
        tr = ~te
        ytr = D["y"][tr].copy()
        if len(set(ytr)) < 2:
            continue
        if rng is not None:
            wins = np.unique(D["win"][tr])
            wy = {w: D["y"][tr][D["win"][tr] == w][0] for w in wins}
            perm = rng.permutation([wy[w] for w in wins])
            m = dict(zip(wins, perm))
            ytr = np.array([m[w] for w in D["win"][tr]])
            if len(set(ytr)) < 2:
                continue
        folds += 1
        idx_test_all.append(np.where(te)[0])
        if "M2" in models:
            pooled["M2"].append(predict_lookup(fit_lookup(D["raw"][tr], ytr), D["raw"][te]))
        if "M3" in models:
            pooled["M3"].append(predict_logistic(fit_logistic(D["roll"][tr], ytr), D["roll"][te]))
    if folds == 0:
        return None, None, 0
    idx = np.concatenate(idx_test_all)
    return idx, {m: np.concatenate(v) for m, v in pooled.items()}, folds


def window_level(D, idx, pred):
    wins = np.unique(D["win"][idx])
    wy, wp = [], []
    for w in wins:
        sel = D["win"][idx] == w
        wy.append(D["y"][idx][sel][0]); wp.append(int(pred[sel].mean() > 0.5))
    return np.array(wy), np.array(wp)


def boot_ci(wy, wp, rng, n=2000):
    vals = []
    for _ in range(n):
        k = rng.integers(0, len(wy), len(wy))
        b = bal_acc(wy[k], wp[k])
        if b == b:
            vals.append(b)
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) > 50 else (float("nan"),) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "live_sessions"))
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--no-qc", action="store_true", help="skip dropout cleaning (shows how artefacts can fake a signal)")
    a = ap.parse_args()

    sessions = load_sessions(a.dir)
    print(f"Loaded {len(sessions)} session file(s) from {a.dir}")
    D, report = build(sessions, use_qc=not a.no_qc)
    print("\nPer-session summary (QC %s):" % ("OFF" if a.no_qc else "on"))
    for r in report:
        k = [(l, n) for l, n, _ in r["kept"]]
        print(f"  {r['session']}: {r['frames']} frames, {r['bad']} flagged as tracking dropout, "
              f"{len(r['kept'])} usable window(s), {len(r['dropped'])} dropped")
        for l, n, fb in r["dropped"]:
            print(f"      dropped: {l} ({n} frames, bad={fb})")
    if D is None:
        print("\nRESULT: NOT VALIDATABLE -- no usable labeled windows."); return 2

    sess_pos = sorted({s for s, y in zip(D["sess"], D["y"]) if y == 1})
    sess_neg = sorted({s for s, y in zip(D["sess"], D["y"]) if y == 0})
    nwin = {c: len(set(D["win"][D["y"] == c])) for c in (0, 1)}
    print(f"\nUsable windows: focused={nwin[0]} (in {len(sess_neg)} session(s)), "
          f"not-focused={nwin[1]} (in {len(sess_pos)} session(s))")

    idx, preds, folds = loso(D)
    problems = []
    if len(sess_pos) < 2:
        problems.append(f"only {len(sess_pos)} session(s) contain usable NOT-focused windows (need >= 2, ideally 3+)")
    if len(sess_neg) < 2:
        problems.append(f"only {len(sess_neg)} session(s) contain usable focused windows (need >= 2)")
    if nwin[1] < 6 or nwin[0] < 6:
        problems.append(f"need roughly >= 6 independent windows per class (have focused={nwin[0]}, not-focused={nwin[1]})")
    if idx is not None:
        t0 = D["y"][idx]
        if len(set(t0)) < 2:
            problems.append("held-out folds that could be trained never contain both classes")
    else:
        problems.append("no fold had a training set containing both classes")
    if problems:
        print("\nRESULT: NOT VALIDATABLE. A balanced-accuracy number from this data would be meaningless because:")
        for p in problems:
            print("  - " + p)
        print("Fix: record 3+ sessions on different days/times, each with focused AND unfocused blocks of\n"
              "3-5 minutes (alternate them), labeled with label_control.html. Then re-run this script.")
        return 2

    rng = np.random.default_rng(0)
    y_test = D["y"][idx]
    print(f"\nLeave-one-session-out over {folds} fold(s); held-out frames={len(idx)}")
    print(f"{'model':34s} {'frame BA':>9s} {'window BA':>10s} {'95% CI (windows)':>18s} {'null p':>8s}")
    results = {}
    m1 = deployed_predictions(D["raw"][idx])
    allm = []
    if m1 is not None:
        allm.append(("M1 deployed BNN (as run_live)", m1, None))
    allm += [("M2 pattern lookup (retrain_bnn)", preds["M2"], "M2"), ("M3 rolling+baseline logistic", preds["M3"], "M3")]
    for name, p, key in allm:
        fb = bal_acc(y_test, p)
        wy, wp = window_level(D, idx, p)
        wb = bal_acc(wy, wp)
        lo, hi = boot_ci(wy, wp, rng)
        pval = float("nan")
        if key:
            null = []
            for i in range(a.perms):
                nidx, npred, nf = loso(D, rng=np.random.default_rng(1000 + i), models=(key,))
                if nidx is not None and len(set(D["y"][nidx])) == 2:
                    null.append(bal_acc(D["y"][nidx], npred[key]))
            if null:
                pval = (1 + sum(v >= fb for v in null)) / (1 + len(null))
        results[name] = (fb, wb, lo, pval)
        print(f"{name:34s} {fb:9.3f} {wb:10.3f} {f'[{lo:.2f}, {hi:.2f}]':>18s} {pval:8.3f}")

    best = max((v for k, v in results.items() if not k.startswith("M1")), key=lambda v: v[0])
    fb, wb, lo, pval = best
    print("\nRESULT:", end=" ")
    if fb >= 0.65 and lo > 0.5 and pval < 0.05:
        print("SIGNAL SUPPORTED on held-out sessions (best frame BA %.2f, p=%.3f). Still single-user unless the sessions are from several people." % (fb, pval))
    elif pval >= 0.05 or fb < 0.55:
        print("NO SIGNAL DETECTED beyond chance (best frame BA %.2f, p=%.3f): suspect features, labels, or protocol before changing the model." % (fb, pval))
    else:
        print("INCONCLUSIVE / WEAK (best frame BA %.2f, CI low %.2f, p=%.3f): collect more sessions." % (fb, lo, pval))
    return 0


if __name__ == "__main__":
    sys.exit(main())
