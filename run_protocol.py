"""
run_protocol.py - validation session where the CONDITION is the label.
FOCUS: one real task. OVERLOAD: same task while counting back by 7.
DISTRACT: same task with phone in hand. Labels posted to run_live.py every 20 s.
"""
import argparse, json, random, sys, time, urllib.request, urllib.error
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8765")
ap.add_argument("--block", type=float, default=240)
ap.add_argument("--blocks", type=int, default=6)
ap.add_argument("--every", type=float, default=20)
ap.add_argument("--grace", type=float, default=20)
ap.add_argument("--rest", type=float, default=15)
a = ap.parse_args()

TAG = {"FOCUS": "focused", "OVERLOAD": "cognitive_overload", "DISTRACT": "spatial_restlessness"}
TEXT = {
 "FOCUS":    "FOCUS - do ONE real task (code / read / write). Phone away. Nothing else.",
 "OVERLOAD": "OVERLOAD - same kind of task, but silently count BACKWARDS BY 7 the whole time.",
 "DISTRACT": "DISTRACT - same kind of task, but keep your PHONE in hand: scroll / reply as you normally would.",
}

def post(value):
    req = urllib.request.Request(f"{a.url}/label?value={value}", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200, ""
    except urllib.error.HTTPError as e:
        return False, e.read().decode(errors="ignore")[:200]
    except Exception as e:
        return False, str(e)

try:
    urllib.request.urlopen(f"{a.url}/dashboard.html", timeout=3).read(1)
except Exception as e:
    sys.exit(f"Cannot reach {a.url}. Start `python3 run_live.py` in another terminal first.\n({e})")

n_f = a.blocks // 2
others = ["OVERLOAD", "DISTRACT"] * a.blocks
sched = ["FOCUS"] * n_f + others[: a.blocks - n_f]
random.shuffle(sched)
print("\nSchedule (random order, so time-of-session isn't a confound):")
for i, s in enumerate(sched, 1): print(f"  {i}. {s}")
print("\nKeep run_live.py running. Sit as normal, upper body in frame. Starting in 10 s...")
time.sleep(10)

log = {"started": datetime.now().isoformat(timespec="seconds"), "blocks": []}
posted = {k: 0 for k in TAG}; missed = {k: 0 for k in TAG}
for i, cond in enumerate(sched, 1):
    print("\a\n" + "=" * 70 + f"\nBLOCK {i}/{len(sched)}  ({a.block/60:.1f} min)\n{TEXT[cond]}\n" + "=" * 70, flush=True)
    t0 = time.time(); nxt = a.grace; last_n7 = -99; blk = {"i": i, "cond": cond, "label": TAG[cond], "posts": []}
    while True:
        el = time.time() - t0
        if el >= a.block: break
        if cond == "OVERLOAD" and el - last_n7 >= 30:
            print(f"   count back by 7 from {random.randint(600, 990)}", flush=True); last_n7 = el
        if el >= nxt or el >= a.block - 1:
            ok, err = post(TAG[cond]); blk["posts"].append(round(el, 1))
            if ok: posted[cond] += 1
            else:
                missed[cond] += 1; print(f"   ! label not saved: {err}", flush=True)
            nxt = el + a.every
            if el >= a.block - 1: time.sleep(max(0, a.block - (time.time() - t0))); break
        time.sleep(0.25)
    log["blocks"].append(blk)
    if i < len(sched):
        print(f"\nRest {a.rest:.0f} s (not labelled). Next block starts after.", flush=True); time.sleep(a.rest)
print("\a\nDone. Labels posted:", {TAG[k]: v for k, v in posted.items()}, "missed:", sum(missed.values()))
log["posted"] = posted; log["missed"] = missed
out = Path(__file__).parent / "live_sessions" / f"protocol_{datetime.now():%Y%m%d_%H%M%S}.json"
out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(log, indent=1)); print("Saved", out)
print("Now Ctrl+C run_live.py, then run: python3 eval_session.py")
