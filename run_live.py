"""
run_live.py
────────────────────────────────────────────
MVP Step 1 (data source): run the REAL Aura-M1 pipeline on your own
webcam, live, using aura_sdk.sensors.sources.MediaPipeSource instead
of the synthetic demo source. No depth camera, no custom hardware.

MVP Step 2 (real training data): every frame's entropy features are
logged to a timestamped CSV, and a second browser tab
(label_control.html) lets you self-label your own state in real time
without interrupting your work. A label applies to the LABEL_WINDOW_S
seconds of frames immediately before you click it, since people
notice and report a state after it's been true for a while, not the
instant it starts.

Everything downstream of the joint-position stream is unchanged and
already tested: HEMMExtractor -> AuraBNN -> ThresholdDetector ->
EscalatingRecommender. This script only swaps what feeds them and
adds logging on top.

Requires:
    pip install mediapipe opencv-python

Usage:
    python3 run_live.py
    # then open, in two separate browser tabs:
    #   http://localhost:8765/dashboard.html       (watch it work)
    #   http://localhost:8765/label_control.html   (label your state)
"""
from __future__ import annotations
import csv
import json
import threading
import time
import traceback
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from aura_sdk.hemm.extractor import HEMMExtractor
from aura_sdk.inference.bnn import AuraBNN
from aura_sdk.ncl.circuits import ThresholdDetector
from aura_sdk.recommend import (
    ACTION_LIBRARY, EscalationConfig, EscalationContact,
    EscalatingRecommender, ProblemClass,
)
from aura_sdk.sensors.sources import MediaPipeSource
from aura_sdk.voice import VoiceCoach

PORT = 8765
STATE_PATH = Path(__file__).parent / "dashboard_state.json"

# How far back a self-applied label reaches, in seconds. People notice
# and report a state ("I've been restless") after it's been true for a
# stretch, not at the exact instant it started, so a label is applied
# to the whole preceding window rather than just the single frame at
# click-time.
LABEL_WINDOW_S = 20

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path(__file__).parent / "live_sessions"
LOG_DIR.mkdir(exist_ok=True)
FRAMES_CSV = LOG_DIR / f"session_{SESSION_ID}_frames.csv"
LABELS_CSV = LOG_DIR / f"session_{SESSION_ID}_labels.csv"

_labels_lock = threading.Lock()
_pending_labels: list[dict] = []  # {"t": wall_clock_seconds, "value": str}

# Set True only once the pipeline thread is actually looping over real
# frames, and back to False the instant it stops for ANY reason (clean
# exit, Python exception, or the watchdog noticing it died silently).
# The /label endpoint checks this so it can never again report "Logged"
# for a label that has nowhere to actually land.
_pipeline_alive = threading.Event()
_last_frame_processed_at = [0.0]


def bandit_snapshot(bandit) -> dict:
    """Identical to dashboard_server.py's version, so the existing
    dashboard.html needs no changes to read it."""
    snap = {}
    for pc in ProblemClass:
        rows = []
        for action in ACTION_LIBRARY[pc]:
            mean = bandit._arms[(pc, action)].mean()
            rows.append([action, round(mean, 3)])
        rows.sort(key=lambda r: r[1], reverse=True)
        snap[pc.value] = rows
    return snap


def run_pipeline():
    escalation_config = EscalationConfig(
        enabled=True,
        contacts=[EscalationContact("Demo contact", "email:demo@example.com", threshold=3)],
    )
    recommender = EscalatingRecommender(
        escalation_config=escalation_config,
        eval_window_frames=60,
        recovery_margin=0.05,
    )
    extractor = HEMMExtractor()
    bnn = AuraBNN()
    detector = ThresholdDetector(threshold=0.45)
    entropy_scale = load_entropy_scale()
    voice = VoiceCoach()  # off unless AURA_VOICE=1
    print(f"[run_live] Voice coaching {'ON' if voice.enabled else 'off (set AURA_VOICE=1)'}")
    print("[run_live] Entropy rescaling ACTIVE" if entropy_scale
          else "[run_live] No entropy_scale.json -- raw entropy goes to the classifier")

    events: list[dict] = []

    def log_event(text: str, frame_idx: int):
        events.insert(0, {"frame_idx": frame_idx, "text": text})
        del events[30:]

    print("[run_live] Opening webcam + MediaPipe pose model "
          "(first run downloads the model, ~a few MB)...")
    source = MediaPipeSource(camera_idx=0, model_complexity="full", min_confidence=0.3)
    source.open()
    print("[run_live] Webcam open. Sit where your upper body is visible. "
          "Ctrl+C to stop.")
    print(f"[run_live] Logging frames to {FRAMES_CSV.name}")
    print(f"[run_live] Logging labels to {LABELS_CSV.name}")

    header = ["frame_idx", "wall_time_s", "h_sp", "h_te", "h_in",
              "q_hci", "bnn_state", "alert", "label"]

    # All rows live in memory as the single source of truth for this
    # session. Labels are applied here, in memory, never by re-reading
    # the CSV back off disk. Disk writes are periodic, full rewrites
    # through ONE handle that is opened, written, and closed each time
    # -- never two independent handles open on the same path at once.
    # (An earlier version kept one handle open continuously for
    # appending while a second handle did a truncating rewrite for
    # labels; the appending handle's stale write offset then corrupted
    # the file. This design can't do that, because there's never more
    # than one handle on the file at any moment.)
    all_rows: list[list] = []
    # Written this often specifically so an abrupt termination (closed
    # terminal, laptop sleep, or a native crash inside MediaPipe/OpenCV's
    # C++ layer that bypasses Python's `finally` entirely) loses at most
    # a couple of seconds of data, not the whole session. A full CSV
    # rewrite at this scale is a few milliseconds, so this is cheap.
    WRITE_EVERY_N_FRAMES = 10

    def persist_to_disk():
        with open(FRAMES_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(all_rows)

    def apply_labels_in_memory():
        with _labels_lock:
            labels = list(_pending_labels)
        if not labels:
            return
        t_idx = header.index("wall_time_s")
        lab_idx = header.index("label")
        for r in all_rows:
            t = r[t_idx]
            for lab in labels:
                if lab["t"] - LABEL_WINDOW_S <= t <= lab["t"]:
                    r[lab_idx] = lab["value"]

    session_start = _SESSION_START_WALLCLOCK[0]
    prev_joints = None
    frame_idx = 0
    last_rec_label = "Monitoring \u2014 no issue detected yet"
    frames_since_write = 0

    _pipeline_alive.set()
    try:
        for frame in source.frames():
            _last_frame_processed_at[0] = time.time()
            if prev_joints is not None:
                feats = extractor.compute(prev_joints, frame.joints)
                # Classifier gets rescaled values; CSV/dashboard keep RAW entropy
                # so retrain_bnn.py never double-scales.
                sp_in, te_in, in_in = rescale_features(
                    feats.h_sp, feats.h_te, feats.h_in, entropy_scale)
                result = bnn.infer(sp_in, te_in, in_in)
                alert = detector.update(result.q_hci)

                rec = recommender.update(
                    alert=alert, state=result.state, q_hci=result.q_hci,
                    features=feats, frame_idx=frame_idx,
                )
                if rec:
                    last_rec_label = rec.problem_class.value.replace("_", " ")
                    voice.announce(rec)
                    log_event(
                        f"Recommendation fired: {rec.problem_class.value} -> {rec.suggested_actions[0]}",
                        frame_idx,
                    )
                if recommender.last_escalation:
                    e = recommender.last_escalation
                    log_event(f"ESCALATION tier {e.tier} -> {e.contact_name}", frame_idx)

                wall_t = round(time.time() - session_start, 2)
                row = [frame_idx, wall_t, round(feats.h_sp, 4),
                       round(feats.h_te, 4), round(feats.h_in, 4),
                       round(result.q_hci, 4), result.state, alert, ""]
                all_rows.append(row)

                # Re-stamp any rows now covered by a freshly-clicked label,
                # entirely in memory -- no disk read, no second file handle.
                with _labels_lock:
                    have_new = bool(_pending_labels)
                if have_new:
                    apply_labels_in_memory()

                frames_since_write += 1
                if frames_since_write >= WRITE_EVERY_N_FRAMES or have_new:
                    persist_to_disk()
                    frames_since_write = 0

                state_payload = {
                    "frame_idx": frame_idx,
                    "mode": "live",
                    "target_scenario": last_rec_label,
                    "scenario_status": {
                        "status": "working",
                        "note": "Live webcam session \u2014 this is your real, current reading, not a scripted demo.",
                    },
                    "entropy": {
                        "h_sp": round(feats.h_sp, 3),
                        "h_te": round(feats.h_te, 3),
                        "h_in": round(feats.h_in, 3),
                    },
                    "bnn_state": result.state,
                    "q_hci": round(result.q_hci, 3),
                    "alert": alert,
                    "current_recommendation": {
                        "problem_class": rec.problem_class.value,
                        "action": rec.suggested_actions[0],
                        "why": rec.why,
                    } if rec else None,
                    "bandit": bandit_snapshot(recommender.bandit),
                    "escalation": {
                        "enabled": escalation_config.enabled,
                        "unresolved_streak": recommender.escalation._unresolved_streak,
                    },
                    "events": events,
                }
                STATE_PATH.write_text(json.dumps(state_payload))

            prev_joints = frame.joints
            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[run_live] Stopping.")
    except Exception:
        # Anything else is a real bug, not a normal shutdown. Print the
        # full traceback loudly (the old behaviour let this vanish into
        # "Exception in thread Thread-1" with no further explanation)
        # and, critically, make sure whatever data was already
        # collected still gets saved rather than lost with the crash.
        print("\n[run_live] CRASHED with an unexpected error:")
        traceback.print_exc()
        print(f"[run_live] Attempting to save the {len(all_rows)} frames "
              f"collected before the crash...")
    finally:
        # One last full, consistent write so nothing since the last
        # periodic save is lost, then close the camera. Runs even after
        # a crash above (though NOT if the whole process is killed
        # externally or a native/C++ crash bypasses Python entirely --
        # that's what the frequent WRITE_EVERY_N_FRAMES saves above are
        # really for).
        _pipeline_alive.clear()
        try:
            apply_labels_in_memory()
            persist_to_disk()
        except Exception:
            print("[run_live] Also failed while trying to save on exit:")
            traceback.print_exc()
        source.close()
        print(f"[run_live] Session saved: {FRAMES_CSV} ({len(all_rows)} frames)")
        if LABELS_CSV.exists():
            print(f"[run_live] Labels saved: {LABELS_CSV}")


class LiveHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/label":
            params = parse_qs(parsed.query)
            value = params.get("value", [None])[0]
            if not value:
                self.send_response(400)
                self.end_headers()
                return

            # A pipeline that died from a native crash (e.g. deep inside
            # MediaPipe/OpenCV's C++ layer) bypasses Python's exception
            # handling entirely -- _pipeline_alive would never get
            # cleared in that case. Staleness of the last processed
            # frame is the real, can't-lie-to-itself signal: if nothing
            # has been processed in 5+ seconds, treat it as dead
            # regardless of what the flag says.
            seconds_since_last_frame = time.time() - _last_frame_processed_at[0]
            pipeline_actually_alive = _pipeline_alive.is_set() and seconds_since_last_frame < 5.0
            if not pipeline_actually_alive:
                print(f"[run_live] WARNING: label '{value}' was clicked but the "
                      f"pipeline appears to be DEAD (no frame processed in "
                      f"{seconds_since_last_frame:.1f}s). This label will NOT "
                      f"be saved to any frame data. Restart run_live.py.")
                body = json.dumps({
                    "ok": False,
                    "error": f"Pipeline is not running (idle {seconds_since_last_frame:.0f}s). "
                             f"Restart run_live.py -- this label was not saved.",
                }).encode()
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            t = time.time() - _SESSION_START_WALLCLOCK[0]
            with _labels_lock:
                _pending_labels.append({"t": t, "value": value})
            is_new = not LABELS_CSV.exists()
            with open(LABELS_CSV, "a", newline="") as f:
                w = csv.writer(f)
                if is_new:
                    w.writerow(["wall_time_s", "value"])
                w.writerow([round(t, 2), value])
            print(f"[run_live] Label received: '{value}' "
                  f"(applies to last {LABEL_WINDOW_S}s)")
            body = json.dumps({"ok": True, "value": value}).encode()
        else:
            # Scenario-trigger buttons on dashboard.html are inert in
            # live mode -- acknowledge quietly rather than 404ing.
            body = json.dumps({"ok": True, "mode": "live", "note": "control bar is inert in live mode"}).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


_SESSION_START_WALLCLOCK = [0.0]

ENTROPY_SCALE_PATH = Path(__file__).parent / "models" / "entropy_scale.json"


def load_entropy_scale():
    """Per-feature medians written by retrain_bnn.py (None if absent)."""
    if not ENTROPY_SCALE_PATH.exists():
        return None
    return json.loads(ENTROPY_SCALE_PATH.read_text())


def rescale_features(h_sp, h_te, h_in, scale):
    if scale is None:
        return h_sp, h_te, h_in
    return (h_sp / (2 * scale["h_sp_median"]),
            h_te / (2 * scale["h_te_median"]),
            h_in / (2 * scale["h_in_median"]))


def main():
    _SESSION_START_WALLCLOCK[0] = time.time()

    if not STATE_PATH.exists():
        STATE_PATH.write_text(json.dumps({
            "frame_idx": 0, "mode": "live", "target_scenario": "starting...",
            "scenario_status": {"status": "working", "note": "Starting webcam..."},
            "entropy": {"h_sp": 0, "h_te": 0, "h_in": 0}, "bnn_state": 3,
            "q_hci": 0.5, "alert": False, "current_recommendation": None,
            "bandit": {}, "escalation": {"enabled": True, "unresolved_streak": 0},
            "events": [],
        }))

    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), LiveHandler)
    print(f"[run_live] Dashboard:      http://localhost:{PORT}/dashboard.html")
    print(f"[run_live] Label control:  http://localhost:{PORT}/label_control.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
