"""
dashboard_server.py
────────────────────────────────────────────
Live, real-time Aura-M1 dashboard for a project defense demo. Runs the
REAL pipeline end to end -- HEMMExtractor -> AuraBNN -> ThresholdDetector
-> EscalatingRecommender (rule table + Thompson bandit + tiered
escalation) -- and writes live state to dashboard_state.json every frame
for the browser page to poll.

Two modes, switchable live from the dashboard page's buttons (POST
/trigger?scenario=...):
  - "auto"   -- cycles through all 3 reachable ProblemClasses on a timer
  - "manual" -- holds on ONE scenario, chosen by you, until you pick a
                different one or click "Resume auto-cycle". Switches
                take effect within a frame or two, not at the end of
                whatever phase happened to be playing.

Deliberately uses the REAL AuraBNN as-is, including its known alerting
limitation (see aura-m1_bandit_bnn_findings.md from this session):
cognitive_overload reliably alerts, spatial_restlessness essentially
never does, sustained_low_quality sits at a fragile threshold edge. The
dashboard surfaces this honestly (see KNOWN_LIMITATION below) rather
than hiding it behind a fake signal.

No third-party dependencies -- Python standard library only (plus your
own aura_sdk), so nothing new to install on demo day.

Usage:
    python3 dashboard_server.py
    # then open http://localhost:8765/dashboard.html in a browser
"""
from __future__ import annotations
import json
import os
import threading
import time
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
from aura_sdk_tests.scenario_synthetic_source import ScenarioSyntheticSource
from aura_sdk.voice import VoiceCoach

PORT = 8765
STATE_PATH = Path(__file__).parent / "dashboard_state.json"

AUTO_ROTATION = [
    (ProblemClass.COGNITIVE_OVERLOAD, 200),
    (ProblemClass.SPATIAL_RESTLESSNESS, 150),
    (ProblemClass.SUSTAINED_LOW_QUALITY, 150),
]

VALID_SCENARIOS = {pc.value: pc for pc, _ in AUTO_ROTATION}

KNOWN_LIMITATION = {
    ProblemClass.COGNITIVE_OVERLOAD.value: {
        "status": "working",
        "note": "Reliably alerts -- verified this session.",
    },
    ProblemClass.SPATIAL_RESTLESSNESS.value: {
        "status": "known_limitation",
        "note": "BNN rarely alerts here even at high H_sp -- known training-data bug, fix designed but not yet integrated.",
    },
    ProblemClass.SUSTAINED_LOW_QUALITY.value: {
        "status": "known_limitation",
        "note": "Sits right at the alert threshold edge -- fires inconsistently. Same root cause as above.",
    },
}

_control_lock = threading.Lock()
_control = {"mode": "auto", "manual_target": None}


def bandit_snapshot(bandit) -> dict:
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
        contacts=[
            EscalationContact("Demo contact", "email:demo@example.com", threshold=3),
        ],
    )
    recommender = EscalatingRecommender(
        escalation_config=escalation_config,
        eval_window_frames=60,
        recovery_margin=0.05,
    )
    extractor = HEMMExtractor()
    voice = VoiceCoach()  # off unless AURA_VOICE=1
    bnn = AuraBNN()
    detector = ThresholdDetector(threshold=0.45)

    events: list[dict] = []

    def log_event(text: str, frame_idx: int):
        events.insert(0, {"frame_idx": frame_idx, "text": text})
        del events[30:]

    frame_idx = 0
    auto_rotation_index = 0

    while True:
        with _control_lock:
            mode = _control["mode"]
            manual_target = _control["manual_target"]

        if mode == "manual" and manual_target is not None:
            target_pc = manual_target
            n_frames = 100_000
        else:
            target_pc, n_frames = AUTO_ROTATION[auto_rotation_index % len(AUTO_ROTATION)]
            auto_rotation_index += 1

        source = ScenarioSyntheticSource(phases=[(target_pc, n_frames)], seed=frame_idx % 1000)
        source.open()
        prev_joints = None

        for frame in source.frames():
            with _control_lock:
                current_mode = _control["mode"]
                current_target = _control["manual_target"]
            if current_mode != mode or (current_mode == "manual" and current_target != manual_target):
                break

            if prev_joints is not None:
                feats = extractor.compute(prev_joints, frame.joints)
                result = bnn.infer(feats.h_sp, feats.h_te, feats.h_in)
                alert = detector.update(result.q_hci)

                rec = recommender.update(
                    alert=alert, state=result.state, q_hci=result.q_hci,
                    features=feats, frame_idx=frame_idx,
                )
                if rec:
                    log_event(
                        f"Recommendation fired: {rec.problem_class.value} -> {rec.suggested_actions[0]}",
                        frame_idx,
                    )
                    voice.announce(rec)
                if recommender.last_escalation:
                    e = recommender.last_escalation
                    log_event(f"ESCALATION tier {e.tier} -> {e.contact_name}", frame_idx)

                state_payload = {
                    "frame_idx": frame_idx,
                    "mode": mode,
                    "target_scenario": target_pc.value,
                    "scenario_status": KNOWN_LIMITATION[target_pc.value],
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
            time.sleep(1.0 / 30.0)
        source.close()


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/trigger":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        scenario = params.get("scenario", [None])[0]

        if scenario == "auto":
            with _control_lock:
                _control["mode"] = "auto"
                _control["manual_target"] = None
            result = {"ok": True, "mode": "auto"}
        elif scenario in VALID_SCENARIOS:
            with _control_lock:
                _control["mode"] = "manual"
                _control["manual_target"] = VALID_SCENARIOS[scenario]
            result = {"ok": True, "mode": "manual", "target": scenario}
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": "unknown scenario"}).encode())
            return

        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    threading.Thread(target=run_pipeline, daemon=True).start()

    os.chdir(Path(__file__).parent)
    server = ThreadingHTTPServer(("localhost", PORT), DashboardHandler)
    print(f"Dashboard running -- open http://localhost:{PORT}/dashboard.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
