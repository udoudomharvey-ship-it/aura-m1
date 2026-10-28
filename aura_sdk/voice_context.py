"""
aura_sdk/voice_context.py
-------------------------
The bridge between the live HEMM pipeline and the voice chat.

run_live.py (camera process) already writes dashboard_state.json every frame, and
VoiceCoach writes voice_state.json each time it speaks a nudge. The chat process
(tools/talk.py) reads both and hands the voice model a small, honest summary.

Only these leave the machine: the workload number 1-5, whether an alert is active,
how old the reading is, and the last few things Aura said. Never entropy values,
the focus score, joints, video or anything on screen.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = Path(os.environ.get("AURA_STATE_PATH", _ROOT / "dashboard_state.json"))
VOICE_STATE_PATH = Path(os.environ.get("AURA_VOICE_STATE_PATH", _ROOT / "voice_state.json"))
MAX_AGE_S = 10.0
RECENT_NUDGE_MAX_AGE_S = 1800.0   # nudges older than 30 minutes are not "recent"

WORKLOAD_WORDS = {1: "very high", 2: "high", 3: "moderate", 4: "low", 5: "very low"}


def write_voice_state(lines: List[str], problem: str, episode: int,
                      path: Optional[Path] = None, now: Callable[[], float] = time.time) -> bool:
    """Atomic write so a reader never sees half a file. Never raises."""
    path = Path(path or VOICE_STATE_PATH)
    try:
        data = {"updated_at": now(), "problem": problem, "episode": episode, "recent": lines[-5:]}
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _read_json(path: Path) -> Optional[dict]:
    """The live state file is rewritten every frame, so one read can catch it mid-write: retry once."""
    for attempt in range(2):
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            if attempt == 0:
                time.sleep(0.05)
    return None


def read_context(state_path: Optional[Path] = None, voice_path: Optional[Path] = None,
                 now: Callable[[], float] = time.time, max_age_s: float = MAX_AGE_S) -> dict:
    """Return extra fields for the voice request. Empty-ish dict if nothing is live."""
    sp, vp = Path(state_path or STATE_PATH), Path(voice_path or VOICE_STATE_PATH)
    ctx: dict = {}
    notes: List[str] = []

    live = None
    try:
        age = now() - sp.stat().st_mtime
        if age <= max_age_s:
            live = _read_json(sp)
        else:
            notes.append("The live monitoring session is not running, so there is no current workload estimate.")
    except OSError:
        notes.append("The live monitoring session is not running, so there is no current workload estimate.")
    if live and live.get("mode") == "live" and isinstance(live.get("bnn_state"), int):
        st = int(live["bnn_state"])
        if 1 <= st <= 5:
            ctx["workload_state_1_to_5"] = st
            notes.append(f"Application estimate from the last few seconds: {WORKLOAD_WORDS[st]} workload "
                         f"({st} of 5). This is an estimate, not a diagnosis.")
            if live.get("alert"):
                notes.append("The application currently has a sustained alert active.")

    vs = _read_json(vp)
    if vs:
        try:
            fresh = (now() - float(vs.get("updated_at", 0))) <= RECENT_NUDGE_MAX_AGE_S
        except (TypeError, ValueError):
            fresh = False
        if not fresh:
            vs = None
    if vs:
        recent = [str(x) for x in vs.get("recent", []) if x][-5:]
        if recent:
            ctx["recent_interventions"] = recent
            try:
                mins = max(0, int((now() - float(vs.get("updated_at", 0))) // 60))
                notes.append(f"The last spoken nudge was about {mins} minute(s) ago "
                             f"(topic: {str(vs.get('problem', '')).replace('_', ' ')}).")
            except Exception:
                pass
    if notes:
        ctx["available_context"] = notes
    return ctx
