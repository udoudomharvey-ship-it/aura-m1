"""
aura_sdk/recommend/explain.py
──────────────────────────────
Language: Python
Why: Step 1 (rules.py) decides WHAT the problem is and WHAT to suggest.
     This module decides HOW to say it to a person — a duration-aware,
     escalation-aware sentence instead of a static template, packaged
     as a small `Notification` a UI layer can render directly.

No network calls, no external model — fully deterministic and testable,
same as step 1. This is step 2 of the 4-part recommend/act layer:
    1. Rule-based recommendation table
    2. Plain-language explanation layer      ← this file
    3. Contextual bandit for personalisation
    4. Escalation / human-in-the-loop path

Two things make this "cognizant" rather than a template lookup:
  - duration:  "for about 15 seconds" is computed from sustained_frames,
               not hardcoded, so it stays accurate as sustain_frames
               config changes.
  - repetition: episode_count lets the tone shift from a gentle first
               nudge to a more direct one on repeat episodes in the
               same session, without ever becoming alarmist.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List

from .rules import RecommendationEvent, ProblemClass


# ── Severity ──────────────────────────────────────────────────────────────
# Derived from workload state, not from Q_HCI directly, so it reads
# naturally: state 1 = "Very High Workload" -> HIGH severity notification.

class Severity(Enum):
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"


def _severity_for_state(state: int) -> Severity:
    if state <= 1:
        return Severity.HIGH
    if state == 2:
        return Severity.MODERATE
    return Severity.LOW


# Human-readable label per problem class, used in the title — short,
# plain, no jargon (never "H_sp dominant", always the everyday name).
_LABELS = {
    ProblemClass.SPATIAL_RESTLESSNESS:  "Restlessness",
    ProblemClass.TEMPORAL_ERRATICNESS:  "Irregular movement",
    ProblemClass.COGNITIVE_OVERLOAD:    "Cognitive overload",
    ProblemClass.SUSTAINED_LOW_QUALITY: "Low interaction quality",
}

# Escalation phrasing appended when this is a repeat episode this
# session. Deliberately supportive, never scolding.
_ESCALATION = {
    1: "",  # first time this session — no escalation needed
    2: " This is the second time this has come up — worth taking seriously.",
    3: " This keeps recurring — a longer break might help more than a short one.",
}
_ESCALATION_DEFAULT = (
    " This has come up several times now — it may be worth stepping away "
    "for longer, or checking in with someone."
)


# ── Notification ──────────────────────────────────────────────────────────

@dataclass
class Notification:
    """A fully-formed, ready-to-render message for a UI layer — the
    companion app, the frontend orb, a push notification, whatever
    consumes this next. No entropy values, no jargon."""
    title:             str
    body:              str
    severity:          Severity
    suggested_actions: List[str]
    tag:               str     # ProblemClass value — for de-duplicating/grouping in a UI

    def to_dict(self) -> dict:
        return {
            "title":             self.title,
            "body":              self.body,
            "severity":          self.severity.value,
            "suggested_actions": self.suggested_actions,
            "tag":               self.tag,
        }


# ── Explain ───────────────────────────────────────────────────────────────

def _duration_phrase(duration_s: float) -> str:
    if duration_s < 1.0:
        return "the last few seconds"
    if duration_s < 60.0:
        return f"about {duration_s:.0f} seconds"
    return f"about {duration_s / 60.0:.1f} minutes"


def explain(event: RecommendationEvent, frame_dt_ms: float = 33.3) -> Notification:
    """
    Turn a RecommendationEvent into a plain-language Notification.

    frame_dt_ms should match the pipeline's PipelineConfig.frame_dt_ms —
    it's only used to turn `sustained_frames` into a human duration.
    """
    severity   = _severity_for_state(event.state)
    label      = _LABELS[event.problem_class]
    duration_s = (event.sustained_frames * frame_dt_ms) / 1000.0

    title = f"{label} detected" if severity != Severity.LOW else "Quick check-in"

    escalation = _ESCALATION.get(event.episode_count, _ESCALATION_DEFAULT)
    body = f"{event.why} This has been going on for {_duration_phrase(duration_s)}.{escalation}"

    return Notification(
        title             = title,
        body              = body,
        severity          = severity,
        suggested_actions = event.suggested_actions,
        tag               = event.problem_class.value,
    )
