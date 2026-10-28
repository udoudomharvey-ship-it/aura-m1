"""
aura_sdk/recommend/escalate.py
──────────────────────────────────
Language: Python
Why: Steps 1-3 all end at "notify the person." That's the right amount
     of intervention for a single, resolvable episode — but if
     recommendations keep firing and Q_HCI never actually recovers,
     continuing to just nudge the same person is exactly the nagging
     failure mode this whole layer exists to avoid. This is the
     fallback for that case: escalate to a human — potentially several,
     in order — with consent configured explicitly rather than assumed.

Escalation triggers on UNRESOLVED outcomes, not on raw episode count —
it reuses the exact Q_HCI-recovery signal step 3 already computes (via
PersonalizedRecommender's on_outcome hook). A person who gets five
recommendations that each individually helped never escalates; someone
whose Q_HCI doesn't recover across several attempts in a row does —
that distinction is the actual point of escalating.

Contacts are TIERED: each has its own unresolved-count threshold, so a
typical setup is "partner at 3 unresolved, clinician at 8" — a longer
struggle escalates further up the chain rather than re-notifying the
same person every time. Once the streak resolves (Q_HCI recovers),
the whole chain resets — the next episode starts at tier 1 again.

Consent is structural, not a suggestion: EscalationConfig defaults to
disabled and to zero contacts, and enabling it with no contacts raises
rather than silently doing nothing. Contacts can be saved/loaded as
JSON so a companion app's setup flow owns collecting and storing them —
this module never invents or guesses a contact.

This is step 4 of the 4-part recommend/act layer:
    1. Rule-based recommendation table
    2. Plain-language explanation layer
    3. Contextual bandit for personalisation
    4. Escalation / human-in-the-loop path        ← this file
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Optional, List, Set

from .rules import RecommendationEngine, RecommendationEvent, ProblemClass
from .bandit import ActionBandit
from .personalize import PersonalizedRecommender
from ..hemm.extractor import EntropyFeatures


# ── Contacts ─────────────────────────────────────────────────────────────

@dataclass
class EscalationContact:
    """One entry in the escalation chain. `threshold` is the consecutive
    unresolved-outcome count at which THIS contact gets notified —
    contacts should be given ascending thresholds (nearer/lighter-touch
    contact first, e.g. a partner at 3, a clinician at 8)."""
    name:      str
    method:    str    # e.g. "sms:+234-555-0100", "email:a@b.com"
    threshold: int

    def to_dict(self) -> dict:
        return {"name": self.name, "method": self.method, "threshold": self.threshold}

    @classmethod
    def from_dict(cls, data: dict) -> "EscalationContact":
        return cls(name=data["name"], method=data["method"], threshold=data["threshold"])


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class EscalationConfig:
    """Escalation is OFF by default and requires at least one contact.
    Enabling it with an empty contact list is a programming error, not
    a silent no-op — see validate(). This is meant to be filled in by a
    companion app's setup flow (see to_dict/from_dict/save/load) rather
    than hand-authored per session."""
    enabled:          bool                       = False
    contacts:         List[EscalationContact]    = field(default_factory=list)
    cooldown_frames:  int                        = 9000   # ~5 min at 30fps between any two escalations

    def validate(self) -> None:
        if self.enabled and not self.contacts:
            raise ValueError(
                "EscalationConfig.enabled=True requires at least one "
                "contact — escalation must never fire without an "
                "explicit, configured place to send it."
            )
        thresholds = [c.threshold for c in self.contacts]
        if thresholds != sorted(thresholds):
            raise ValueError(
                "EscalationConfig.contacts must be ordered by ascending "
                "threshold (nearer contact first, e.g. partner then "
                "clinician) — got thresholds " + str(thresholds)
            )

    # ── Persistence ──────────────────────────────────────────────────────
    # A companion app's setup flow owns collecting contacts from the
    # user; this just needs to survive being written to and read back
    # from disk between sessions.

    def to_dict(self) -> dict:
        return {
            "enabled":         self.enabled,
            "contacts":        [c.to_dict() for c in self.contacts],
            "cooldown_frames": self.cooldown_frames,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EscalationConfig":
        return cls(
            enabled         = data.get("enabled", False),
            contacts        = [EscalationContact.from_dict(c) for c in data.get("contacts", [])],
            cooldown_frames = data.get("cooldown_frames", 9000),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EscalationConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class EscalationEvent:
    """One escalation, ready to hand to whatever actually sends the
    message (SMS gateway, email, push to a caregiver app, ...)."""
    frame_idx:         int
    unresolved_count:  int
    tier:              int     # index into config.contacts — 0 = first/nearest contact
    contact_name:      str
    contact_method:    str
    message:           str

    def to_dict(self) -> dict:
        return {
            "frame_idx":        self.frame_idx,
            "unresolved_count": self.unresolved_count,
            "tier":             self.tier,
            "contact_name":     self.contact_name,
            "contact_method":   self.contact_method,
            "message":          self.message,
        }


# ── Manager ──────────────────────────────────────────────────────────────

class EscalationManager:
    """
    Tracks a running streak of consecutive UNRESOLVED recommendation
    outcomes and, as that streak clears each contact's threshold in
    turn, raises one EscalationEvent per newly-reached tier — never
    more often than `cooldown_frames` allows between any two
    escalations, and never at all unless `config.enabled`.

    Once the streak resolves (a recommendation's Q_HCI recovers), the
    whole chain resets: the next episode starts at tier 0 again, even
    if a later tier was already notified this time.

    Usage
    ─────
        manager = EscalationManager(EscalationConfig(
            enabled=True,
            contacts=[
                EscalationContact("Amaka", "sms:+234-555-0100", threshold=3),
                EscalationContact("Dr. Bello", "email:bello@clinic.example", threshold=8),
            ],
        ))
        manager.record_outcome(resolved=False)   # call per resolved outcome
        event = manager.maybe_escalate(frame_idx)
    """

    def __init__(self, config: Optional[EscalationConfig] = None):
        self.config = config if config is not None else EscalationConfig()
        self.config.validate()
        self._unresolved_streak = 0
        self._frames_since_last_escalation = self.config.cooldown_frames
        self._notified_tiers: Set[int] = set()

    def record_outcome(self, resolved: bool) -> None:
        """Call once per resolved recommendation outcome — `resolved`
        should be the same reward boolean the bandit received (True =
        Q_HCI recovered, False = it didn't)."""
        if resolved:
            self._unresolved_streak = 0
            self._notified_tiers.clear()   # fully resolved — next episode starts at tier 0
        else:
            self._unresolved_streak += 1

    def maybe_escalate(self, frame_idx: int) -> Optional[EscalationEvent]:
        self._frames_since_last_escalation += 1

        if not self.config.enabled:
            return None
        if self._frames_since_last_escalation < self.config.cooldown_frames:
            return None

        # Notify the highest tier whose threshold has been reached and
        # hasn't already been notified this episode — i.e. escalate
        # forward through the chain, never re-notify the same tier.
        target = None
        for i, contact in enumerate(self.config.contacts):
            if self._unresolved_streak >= contact.threshold and i not in self._notified_tiers:
                target = i
        if target is None:
            return None

        contact = self.config.contacts[target]
        self._notified_tiers.add(target)
        self._frames_since_last_escalation = 0

        message = (
            f"Automated check-in: recommended breaks haven't seemed to help "
            f"{self._unresolved_streak} times in a row. It might be worth checking in."
        )

        return EscalationEvent(
            frame_idx        = frame_idx,
            unresolved_count = self._unresolved_streak,
            tier             = target,
            contact_name     = contact.name,
            contact_method   = contact.method,
            message          = message,
        )

    def reset(self) -> None:
        self._unresolved_streak = 0
        self._frames_since_last_escalation = self.config.cooldown_frames
        self._notified_tiers.clear()


# ── Full 4-step recommender ─────────────────────────────────────────────────

class EscalatingRecommender:
    """
    All four steps composed into one object: rule-based detection
    (step 1), plain-language rendering available via the module-level
    `explain()` function (step 2), personalized action ordering
    (step 3), and tiered human escalation when recommendations keep
    failing to help (step 4).

    Usage
    ─────
        recommender = EscalatingRecommender(
            escalation_config=EscalationConfig(
                enabled=True,
                contacts=[
                    EscalationContact("Amaka", "sms:+234-555-0100", threshold=3),
                    EscalationContact("Dr. Bello", "email:bello@clinic.example", threshold=8),
                ],
            ),
        )
        for frame in stream:
            event = recommender.update(alert=..., state=..., q_hci=...,
                                        features=..., frame_idx=...)
            if event:
                notify_user(explain(event))
            if recommender.last_escalation:
                notify_contact(recommender.last_escalation)
    """

    def __init__(
        self,
        base_engine:        Optional[RecommendationEngine] = None,
        bandit:              Optional[ActionBandit]          = None,
        escalation_config:   Optional[EscalationConfig]      = None,
        eval_window_frames:  int   = 150,
        recovery_margin:     float = 0.05,
    ):
        self.escalation = EscalationManager(escalation_config)
        self._personalized = PersonalizedRecommender(
            base_engine        = base_engine,
            bandit             = bandit,
            eval_window_frames = eval_window_frames,
            recovery_margin    = recovery_margin,
            on_outcome         = self._on_outcome,
        )
        self.last_escalation: Optional[EscalationEvent] = None

    @property
    def bandit(self) -> ActionBandit:
        return self._personalized.bandit

    def _on_outcome(self, problem_class: ProblemClass, action: str,
                     reward: bool, frame_idx: int) -> None:
        self.escalation.record_outcome(resolved=reward)

    def update(
        self,
        alert:     bool,
        state:     Optional[int],
        q_hci:     Optional[float],
        features:  Optional[EntropyFeatures],
        frame_idx: int,
    ) -> Optional[RecommendationEvent]:
        event = self._personalized.update(alert=alert, state=state, q_hci=q_hci,
                                           features=features, frame_idx=frame_idx)
        self.last_escalation = self.escalation.maybe_escalate(frame_idx)
        return event

    def reset(self) -> None:
        self._personalized.reset()
        self.escalation.reset()
        self.last_escalation = None
