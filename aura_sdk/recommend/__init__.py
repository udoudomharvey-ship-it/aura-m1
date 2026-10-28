"""
aura_sdk.recommend
────────────────────
Language: Python
The recommend/act layer — sits above detect/measure/alert.

    from aura_sdk.recommend import RecommendationEngine, RecommendationEvent, ProblemClass
"""
from .rules import (
    RecommendationEngine,
    RecommendationEvent,
    ProblemClass,
    ACTION_LIBRARY,
    WHY_TEMPLATES,
)
from .explain import (
    explain,
    Notification,
    Severity,
)
from .bandit import ActionBandit
from .personalize import PersonalizedRecommender
from .escalate import (
    EscalationConfig, EscalationContact, EscalationEvent,
    EscalationManager, EscalatingRecommender,
)
from .notify import SMTPConfig, SMTPEmailNotifier, DryRunNotifier, send_escalation

__all__ = [
    "RecommendationEngine", "RecommendationEvent", "ProblemClass",
    "ACTION_LIBRARY", "WHY_TEMPLATES",
    "explain", "Notification", "Severity",
    "ActionBandit", "PersonalizedRecommender",
    "EscalationConfig", "EscalationContact", "EscalationEvent",
    "EscalationManager", "EscalatingRecommender",
    "SMTPConfig", "SMTPEmailNotifier", "DryRunNotifier", "send_escalation",
]
