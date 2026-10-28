#!/usr/bin/env bash
# Run from the ROOT of your aura-m1 project.
#   bash setup_notify.sh
#
# Adds aura_sdk/recommend/notify.py (SMTP email sending for escalations),
# updates __init__.py to export it, adds test_recommend_notify.py, and
# wires send_escalation() into run_with_escalation.py.

set -euo pipefail

echo "Creating aura_sdk/recommend/notify.py ..."
cat > aura_sdk/recommend/notify.py << 'AURA_EOF_NOTIFY'
"""
aura_sdk/recommend/notify.py
──────────────────────────────
Language: Python
Why: EscalationEvent (step 4) carries a message and a contact_method,
     but nothing actually sends it — this module is that last mile,
     for the "email:" contact method specifically (Gmail/any SMTP
     provider). SMS is intentionally NOT implemented here; sending it
     would need a paid gateway (Twilio etc.) with its own setup, so
     send_escalation() reports that clearly rather than pretending to
     have sent something it didn't.

Two safety properties are load-bearing, not incidental:

1. Credentials are NEVER read from EscalationConfig or from any file
   that also holds contact info — they come from environment variables
   only (SMTPConfig.from_env()), so a saved escalation_config.json can
   be safely shared/committed without leaking a password alongside it.

2. Sending is DRY-RUN BY DEFAULT. send_escalation() will only perform
   a real SMTP send when AURA_SEND_REAL_ESCALATIONS=1 is set in the
   environment — otherwise it prints exactly what it WOULD send and
   returns without touching the network. This exists specifically so
   that running a demo, a test, or an experiment never accidentally
   emails a real person. Flipping it on is a deliberate, one-line
   decision made at run time, not a code change.
"""

from __future__ import annotations
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Optional

from .escalate import EscalationEvent


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class SMTPConfig:
    """SMTP credentials and server details — always loaded from the
    environment, never from a config file that also holds contact info.

    Required environment variables:
        AURA_SMTP_USERNAME   — the sending account, e.g. you@gmail.com
        AURA_SMTP_PASSWORD   — an app password (see notes below), NOT
                                your normal account password

    Optional (sensible Gmail defaults if unset):
        AURA_SMTP_HOST         (default: smtp.gmail.com)
        AURA_SMTP_PORT         (default: 587)
        AURA_SMTP_FROM_ADDRESS (default: same as AURA_SMTP_USERNAME)
        AURA_SMTP_FROM_NAME    (default: "Aura-M1")

    Gmail specifically requires an App Password, not your regular
    password — regular-password SMTP login is disabled by Google for
    normal accounts. Generating one requires 2-Step Verification to be
    turned on for the Google account first.
    """
    host:         str
    port:         int
    username:     str
    password:     str
    from_address: str
    from_name:    str = "Aura-M1"

    @classmethod
    def from_env(cls) -> "SMTPConfig":
        username = os.environ.get("AURA_SMTP_USERNAME")
        password = os.environ.get("AURA_SMTP_PASSWORD")
        missing = [name for name, val in
                   [("AURA_SMTP_USERNAME", username), ("AURA_SMTP_PASSWORD", password)]
                   if not val]
        if missing:
            raise ValueError(
                "Missing required environment variable(s): " + ", ".join(missing) +
                ". Set them before sending escalation emails — see "
                "aura_sdk/recommend/notify.py for what each one means."
            )
        return cls(
            host         = os.environ.get("AURA_SMTP_HOST", "smtp.gmail.com"),
            port         = int(os.environ.get("AURA_SMTP_PORT", "587")),
            username     = username,
            password     = password,
            from_address = os.environ.get("AURA_SMTP_FROM_ADDRESS", username),
            from_name    = os.environ.get("AURA_SMTP_FROM_NAME", "Aura-M1"),
        )


# ── Notifiers ────────────────────────────────────────────────────────────

class SMTPEmailNotifier:
    """Sends one EscalationEvent as a plain-text email over SMTP+STARTTLS.

    Usage
    ─────
        notifier = SMTPEmailNotifier(SMTPConfig.from_env())
        ok = notifier.send(event, to_address="someone@example.com")
    """

    def __init__(self, config: SMTPConfig):
        self.config = config

    def send(self, event: EscalationEvent, to_address: str) -> bool:
        """Returns True on a confirmed send, False on any failure — never
        raises, since a failed notification must not crash the pipeline
        that's still trying to help the person it's monitoring."""
        subject = f"Aura-M1 check-in: {event.contact_name}"
        body = (
            f"{event.message}\n\n"
            f"— sent automatically by Aura-M1, tier {event.tier} of the "
            f"check-in chain, at frame {event.frame_idx}."
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = f"{self.config.from_name} <{self.config.from_address}>"
        msg["To"] = to_address

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.host, self.config.port, timeout=10) as server:
                server.starttls(context=context)
                server.login(self.config.username, self.config.password)
                server.sendmail(self.config.from_address, [to_address], msg.as_string())
            return True
        except Exception as e:
            print(f"[notify] Failed to send escalation email to {to_address}: {e}")
            return False


class DryRunNotifier:
    """Never touches the network — prints what would have been sent.
    This is the default path unless AURA_SEND_REAL_ESCALATIONS=1."""

    def send(self, event: EscalationEvent, to_address: str) -> bool:
        print(f"[DRY RUN — no email sent] Would email {to_address}:")
        print(f"    \"{event.message}\"")
        print(f"    (set AURA_SEND_REAL_ESCALATIONS=1 to actually send)")
        return False


# ── Entry point ──────────────────────────────────────────────────────────

def send_escalation(event: EscalationEvent, config: Optional[SMTPConfig] = None) -> bool:
    """
    Dispatches an EscalationEvent based on its contact_method.

    - "email:<address>"  -> sends via SMTP, unless AURA_SEND_REAL_ESCALATIONS
                             is unset/not "1", in which case it dry-runs.
    - "sms:<number>"      -> not implemented; logs clearly and returns False
                             rather than silently pretending to send.
    - anything else       -> logs clearly and returns False.
    """
    if event.contact_method.startswith("email:"):
        to_address = event.contact_method.split(":", 1)[1]
        if os.environ.get("AURA_SEND_REAL_ESCALATIONS") != "1":
            return DryRunNotifier().send(event, to_address)
        notifier = SMTPEmailNotifier(config if config is not None else SMTPConfig.from_env())
        return notifier.send(event, to_address)

    if event.contact_method.startswith("sms:"):
        print(f"[notify] SMS sending isn't implemented yet — {event.contact_name} "
              f"was NOT notified. (Would need an SMS gateway like Twilio.)")
        return False

    print(f"[notify] Unrecognized contact method '{event.contact_method}' — "
          f"{event.contact_name} was NOT notified.")
    return False
AURA_EOF_NOTIFY

echo "Updating aura_sdk/recommend/__init__.py ..."
python3 - << 'PYEOF'
import re
path = "aura_sdk/recommend/__init__.py"
src = open(path).read()
src = src.replace(
    "from .escalate import (\n    EscalationConfig, EscalationContact, EscalationEvent,\n    EscalationManager, EscalatingRecommender,\n)\n\n__all__ = [",
    "from .escalate import (\n    EscalationConfig, EscalationContact, EscalationEvent,\n    EscalationManager, EscalatingRecommender,\n)\nfrom .notify import SMTPConfig, SMTPEmailNotifier, DryRunNotifier, send_escalation\n\n__all__ = ["
)
src = src.replace(
    '"EscalationManager", "EscalatingRecommender",\n]',
    '"EscalationManager", "EscalatingRecommender",\n    "SMTPConfig", "SMTPEmailNotifier", "DryRunNotifier", "send_escalation",\n]'
)
open(path, "w").write(src)
PYEOF

echo "Creating aura_sdk_tests/test_recommend_notify.py ..."
cat > aura_sdk_tests/test_recommend_notify.py << 'AURA_EOF_TNOTIFY'
"""
aura_sdk_tests/test_recommend_notify.py
──────────────────────────────────────────
Language: Python (unittest)
Why: Sending is the highest-stakes action in this whole system — it
reaches a real person's inbox. These tests never touch a real network
(smtplib.SMTP is fully mocked) and focus on the two guarantees that
matter most: a real send only happens with explicit opt-in via
AURA_SEND_REAL_ESCALATIONS=1, and SMTPConfig never silently proceeds
with missing credentials.

Run: python -m unittest aura_sdk_tests.test_recommend_notify -v
"""
from __future__ import annotations
import os
import unittest
from unittest.mock import patch, MagicMock

from aura_sdk.recommend import (
    EscalationEvent, SMTPConfig, SMTPEmailNotifier, DryRunNotifier, send_escalation,
)


def make_event(method="email:harvey@example.com"):
    return EscalationEvent(
        frame_idx=42, unresolved_count=3, tier=0,
        contact_name="Harvey", contact_method=method,
        message="Automated check-in: recommended breaks haven't seemed to help.",
    )


class TestSMTPConfigFromEnv(unittest.TestCase):

    def test_missing_credentials_raises_with_clear_message(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                SMTPConfig.from_env()
            self.assertIn("AURA_SMTP_USERNAME", str(ctx.exception))
            self.assertIn("AURA_SMTP_PASSWORD", str(ctx.exception))

    def test_defaults_applied_when_optional_vars_unset(self):
        env = {"AURA_SMTP_USERNAME": "me@gmail.com", "AURA_SMTP_PASSWORD": "app-password"}
        with patch.dict(os.environ, env, clear=True):
            config = SMTPConfig.from_env()
        self.assertEqual(config.host, "smtp.gmail.com")
        self.assertEqual(config.port, 587)
        self.assertEqual(config.from_address, "me@gmail.com")
        self.assertEqual(config.from_name, "Aura-M1")

    def test_explicit_overrides_are_respected(self):
        env = {
            "AURA_SMTP_USERNAME": "me@gmail.com", "AURA_SMTP_PASSWORD": "app-password",
            "AURA_SMTP_HOST": "smtp.custom.example", "AURA_SMTP_PORT": "465",
            "AURA_SMTP_FROM_ADDRESS": "noreply@custom.example", "AURA_SMTP_FROM_NAME": "Custom Bot",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SMTPConfig.from_env()
        self.assertEqual(config.host, "smtp.custom.example")
        self.assertEqual(config.port, 465)
        self.assertEqual(config.from_address, "noreply@custom.example")
        self.assertEqual(config.from_name, "Custom Bot")


class TestSMTPEmailNotifier(unittest.TestCase):

    def _config(self):
        return SMTPConfig(host="smtp.gmail.com", port=587, username="me@gmail.com",
                           password="app-password", from_address="me@gmail.com")

    @patch("aura_sdk.recommend.notify.smtplib.SMTP")
    def test_successful_send_logs_in_and_sends(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        notifier = SMTPEmailNotifier(self._config())
        result = notifier.send(make_event(), to_address="harvey@example.com")

        self.assertTrue(result)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("me@gmail.com", "app-password")
        mock_server.sendmail.assert_called_once()
        args = mock_server.sendmail.call_args[0]
        self.assertEqual(args[0], "me@gmail.com")
        self.assertEqual(args[1], ["harvey@example.com"])
        self.assertIn("check-in", args[2])

    @patch("aura_sdk.recommend.notify.smtplib.SMTP")
    def test_smtp_failure_returns_false_without_raising(self, mock_smtp_cls):
        mock_smtp_cls.side_effect = Exception("connection refused")
        notifier = SMTPEmailNotifier(self._config())
        result = notifier.send(make_event(), to_address="harvey@example.com")
        self.assertFalse(result)  # must not raise — pipeline must survive this


class TestDryRunNotifier(unittest.TestCase):

    def test_dry_run_never_touches_network_and_returns_false(self):
        with patch("aura_sdk.recommend.notify.smtplib.SMTP") as mock_smtp_cls:
            result = DryRunNotifier().send(make_event(), to_address="harvey@example.com")
            mock_smtp_cls.assert_not_called()
        self.assertFalse(result)


class TestSendEscalationDispatch(unittest.TestCase):

    def test_email_defaults_to_dry_run_without_opt_in_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("aura_sdk.recommend.notify.smtplib.SMTP") as mock_smtp_cls:
                result = send_escalation(make_event())
                mock_smtp_cls.assert_not_called()
        self.assertFalse(result)

    def test_email_sends_for_real_only_with_explicit_opt_in(self):
        env = {
            "AURA_SEND_REAL_ESCALATIONS": "1",
            "AURA_SMTP_USERNAME": "me@gmail.com", "AURA_SMTP_PASSWORD": "app-password",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("aura_sdk.recommend.notify.smtplib.SMTP") as mock_smtp_cls:
                mock_server = MagicMock()
                mock_smtp_cls.return_value.__enter__.return_value = mock_server
                result = send_escalation(make_event())
        self.assertTrue(result)
        mock_server.sendmail.assert_called_once()

    def test_sms_is_never_sent_and_reports_false(self):
        with patch("aura_sdk.recommend.notify.smtplib.SMTP") as mock_smtp_cls:
            result = send_escalation(make_event(method="sms:+234-555-0100"))
            mock_smtp_cls.assert_not_called()
        self.assertFalse(result)

    def test_unrecognized_method_reports_false(self):
        result = send_escalation(make_event(method="carrier-pigeon:loft-3"))
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
AURA_EOF_TNOTIFY

echo "Wiring send_escalation() into run_with_escalation.py ..."
python3 - << 'PYEOF'
path = "run_with_escalation.py"
src = open(path).read()
src = src.replace(
    "from aura_sdk.recommend import RecommendationEngine, EscalationConfig, EscalatingRecommender, explain",
    "from aura_sdk.recommend import (\n    RecommendationEngine, EscalationConfig, EscalatingRecommender, explain,\n    send_escalation,\n)"
)
src = src.replace(
    'for i, c in enumerate(escalation_config.contacts):\n    print(f"  tier {i}: {c.name} ({c.method}) — notified at {c.threshold} unresolved outcomes")\nprint()',
    'for i, c in enumerate(escalation_config.contacts):\n    print(f"  tier {i}: {c.name} ({c.method}) — notified at {c.threshold} unresolved outcomes")\nsend_mode = "REAL SENDING ENABLED" if os.environ.get("AURA_SEND_REAL_ESCALATIONS") == "1" else "dry run (no real emails will be sent)"\nprint(f"  send mode: {send_mode}")\nprint()'
)
src = src.replace(
    "    if recommender.last_escalation:\n        escalations.append(recommender.last_escalation)",
    "    if recommender.last_escalation:\n        escalations.append(recommender.last_escalation)\n        send_escalation(recommender.last_escalation)"
)
open(path, "w").write(src)
PYEOF

echo ""
echo "Done. Running the full test suite now..."
python3 -m unittest discover -s aura_sdk_tests -v
