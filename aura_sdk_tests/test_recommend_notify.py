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
