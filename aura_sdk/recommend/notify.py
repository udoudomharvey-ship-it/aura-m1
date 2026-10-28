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
