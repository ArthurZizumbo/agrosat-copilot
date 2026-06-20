"""Drift alert notification resource (US-060, AC-6).

Models the "email alert when drift score exceeds the threshold" requirement as
a Dagster ``ConfigurableResource`` (the same injection pattern as the ``mlflow``
resource) rather than a loose function. The resource exposes a single
:meth:`DriftNotifier.send` method.

Configuration lives on the resource itself (env vars ``DRIFT_SMTP_*``), NOT in
``backend/app/core/config.py``: that ``Settings`` object (``extra="forbid"``) is
owned by the FastAPI backend, and Dagster runs as a separate process.

Behaviour:

- ``enabled=False`` (default in dev / CI): :meth:`send` logs a structured
  ``drift_alert`` event via ``structlog`` and returns ``False`` (delivered=no).
  Tests assert this path; no SMTP connection is attempted.
- ``enabled=True`` with SMTP config: :meth:`send` delivers the alert email and
  returns ``True``. Real SMTP delivery is exercised only in production
  (documented as not verified in dev in ``docs/blockers/epic10-notas.md``).
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

import structlog
from dagster import ConfigurableResource
from pydantic import Field

_log = structlog.get_logger(__name__)


class DriftNotifier(ConfigurableResource):
    """Sends drift alerts by email, with a structlog fallback in dev.

    Attributes:
        enabled: If ``False`` (default) no SMTP connection is made; alerts are
            only logged. Production sets ``DRIFT_SMTP_ENABLED=true``.
        smtp_host: SMTP server host (e.g. ``smtp.gmail.com``).
        smtp_port: SMTP server port (default ``587`` STARTTLS).
        smtp_user: SMTP auth user; empty disables authentication.
        smtp_password: SMTP auth password / app token; empty disables auth.
        sender: ``From`` address.
        recipients: Comma-separated list of recipient addresses.
    """

    enabled: bool = Field(default=False, description="Habilita el envio SMTP real.")
    smtp_host: str = Field(default="", description="Host del servidor SMTP.")
    smtp_port: int = Field(default=587, description="Puerto SMTP (587 STARTTLS).")
    smtp_user: str = Field(default="", description="Usuario SMTP (vacio = sin auth).")
    smtp_password: str = Field(default="", description="Password/token SMTP (vacio = sin auth).")
    sender: str = Field(default="agrosat-drift@localhost", description="Remitente del alert.")
    recipients: str = Field(default="", description="Destinatarios separados por coma.")

    def send(self, subject: str, body: str) -> bool:
        """Send a drift alert, or log it when disabled / misconfigured.

        Args:
            subject: Email subject line.
            body: Plain-text email body (drift score, week, report URL).

        Returns:
            ``True`` if an email was actually delivered, ``False`` if the alert
            was only logged (disabled, missing host or no recipients). Never
            raises: a delivery failure is logged and returns ``False`` so a drift
            run is not aborted by a notification outage.
        """
        recipients = [r.strip() for r in self.recipients.split(",") if r.strip()]
        if not self.enabled or not self.smtp_host or not recipients:
            _log.info(
                "drift_alert",
                delivered=False,
                reason="notifier_disabled_or_unconfigured",
                subject=subject,
                body=body,
                enabled=self.enabled,
                has_host=bool(self.smtp_host),
                n_recipients=len(recipients),
            )
            return False

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.sender
        message["To"] = ", ".join(recipients)
        message.set_content(body)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as client:
                client.starttls()
                if self.smtp_user and self.smtp_password:
                    client.login(self.smtp_user, self.smtp_password)
                client.send_message(message)
        except (OSError, smtplib.SMTPException) as exc:  # pragma: no cover - real SMTP only
            _log.warning(
                "drift_alert",
                delivered=False,
                reason="smtp_error",
                error=str(exc),
                subject=subject,
            )
            return False

        _log.info("drift_alert", delivered=True, subject=subject, n_recipients=len(recipients))
        return True


def build_drift_notifier() -> DriftNotifier:
    """Build the default ``DriftNotifier`` from ``DRIFT_SMTP_*`` env vars.

    Every field is read eagerly from the environment of the Dagster process
    (no secret is ever inlined in code). When the vars are absent the notifier
    is disabled and dev-safe: :meth:`DriftNotifier.send` only logs. In
    production the SMTP credentials are injected via the environment of the
    code-location container / Secret Manager.

    Returns:
        A configured ``DriftNotifier`` (disabled by default in dev / CI).
    """
    import os

    enabled = os.environ.get("DRIFT_SMTP_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    try:
        port = int(os.environ.get("DRIFT_SMTP_PORT", "587"))
    except ValueError:
        port = 587

    return DriftNotifier(
        enabled=enabled,
        smtp_host=os.environ.get("DRIFT_SMTP_HOST", ""),
        smtp_port=port,
        smtp_user=os.environ.get("DRIFT_SMTP_USER", ""),
        smtp_password=os.environ.get("DRIFT_SMTP_PASSWORD", ""),
        sender=os.environ.get("DRIFT_SMTP_SENDER", "agrosat-drift@localhost"),
        recipients=os.environ.get("DRIFT_SMTP_RECIPIENTS", ""),
    )
