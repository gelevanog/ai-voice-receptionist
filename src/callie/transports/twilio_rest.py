"""Twilio REST calls Callie makes: send an SMS, and redirect a live call to a human (`<Dial>`).

Without credentials both run in dry-run mode: the SMS lands in the outbox table as `dry_run` and the transfer
is logged, so the demo and the tests behave the same as production minus the network call. No live Twilio
account was used to build this; the request shapes follow Twilio's REST API and are tested against a mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape, quoteattr

import httpx

from callie.logging_config import get_logger

log = get_logger(__name__)
API = "https://api.twilio.com/2010-04-01"


@dataclass(frozen=True)
class SmsResult:
    status: str  # sent | dry_run | failed
    provider_id: str | None = None
    error: str | None = None


def transfer_twiml(number: str, message: str = "Connecting you now.") -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><Response><Say>{escape(message)}</Say>'
        f"<Dial timeout={quoteattr('25')}>{escape(number)}</Dial>"
        "<Say>Sorry, nobody could answer. Please call back during opening hours. Goodbye.</Say></Response>"
    )


class TwilioRest:
    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.account_sid = account_sid
        self.from_number = from_number
        self.enabled = bool(account_sid and auth_token and from_number)
        self._client = httpx.Client(
            base_url=f"{API}/Accounts/{account_sid}",
            auth=(account_sid, auth_token),
            timeout=timeout,
            transport=transport,
        )

    def send_sms(self, to: str, body: str) -> SmsResult:
        if not self.enabled:
            log.info("sms.dry_run", chars=len(body))
            return SmsResult("dry_run")
        try:
            response = self._client.post("/Messages.json", data={"To": to, "From": self.from_number, "Body": body})
        except httpx.HTTPError as exc:
            return SmsResult("failed", error=str(exc)[:200])
        if response.status_code >= 400:
            return SmsResult("failed", error=f"{response.status_code} {response.text[:200]}")
        return SmsResult("sent", provider_id=str(response.json().get("sid")))

    def redirect_call(self, call_sid: str, twiml: str) -> bool:
        """Replace a live call's instructions (ends the media stream, then runs the new TwiML)."""
        if not self.enabled or not call_sid:
            log.info("twilio.transfer.dry_run", call_sid=call_sid)
            return False
        response = self._client.post(f"/Calls/{call_sid}.json", data={"Twiml": twiml})
        if response.status_code >= 400:
            log.error("twilio.transfer.failed", status=response.status_code, body=response.text[:200])
            return False
        return True

    def hangup(self, call_sid: str) -> bool:
        if not self.enabled or not call_sid:
            return False
        response = self._client.post(f"/Calls/{call_sid}.json", data={"Status": "completed"})
        return response.status_code < 400
