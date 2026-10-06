"""Optional Google Calendar connector (Calendar API v3 over plain HTTPS).

- busy times from Google (`freeBusy`) block slots in Callie's calendar, so the clinic can keep using the
  calendar its staff already looks at;
- every booking, reschedule and cancellation made by Callie is mirrored as an event.

Authentication is an OAuth access token (`CALLIE_GOOGLE_ACCESS_TOKEN`), e.g. from a service account that the
calendar is shared with. Token refresh is left to the deployment (a sidecar or `google-auth`), which keeps
this module dependency-free. The tests run it against a mocked transport; no live Google account was used.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from callie.logging_config import get_logger

log = get_logger(__name__)
API = "https://www.googleapis.com/calendar/v3"


class GoogleCalendarError(RuntimeError):
    pass


class GoogleCalendarClient:
    def __init__(
        self,
        calendar_id: str,
        access_token: str,
        *,
        timezone: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.calendar_id = calendar_id
        self.timezone = timezone
        self._client = httpx.Client(
            base_url=API,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=timeout,
            transport=transport,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = self._client.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise GoogleCalendarError(f"Google Calendar {method} {url}: {response.status_code} {response.text[:200]}")
        if not response.content:
            return {}
        data: dict[str, Any] = response.json()
        return data

    def busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Busy intervals from Google in [start, end).

        Errors raise instead of returning "no busy time": offering no slot is better than a double booking.
        """
        body = {
            "timeMin": start.astimezone(UTC).isoformat(),
            "timeMax": end.astimezone(UTC).isoformat(),
            "items": [{"id": self.calendar_id}],
        }
        data = self._request("POST", "/freeBusy", json=body)
        calendar = data.get("calendars", {}).get(self.calendar_id, {})
        if calendar.get("errors"):
            raise GoogleCalendarError(f"freeBusy error: {calendar['errors']}")
        return [
            (datetime.fromisoformat(item["start"]), datetime.fromisoformat(item["end"]))
            for item in calendar.get("busy", [])
        ]

    def create_event(self, summary: str, start: datetime, end: datetime, description: str = "") -> str:
        body = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": self.timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": self.timezone},
        }
        data = self._request("POST", f"/calendars/{self.calendar_id}/events", json=body)
        event_id = str(data.get("id", ""))
        if not event_id:
            raise GoogleCalendarError("event created without an id")
        return event_id

    def move_event(self, event_id: str, start: datetime, end: datetime) -> None:
        body = {
            "start": {"dateTime": start.isoformat(), "timeZone": self.timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": self.timezone},
        }
        self._request("PATCH", f"/calendars/{self.calendar_id}/events/{event_id}", json=body)

    def delete_event(self, event_id: str) -> None:
        self._request("DELETE", f"/calendars/{self.calendar_id}/events/{event_id}")
