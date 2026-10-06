"""Copy evaluation calls into a dashboard database, so the call history and calendar show real runs."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from callie.scheduling.db import Appointment, CallRecord, make_session_factory


def import_calls(source: Path, target: Path) -> dict[str, int]:
    destination = make_session_factory(f"sqlite:///{target}")
    calls = bookings = skipped = 0
    for path in sorted(source.glob("*.db")):
        origin = make_session_factory(f"sqlite:///{path}")
        with origin() as src:
            records = src.scalars(select(CallRecord)).all()
            appointments = src.scalars(select(Appointment).where(Appointment.source == "call")).all()
            for row in (*records, *appointments):
                src.expunge(row)
        with destination() as dst, dst.begin():
            for record in records:
                if dst.get(CallRecord, record.id) is None:
                    dst.merge(record)
                    calls += 1
            for appointment in appointments:
                overlap = dst.scalars(
                    select(Appointment).where(
                        Appointment.resource_id == appointment.resource_id,
                        Appointment.status == "booked",
                        Appointment.start < appointment.end,
                        Appointment.end > appointment.start,
                    )
                ).first()
                if overlap is not None or dst.get(Appointment, appointment.id) is not None:
                    skipped += 1
                    continue
                dst.merge(appointment)
                bookings += 1
    return {"calls": calls, "bookings": bookings, "skipped_overlapping_bookings": skipped}
