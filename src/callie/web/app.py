"""FastAPI app: browser calls, Twilio webhook and media stream, dashboard pages and a small JSON API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from callie.config import Settings, get_settings
from callie.logging_config import configure_logging, get_logger
from callie.privacy import mask_name
from callie.runtime import Runtime, build_runtime
from callie.scheduling.db import CallRecord
from callie.scheduling.timeparse import speak_date, speak_time
from callie.speech import SpeechStack, build_speech, models_present
from callie.transports.browser import serve_browser_call
from callie.transports.twilio import serve_twilio_stream, stream_twiml, validate_signature

log = get_logger(__name__)
WEB_DIR = Path(__file__).parent
DEMO_SCRIPT = [
    "Hi, I'd like to book a cleaning next Tuesday after lunch.",
    "The 3 PM one, please.",
    "It's Jane Doe, and my number is 555 123 4567.",
    "Yes, that's right.",
    "Do you take Delta Dental?",
    "No, that's all. Thanks!",
]


def create_app(
    settings: Settings | None = None, *, runtime: Runtime | None = None, speech: SpeechStack | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    runtime = runtime or build_runtime(settings)
    speech = speech or build_speech(settings, fake_lines=list(DEMO_SCRIPT * 3))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Load models in the background so the first call does not wait and the dashboard is up at once.
        async def load() -> None:
            for component in (speech.stt, speech.tts):
                loader = getattr(component, "load", None)
                if loader is not None:
                    try:
                        await asyncio.to_thread(loader)
                    except Exception as exc:
                        log.error("model.load_failed", component=type(component).__name__, error=str(exc)[:200])

        task = asyncio.get_running_loop().create_task(load())
        yield
        task.cancel()

    app = FastAPI(
        title="Callie", description="AI voice receptionist: browser and Twilio calls, dashboard.", lifespan=lifespan
    )
    app.state.runtime = runtime
    app.state.speech = speech
    templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

    def page(request: Request, name: str, **context: Any) -> HTMLResponse:
        base = {"request": request, "stack": stack(), "clinic": runtime.clinic, "page": name.split(".")[0]}
        return templates.TemplateResponse(request, name, {**base, **context})

    def stack() -> dict[str, str]:
        return {**speech.describe(), "llm": runtime.llm.label}

    # -- calls -------------------------------------------------------------------------------------------------
    @app.websocket("/ws/call")
    async def browser_call(websocket: WebSocket) -> None:
        await serve_browser_call(websocket, runtime, speech)

    @app.post("/twilio/voice")
    async def twilio_voice(request: Request) -> Response:
        form = {key: str(value) for key, value in (await request.form()).items()}
        if settings.twilio_validate_signature and settings.twilio_auth_token:
            url = _public_url(request, settings)
            if not validate_signature(settings.twilio_auth_token, url, form, request.headers.get("X-Twilio-Signature")):
                return PlainTextResponse("invalid signature", status_code=403)
        base = settings.public_base_url or str(request.base_url).rstrip("/")
        stream_url = base.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/twilio/media"
        twiml = stream_twiml(stream_url, caller=form.get("From"), call_sid=form.get("CallSid"))
        return Response(content=twiml, media_type="application/xml")

    @app.websocket("/twilio/media")
    async def twilio_media(websocket: WebSocket) -> None:
        await serve_twilio_stream(websocket, runtime, speech)

    # -- dashboard ---------------------------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def live_call(request: Request) -> HTMLResponse:
        return page(request, "call.html", fake_stt=speech.stt.name.startswith("fake"), demo_script=DEMO_SCRIPT)

    @app.get("/calls", response_class=HTMLResponse)
    async def calls(request: Request) -> HTMLResponse:
        with runtime.sessions() as db:
            rows = db.scalars(select(CallRecord).order_by(CallRecord.started_at.desc()).limit(200)).all()
        return page(request, "calls.html", calls=[_call_row(r, runtime) for r in rows])

    @app.get("/calls/{call_id}", response_class=HTMLResponse)
    async def call_detail(request: Request, call_id: str) -> HTMLResponse:
        with runtime.sessions() as db:
            record = db.get(CallRecord, call_id)
        if record is None:
            return HTMLResponse("call not found", status_code=404)
        return page(request, "call_detail.html", call=_call_row(record, runtime), record=record)

    @app.get("/calls/{call_id}/recording.wav")
    async def recording(call_id: str) -> Response:
        with runtime.sessions() as db:
            record = db.get(CallRecord, call_id)
        path = _recording_file(record)
        if path is None:
            return PlainTextResponse("no recording", status_code=404)
        return FileResponse(path, media_type="audio/wav")

    @app.get("/calendar", response_class=HTMLResponse)
    async def calendar_page(request: Request, week: str | None = Query(default=None)) -> HTMLResponse:
        today = runtime.now().date()
        start = date.fromisoformat(week) if week else today
        monday = start - timedelta(days=start.weekday())
        return page(request, "calendar.html", **_week(runtime, monday))

    @app.get("/eval", response_class=HTMLResponse)
    async def eval_page(request: Request) -> HTMLResponse:
        return page(request, "eval.html", results=_load_results(settings.results_dir))

    # -- API ---------------------------------------------------------------------------------------------------
    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {"status": "ok", "stack": stack(), "models": models_present(settings), "clinic": runtime.clinic.name}
        )

    @app.get("/api/calls")
    async def api_calls() -> JSONResponse:
        with runtime.sessions() as db:
            rows = db.scalars(select(CallRecord).order_by(CallRecord.started_at.desc()).limit(100)).all()
        return JSONResponse([_call_row(r, runtime) for r in rows])

    @app.get("/api/appointments")
    async def api_appointments(start: str | None = None, days: int = 7) -> JSONResponse:
        first = date.fromisoformat(start) if start else runtime.now().date()
        begin = datetime.combine(first, time(0, 0), tzinfo=runtime.clinic.tz)
        rows = runtime.calendar.between(begin, begin + timedelta(days=days))
        return JSONResponse(
            [
                {
                    "id": r.id,
                    "service": r.service_id,
                    "resource": r.resource_id,
                    "start": r.start.isoformat(),
                    "end": r.end.isoformat(),
                    "patient": mask_name(r.patient_name),
                    "source": r.source,
                }
                for r in rows
            ]
        )

    return app


def _recording_file(record: CallRecord | None) -> Path | None:
    if record is None or not record.recording_path:
        return None
    path = Path(record.recording_path)
    return path if path.exists() else None


def _public_url(request: Request, settings: Settings) -> str:
    """The URL Twilio signed: the public base URL when behind a tunnel or proxy, else what we received."""
    if settings.public_base_url:
        query = f"?{request.url.query}" if request.url.query else ""
        return settings.public_base_url.rstrip("/") + request.url.path + query
    return str(request.url)


def _call_row(record: CallRecord, runtime: Runtime) -> dict[str, Any]:
    v2v = [
        t["voice_to_voice_ms"]
        for t in record.turns or []
        if t.get("voice_to_voice_ms") is not None and not t.get("typed")
    ]
    started = record.started_at.astimezone(runtime.clinic.tz)
    return {
        "id": record.id,
        "started": started.strftime("%b %d, %H:%M"),
        "transport": record.transport,
        "duration": record.duration_s,
        "outcome": record.outcome,
        "turns": len([e for e in record.transcript or [] if e.get("role") == "caller"]),
        "tools": len(record.tool_calls or []),
        "v2v_median": sorted(v2v)[len(v2v) // 2] if v2v else None,
        "llm": (record.stack or {}).get("llm", ""),
        "scenario": record.scenario,
        "has_recording": bool(record.recording_path and Path(record.recording_path).exists()),
        "caller": record.caller or "",
    }


def _week(runtime: Runtime, monday: date) -> dict[str, Any]:
    tz = runtime.clinic.tz
    begin = datetime.combine(monday, time(0, 0), tzinfo=tz)
    rows = runtime.calendar.between(begin, begin + timedelta(days=6))
    days = []
    for offset in range(6):
        day = monday + timedelta(days=offset)
        intervals = runtime.clinic.intervals(day)
        items = []
        for row in rows:
            start = row.start.astimezone(tz)
            if start.date() != day:
                continue
            service = runtime.clinic.service(row.service_id)
            items.append(
                {
                    "id": row.id,
                    "start": speak_time(start.time()),
                    "top": (start.hour - 7) * 60 + start.minute,
                    "height": int((row.end - row.start).total_seconds() // 60),
                    "service": service.name if service else row.service_id,
                    "resource": row.resource_id,
                    "patient": _initials(row.patient_name),
                    "source": row.source,
                }
            )
        days.append(
            {
                "date": day,
                "label": speak_date(day).split(",")[0][:3] + " " + str(day.day),
                "appts": items,
                "open": [((s.hour - 7) * 60 + s.minute, int((e - s).total_seconds() // 60)) for s, e in intervals],
                "is_today": day == runtime.now().date(),
            }
        )
    booked_by_callie = sum(1 for r in rows if r.source == "call")
    return {
        "days": days,
        "monday": monday,
        "prev": (monday - timedelta(days=7)).isoformat(),
        "next": (monday + timedelta(days=7)).isoformat(),
        "hours": list(range(7, 20)),
        "count": len(rows),
        "booked_by_callie": booked_by_callie,
        "title": f"Week of {speak_date(monday).split(', ', 1)[1]}",
    }


def _initials(name: str) -> str:
    parts = name.split()
    return " ".join([parts[0][0] + ".", *(p[0] + "." for p in parts[1:])]) if parts else ""


def _load_results(directory: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("summary", "e2e", "wer", "bargein", "smoke_models", "tts_bench", "calls_summary"):
        path = directory / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    out["generated"] = datetime.now(UTC).isoformat(timespec="seconds")
    return out


def create_default_app() -> FastAPI:
    return create_app()
