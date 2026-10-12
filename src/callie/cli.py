"""`callie` command line: serve | call | simulate | eval | models | calls | download-models | db."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from callie.config import Settings, get_settings
from callie.llm.base import ChatModel

app = typer.Typer(help="Callie: an AI voice receptionist.", no_args_is_help=True, add_completion=False)
eval_app = typer.Typer(help="Evaluation with simulated callers.", no_args_is_help=True)
models_app = typer.Typer(help="Models: free OpenRouter models, smoke tests.", no_args_is_help=True)
db_app = typer.Typer(help="Database: seed or reset demo data.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
app.add_typer(models_app, name="models")
app.add_typer(db_app, name="db")
console = Console()


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes")] = False,
) -> None:
    """Dashboard, browser calls (/ws/call) and the Twilio webhook (/twilio/voice) on one server."""
    import uvicorn

    uvicorn.run("callie.web.app:create_default_app", factory=True, host=host, port=port, reload=reload)


@app.command()
def call(
    lines: Annotated[list[str] | None, typer.Argument(help="Caller lines; omit for an interactive chat")] = None,
    provider: Annotated[str | None, typer.Option(help="LLM provider (fake, openrouter, openai, anthropic)")] = None,
    model: Annotated[str | None, typer.Option(help="LLM model id")] = None,
    wav: Annotated[Path | None, typer.Option(help="Send a WAV file through the full audio pipeline instead")] = None,
    out: Annotated[Path, typer.Option(help="Where --wav writes the call recording")] = Path("data/call.wav"),
) -> None:
    """Talk to the agent in the terminal (text), or run a recorded caller WAV through VAD -> STT -> agent -> TTS."""
    settings = _settings(provider=provider, model=model)
    if wav is not None:
        from callie.eval.simulator import run_wav_call

        summary = asyncio.run(run_wav_call(settings, wav, out))
        console.print_json(
            json.dumps({k: summary[k] for k in ("call_id", "outcome", "transcript", "turns")}, default=str)
        )
        console.print(f"recording: {out}")
        return
    asyncio.run(_text_call(settings, lines))


async def _text_call(settings: Settings, lines: list[str] | None) -> None:
    from callie.agent.agent import CallAction, Sentence, ToolEvent
    from callie.runtime import build_runtime

    runtime = build_runtime(settings)
    agent = runtime.new_agent(runtime.new_context())
    console.print(f"[bold cyan]Callie:[/] {agent.greeting()}")
    queue = list(lines or [])
    while True:
        if lines is not None:
            if not queue:
                break
            text = queue.pop(0)
            console.print(f"[bold]Caller:[/] {text}")
        else:
            try:
                text = console.input("[bold]Caller:[/] ")
            except (EOFError, KeyboardInterrupt):
                break
        ended = False
        async for event in agent.respond(text):
            if isinstance(event, Sentence):
                console.print(
                    f"[bold cyan]Callie:[/] {event.text}" + (" [dim](filler)[/]" if event.source == "filler" else "")
                )
            elif isinstance(event, ToolEvent):
                arguments = json.dumps(event.result.arguments)
                console.print(f"  [dim]tool {event.result.name} {arguments} -> {event.result.data.get('status')}[/]")
            elif isinstance(event, CallAction):
                console.print(f"  [yellow]{event.kind}[/] {event.reason}")
                ended = True
        if ended:
            break
    console.print(f"outcome: [bold]{agent.ctx.outcome.label()}[/]")


@app.command()
def simulate(
    scenario: Annotated[str, typer.Argument(help="Scenario id (see `callie eval scenarios`)")],
    provider: Annotated[str | None, typer.Option(help="Agent LLM provider")] = None,
    model: Annotated[str | None, typer.Option(help="Agent LLM model id")] = None,
    caller: Annotated[str, typer.Option(help="Caller: scripted (no API calls) or llm")] = "scripted",
    channel: Annotated[str | None, typer.Option(help="Override the channel: clean | phone | phone_noisy")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the call summary JSON here")] = None,
) -> None:
    """One simulated caller, end to end through audio (caller TTS -> channel -> VAD -> STT -> agent -> TTS)."""
    from callie.eval.simulator import simulate_one

    settings = _settings(provider=provider, model=model)
    result = asyncio.run(simulate_one(settings, scenario, caller_mode=caller, channel=channel))
    console.print_json(json.dumps(result["check"], default=str))
    for entry in result["summary"]["transcript"]:
        who = "[bold]Caller[/]" if entry["role"] == "caller" else "[bold cyan]Callie[/]"
        console.print(f"{entry['t']:>6.1f}s {who}: {entry['text']}")
    if out:
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


@eval_app.command("scenarios")
def eval_scenarios() -> None:
    """List the caller scenarios."""
    from callie.eval.scenarios import load_scenarios

    table = Table("id", "tier", "category", "channel", "expect", "goal")
    for s in load_scenarios():
        table.add_row(s.id, s.tier, s.category, s.channel, s.expect.outcome, s.goal[:60])
    console.print(table)


@eval_app.command("run")
def eval_run(
    provider: Annotated[str | None, typer.Option(help="Agent LLM provider")] = None,
    model: Annotated[str | None, typer.Option(help="Agent LLM model id")] = None,
    fallback: Annotated[list[str] | None, typer.Option(help="Agent fallback model ids")] = None,
    caller: Annotated[str, typer.Option(help="scripted | llm")] = "llm",
    caller_model: Annotated[
        str, typer.Option(help="Caller LLM model id (OpenRouter, :free)")
    ] = "liquid/lfm-2.5-2.6b:free",
    caller_provider: Annotated[
        str, typer.Option(help="Caller LLM provider; anything but free OpenRouter needs CALLIE_ALLOW_PAID_MODELS=1")
    ] = "openrouter",
    caller_temperature: Annotated[
        bool, typer.Option(help="Send a temperature with caller requests (some models reject it)")
    ] = True,
    only: Annotated[list[str] | None, typer.Option(help="Run only these scenario ids")] = None,
    out_dir: Annotated[Path, typer.Option(help="Results directory")] = Path("results"),
    tag: Annotated[str, typer.Option(help="Name of this run (file prefix)")] = "e2e",
    include_all: Annotated[bool, typer.Option("--all", help="Also the extended scenarios (more API calls)")] = False,
    cache: Annotated[bool, typer.Option(help="Replay identical requests from the disk cache")] = True,
) -> None:
    """Simulated callers end to end through audio; writes results/<tag>.json and the call recordings."""
    from callie.eval.run import run_e2e

    settings = _settings(provider=provider, model=model, fallbacks=fallback).model_copy(update={"llm_cache": cache})
    asyncio.run(
        run_e2e(
            settings,
            caller_mode=caller,
            caller_model=caller_model,
            only=only,
            out_dir=out_dir,
            tag=tag,
            include_extended=include_all,
            caller_provider=caller_provider,
            caller_send_temperature=caller_temperature,
        )
    )


@eval_app.command("text")
def eval_text(
    provider: Annotated[str | None, typer.Option(help="Agent LLM provider")] = None,
    model: Annotated[str | None, typer.Option(help="Agent LLM model id")] = None,
    fallback: Annotated[list[str] | None, typer.Option(help="Agent fallback model ids")] = None,
    caller: Annotated[str, typer.Option(help="scripted | llm")] = "llm",
    caller_model: Annotated[str, typer.Option(help="Caller LLM model id")] = "liquid/lfm-2.5-2.6b:free",
    caller_provider: Annotated[
        str, typer.Option(help="Caller LLM provider; anything but free OpenRouter needs CALLIE_ALLOW_PAID_MODELS=1")
    ] = "openrouter",
    caller_temperature: Annotated[
        bool, typer.Option(help="Send a temperature with caller requests (some models reject it)")
    ] = True,
    only: Annotated[list[str] | None, typer.Option(help="Run only these scenario ids")] = None,
    out: Annotated[Path, typer.Option(help="Result file (records in the `eval run` format)")] = Path(
        "results/text.json"
    ),
    include_all: Annotated[bool, typer.Option("--all", help="Also the extended scenarios (more API calls)")] = False,
) -> None:
    """Text-level conversation eval: the same scenarios, caller and checks without audio (only the LLM is measured)."""
    from callie.clinic import load_clinic
    from callie.eval.caller import build_caller_llm, ensure_agent_allowed
    from callie.eval.text_run import run_text_eval, text_mode_scenarios
    from callie.llm.factory import build_chat_model, shared_ledger

    settings = _settings(provider=provider, model=model, fallbacks=fallback)
    clinic = load_clinic(settings.clinic_file)
    ensure_agent_allowed(settings)
    caller_llm = (
        build_caller_llm(
            settings, clinic, provider=caller_provider, model=caller_model, send_temperature=caller_temperature
        )
        if caller == "llm"
        else None
    )
    agent_llm = build_chat_model(settings, clinic, tag="agent")
    data = asyncio.run(
        run_text_eval(
            settings,
            text_mode_scenarios(include_extended=include_all, only=only),
            agent_llm=agent_llm,
            caller_llm=caller_llm,
            caller_label=f"{caller_provider}/{caller_model}" if caller == "llm" else "scripted",
            ledger=shared_ledger(settings),
            out_path=out,
        )
    )
    passed = sum(r["passed"] for r in data.get("scenarios", []))
    console.print(f"passed {passed}/{len(data.get('scenarios', []))} -> {out}")


@eval_app.command("wer")
def eval_wer(
    models: Annotated[list[str] | None, typer.Option(help="faster-whisper models")] = None,
    out_dir: Annotated[Path, typer.Option()] = Path("results"),
) -> None:
    """STT word error rate on the synthesized caller utterances: clean vs phone line vs phone line + noise."""
    from callie.eval.wer_bench import run_wer

    run_wer(_settings(), models or ["tiny.en", "base.en", "small.en"], out_dir)


@eval_app.command("bargein")
def eval_bargein(
    trials: Annotated[int, typer.Option(help="Interruptions and backchannels to simulate, each")] = 20,
    out_dir: Annotated[Path, typer.Option()] = Path("results"),
) -> None:
    """Barge-in reaction time and backchannel handling with real VAD and STT (fake LLM, no API calls)."""
    from callie.eval.bargein import run_bargein

    asyncio.run(run_bargein(_settings(), trials, out_dir))


@eval_app.command("tts")
def eval_tts(out_dir: Annotated[Path, typer.Option()] = Path("results")) -> None:
    """TTS speed on this CPU (Kokoro fp32 vs int8 vs Piper) and Whisper WER on Callie's own voice."""
    from callie.eval.tts_bench import run_tts_bench

    console.print_json(json.dumps(run_tts_bench(_settings(), out_dir), default=str))


@eval_app.command("report")
def eval_report(out_dir: Annotated[Path, typer.Option()] = Path("results")) -> None:
    """Combine the result files into results/summary.json and results/report.md."""
    from callie.eval.report import build_report

    console.print(build_report(out_dir))


@eval_app.command("import-calls")
def eval_import_calls(
    db: Annotated[Path, typer.Option(help="Target SQLite database (the dashboard's)")] = Path("data/callie.db"),
    source: Annotated[Path, typer.Option(help="Per-scenario databases written by `eval run`")] = Path("data/eval/db"),
) -> None:
    """Copy the evaluation's call records (masked) and Callie's bookings into a dashboard database."""
    from callie.eval.importer import import_calls

    console.print(import_calls(source, db))


@eval_app.command("export-demo")
def eval_export_demo(
    scenario: Annotated[str, typer.Argument(help="Scenario id whose recording to export")],
    name: Annotated[str, typer.Option(help="Output file name (without extension)")],
    title: Annotated[str, typer.Option(help="Title of the transcript")],
    out_dir: Annotated[Path, typer.Option()] = Path("docs/demo-calls"),
    results: Annotated[Path, typer.Option()] = Path("results/e2e.json"),
) -> None:
    """Export one evaluation call: MP3 of both sides and a timestamped transcript (from the recording)."""
    from callie.eval.demo_calls import export_call

    data = json.loads(results.read_text(encoding="utf-8"))
    record = next(r for r in data["scenarios"] if r["scenario"] == scenario)
    console.print(export_call(Path(record["recording"]), out_dir, name, title))


@models_app.command("free")
def models_free(
    smoke: Annotated[int, typer.Option(help="Smoke-test tool calling and latency on the first N candidates")] = 0,
    candidates: Annotated[list[str] | None, typer.Option(help="Model ids to smoke-test")] = None,
) -> None:
    """List free OpenRouter models (ids ending in :free) and optionally smoke-test them."""
    from callie.eval.smoke import list_and_smoke

    asyncio.run(list_and_smoke(_settings(), smoke, candidates))


@app.command()
def calls(ledger: Annotated[Path, typer.Option()] = Path("results/calls.jsonl")) -> None:
    """Every real LLM request in the ledger, by tag, status and model."""
    from callie.eval.report import ledger_summary

    console.print_json(json.dumps(ledger_summary(ledger)))


@app.command("download-models")
def download_models_cmd(
    piper: Annotated[bool, typer.Option(help="Also the Piper caller voice (evaluation only)")] = False,
    whisper: Annotated[list[str] | None, typer.Option(help="faster-whisper models")] = None,
) -> None:
    """Silero VAD, Kokoro-82M and faster-whisper weights (and optionally a Piper voice) into the model cache."""
    from callie.speech import download_models

    download_models(get_settings(), piper=piper, whisper=whisper)


@db_app.command("seed")
def db_seed() -> None:
    """Create the database and fill the next three weeks with fictional appointments."""
    from callie.runtime import build_runtime

    runtime = build_runtime(get_settings().model_copy(update={"seed_demo_data": True}), llm=_fake_llm())
    now = runtime.now()
    upcoming = runtime.calendar.between(now, now + timedelta(days=21))
    console.print(f"appointments in the next 3 weeks: {len(upcoming)}")


@db_app.command("reset")
def db_reset() -> None:
    """Delete the SQLite database file (demo data is re-seeded on the next start)."""
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        path = Path(url.removeprefix("sqlite:///"))
        for suffix in ("", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
        console.print(f"removed {path}")
    else:
        console.print("only SQLite databases are reset by this command")


def _fake_llm() -> ChatModel:
    from callie.clinic import load_clinic
    from callie.llm.fake import FakeReceptionist

    return FakeReceptionist(load_clinic(get_settings().clinic_file))


def _settings(*, provider: str | None = None, model: str | None = None, fallbacks: list[str] | None = None) -> Settings:
    settings = get_settings()
    update: dict[str, object] = {}
    if provider:
        update["llm_provider"] = provider
    if model:
        update["llm_model"] = model
    if fallbacks:
        update["llm_fallback_models"] = fallbacks
    return settings.model_copy(update=update) if update else settings


if __name__ == "__main__":
    app()
