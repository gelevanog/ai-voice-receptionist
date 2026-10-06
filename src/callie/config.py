"""Runtime settings: environment variables with the `CALLIE_` prefix (see `.env.example`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["fake", "openrouter", "openai", "anthropic"]
STTProviderName = Literal["fake", "whisper"]
TTSProviderName = Literal["fake", "kokoro", "piper"]
VADProviderName = Literal["energy", "silero"]

DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def default_models_dir() -> Path:
    return Path.home() / ".cache" / "callie" / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CALLIE_", env_file=".env", extra="ignore")

    # --- providers -------------------------------------------------------------------------------------------
    llm_provider: LLMProviderName = "fake"
    llm_model: str = ""
    llm_fallback_models: list[str] = Field(default_factory=list)
    llm_temperature: float = 0.3
    llm_max_tokens: int = 400
    llm_timeout_seconds: float = 60.0
    require_free_models: bool = True
    stt_provider: STTProviderName = "fake"
    stt_model: str = "base.en"
    stt_threads: int = 4
    tts_provider: TTSProviderName = "fake"
    tts_voice: str = "af_heart"
    tts_speed: float = 1.05
    tts_threads: int = 4
    vad_provider: VADProviderName = "energy"
    models_dir: Path = Field(default_factory=default_models_dir)

    # --- turn-taking -----------------------------------------------------------------------------------------
    vad_threshold: float = 0.5
    endpoint_silence_ms: int = 550
    min_speech_ms: int = 120
    barge_in_min_speech_ms: int = 250
    hard_interrupt_ms: int = 1200
    silence_prompt_seconds: float = 9.0
    filler_after_ms: int = 1800
    fast_confirm: bool = True

    # --- business, storage -----------------------------------------------------------------------------------
    clinic_file: Path | None = None
    database_url: str = "sqlite:///data/callie.db"
    recordings_dir: Path = Path("data/recordings")
    results_dir: Path = Path("results")
    now: str | None = None  # freeze the clinic clock (ISO datetime, clinic local time), e.g. for demos and evals
    seed_demo_data: bool = True

    # --- telephony and integrations --------------------------------------------------------------------------
    public_base_url: str = ""  # e.g. https://abc.ngrok.app, used in TwiML <Stream url>
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    twilio_validate_signature: bool = True
    transfer_number: str = "+15555550100"
    google_calendar_id: str = ""
    google_access_token: str = ""

    # --- budget for real API calls ---------------------------------------------------------------------------
    llm_cache: bool = False  # replay identical requests from disk (the evaluation turns it on to save calls)
    llm_cache_dir: Path = Path(".cache/llm")
    llm_ledger: Path | None = Path("results/calls.jsonl")
    llm_max_calls: int = 420
    llm_min_seconds_between_requests: float = 1.0
    llm_max_retries: int = 3

    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "openrouter": DEFAULT_OPENROUTER_MODEL,
            "openai": DEFAULT_OPENAI_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
            "fake": "callie-rules",
        }[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
