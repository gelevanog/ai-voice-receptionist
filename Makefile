.DEFAULT_GOAL := help
.PHONY: help install models serve serve-real dev test lint format call simulate free-models eval eval-wer eval-bargein report docker-build docker-up clean

REAL = CALLIE_STT_PROVIDER=whisper CALLIE_TTS_PROVIDER=kokoro CALLIE_VAD_PROVIDER=silero

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies incl. dev tools and the local speech runtimes (`voice` + `piper` extras)
	uv sync --all-extras

models:  ## Download Silero VAD, Kokoro-82M, faster-whisper base.en and the Piper caller voice (~0.6 GB, once)
	uv run callie download-models --piper

serve:  ## Dashboard + browser calls on http://localhost:8000 with fake speech and the fake model (no keys, no downloads)
	uv run callie serve --port 8000

serve-real:  ## Same with real local speech (after `make models`); add CALLIE_LLM_PROVIDER=openrouter for a real LLM
	$(REAL) uv run callie serve --port 8000

dev:  ## serve with auto-reload
	uv run callie serve --port 8000 --reload

test:  ## Test suite (no API keys, no model downloads)
	uv run pytest

lint:  ## Ruff lint + format check + mypy (strict)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

format:  ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

call:  ## Text chat with the agent in the terminal (fake model unless CALLIE_LLM_PROVIDER is set)
	uv run callie call

simulate:  ## One simulated caller end to end through audio (scripted caller, no API calls; needs `make models`)
	CALLIE_NOW=2026-10-06T09:30 $(REAL) uv run callie simulate book_basic --caller scripted

free-models:  ## List free OpenRouter models and smoke-test tool calling on three (needs OPENROUTER_API_KEY)
	uv run callie models free --smoke 3

eval:  ## Full end-to-end evaluation with an LLM caller (needs OPENROUTER_API_KEY and `make models`; ~80 min)
	CALLIE_NOW=2026-10-06T09:30 CALLIE_LLM_PROVIDER=openrouter CALLIE_LLM_MODEL=inclusionai/ling-3.0-flash-sante:free \
	CALLIE_LLM_MAX_RETRIES=1 $(REAL) uv run callie eval run --caller llm --caller-model liquid/lfm-2.5-2.6b:free

eval-wer:  ## STT word error rate on the evaluation's caller utterances: clean / phone / phone + noise
	uv run callie eval wer

eval-bargein:  ## Barge-in reaction time and backchannel handling (real VAD/STT/TTS, no API calls)
	CALLIE_NOW=2026-10-06T09:30 $(REAL) uv run callie eval bargein --trials 20

report:  ## Rebuild results/summary.json and results/report.md from the result files
	uv run callie eval report

docker-build:  ## Build the image (speech runtimes included, model weights downloaded separately)
	docker compose build

docker-up:  ## Run in Docker on http://localhost:8000
	docker compose up --build

clean:  ## Remove caches (keeps results/ and the database)
	rm -rf .cache .pytest_cache .mypy_cache .ruff_cache
