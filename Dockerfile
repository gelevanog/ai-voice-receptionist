# syntax=docker/dockerfile:1

# ---- build: resolve dependencies with uv into a self-contained virtualenv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
# EXTRAS="voice" (default) adds the local speech runtimes: faster-whisper (CTranslate2), kokoro-onnx, onnxruntime.
# EXTRAS="" builds a small image that runs with the fake speech components (dashboard and text demo only).
ARG EXTRAS="voice"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)

# ---- runtime: slim image, non-root user ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app results ./results

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    CALLIE_MODELS_DIR=/home/app/.cache/callie/models \
    CALLIE_DATABASE_URL=sqlite:////home/app/data/callie.db \
    CALLIE_RECORDINGS_DIR=/home/app/data/recordings \
    CALLIE_LOG_FORMAT=json

USER app
# Volume mount points must exist (owned by the app user) before Docker initializes named volumes.
RUN mkdir -p /home/app/.cache/huggingface /home/app/.cache/callie/models /home/app/data

# Model weights are not baked in by default: with the real speech stack, run `callie download-models` once
# (docker compose run --rm callie callie download-models) to fill the models volume (~0.5 GB with base.en).
# BAKE_MODELS=true downloads them at build time instead (larger image, no first-run download).
ARG BAKE_MODELS=false
RUN if [ "$BAKE_MODELS" = "true" ]; then callie download-models; fi

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["uvicorn", "callie.web.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
