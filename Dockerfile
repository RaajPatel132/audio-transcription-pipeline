FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

FROM base AS test
COPY scripts ./scripts
COPY tests ./tests
COPY samples ./samples
RUN uv sync --frozen
RUN ruff check . && ruff format --check . && mypy && HF_HUB_OFFLINE=1 pytest

FROM base AS runtime
ARG MODEL=base.en
ENV TRANSCRIBE_MODEL=$MODEL TRANSCRIBE_MODEL_DIR=/models TRANSCRIBE_DATA_DIR=/data
COPY scripts/fetch_model.py ./scripts/
RUN python scripts/fetch_model.py && chmod -R a+rX /models
COPY samples ./samples
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
ENV HF_HUB_OFFLINE=1
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["uvicorn", "transcription.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
