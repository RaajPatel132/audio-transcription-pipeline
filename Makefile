.PHONY: setup lint format test test-integration fetch-model demo-mock demo serve serve-mock demo-api long-sample benchmark samples clean

DEMO_CLIP = samples/garden.mp3
SERVE = uv run uvicorn transcription.api:create_app --factory --host 127.0.0.1 --port 8000

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

test-integration:
	TRANSCRIBE_MODEL=tiny.en uv run python scripts/fetch_model.py
	uv run pytest -m integration

fetch-model:
	uv run python scripts/fetch_model.py

demo-mock:
	uv run transcribe $(DEMO_CLIP) --mock

demo:
	uv run transcribe $(DEMO_CLIP) --format srt

serve:
	$(SERVE)

serve-mock:
	TRANSCRIBE_MOCK=1 $(SERVE)

demo-api:
	uv run python scripts/api_demo.py $(DEMO_CLIP) --format srt

long-sample:
	uv run python scripts/make_long_sample.py

benchmark:
	test -f data/long-sample.mp3 || $(MAKE) long-sample
	uv run python scripts/benchmark.py

samples:
	uv run python scripts/make_samples.py

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
