.PHONY: setup lint format test test-integration clean

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
