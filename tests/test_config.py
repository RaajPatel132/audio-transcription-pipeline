import os

import pytest
from pydantic import ValidationError

from transcription.config import Settings


def test_defaults() -> None:
    settings = Settings()

    assert settings.model == "base.en"
    assert settings.compute_type == "int8"
    assert settings.max_file_bytes == 500 * 1024 * 1024


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "tiny.en")
    monkeypatch.setenv("TRANSCRIBE_CHUNK_WORKERS", "3")

    settings = Settings()

    assert settings.model == "tiny.en"
    assert settings.chunk_workers == 3


def test_explicit_arguments_beat_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_MODEL", "tiny.en")

    assert Settings(model="small.en").model == "small.en"


def test_chunk_bounds_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="chunk_min_s"):
        Settings(chunk_min_s=80, chunk_max_s=60)


def test_non_positive_limits_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(max_duration_s=0)


@pytest.mark.parametrize(
    ("model", "language", "expected"),
    [("base.en", None, "en"), ("base", None, None), ("base", "de", "de"), ("base.en", "fr", "fr")],
)
def test_forced_language(model: str, language: str | None, expected: str | None) -> None:
    assert Settings(model=model, language=language).forced_language == expected


def test_threads_use_half_the_cpus_split_between_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    assert Settings(chunk_workers=1).threads_per_worker == 4
    assert Settings(chunk_workers=2).threads_per_worker == 2
    assert Settings(chunk_workers=16).threads_per_worker == 1
    assert Settings(chunk_workers=2, cpu_threads=3).threads_per_worker == 3
