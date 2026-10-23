import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from audio_factory import tones_and_gaps, write_wav


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's TRANSCRIBE_* variables out of the tests."""
    for name in [n for n in os.environ if n.startswith("TRANSCRIBE_")]:
        monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    logging.getLogger("transcription").setLevel(logging.NOTSET)


@pytest.fixture(scope="session")
def long_audio(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[tuple[float, float]]]:
    """Six minutes: forty 8-second tones, each followed by a 1.2-second gap."""
    samples, spans = tones_and_gaps(tone_s=8, gap_s=1.2, count=40)
    return write_wav(tmp_path_factory.mktemp("audio") / "long.wav", samples), spans
