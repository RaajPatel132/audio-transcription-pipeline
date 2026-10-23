import logging
import os
from collections.abc import Iterator

import pytest


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
