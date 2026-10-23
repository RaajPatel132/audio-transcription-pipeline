"""Download the Whisper model named by TRANSCRIBE_MODEL (default base.en) ahead of time."""

import sys
from pathlib import Path

from faster_whisper.utils import download_model

from transcription.config import Settings


def fetch(settings: Settings) -> Path:
    cache_dir = str(settings.model_dir) if settings.model_dir else None
    return Path(download_model(settings.model, cache_dir=cache_dir))


def main() -> None:
    settings = Settings()
    sys.stdout.write(f"{settings.model} is ready in {fetch(settings)}\n")


if __name__ == "__main__":
    main()
