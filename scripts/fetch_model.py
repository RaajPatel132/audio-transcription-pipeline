"""Download the Whisper model named by TRANSCRIBE_MODEL (default base.en) ahead of time."""

import sys

from faster_whisper.utils import download_model

from transcription.config import Settings


def main() -> None:
    settings = Settings()
    cache_dir = str(settings.model_dir) if settings.model_dir else None
    path = download_model(settings.model, cache_dir=cache_dir)
    sys.stdout.write(f"{settings.model} is ready in {path}\n")


if __name__ == "__main__":
    main()
