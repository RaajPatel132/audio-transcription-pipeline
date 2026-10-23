import argparse
import logging
import secrets
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from transcription.config import Settings
from transcription.errors import PipelineError
from transcription.export import Format, render
from transcription.log import bind_job, configure_logging
from transcription.pipeline import transcribe_file
from transcription.transcribers import build_transcriber

EXIT_BAD_INPUT = 3
EXIT_FAILED = 4

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.audio.is_file():
        parser.error(f"{args.audio} is not a file")
    settings = _load_settings(args, parser)

    configure_logging("DEBUG" if args.verbose else "INFO")
    with bind_job(secrets.token_hex(3)):
        return _run(args, settings)


def _run(args: argparse.Namespace, settings: Settings) -> int:
    try:
        transcriber = build_transcriber(settings, mock=args.mock)
        with tempfile.TemporaryDirectory(prefix="transcribe-") as work_dir:
            transcript = transcribe_file(
                args.audio,
                settings,
                transcriber,
                Path(work_dir),
                on_progress=lambda done, total: logger.info("%d/%d chunks done", done, total),
            )
    except PipelineError as exc:
        sys.stderr.write(f"error: {exc} [{exc.code}]\n")
        return EXIT_BAD_INPUT if exc.is_bad_input else EXIT_FAILED
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130

    text = render(transcript, args.format)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _load_settings(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Settings:
    overrides = {"model": args.model, "chunk_workers": args.workers, "language": args.language}
    try:
        return Settings(**{k: v for k, v in overrides.items() if v is not None})
    except ValidationError as exc:
        problems = (f"{'.'.join(map(str, e['loc']))} {e['msg']}".strip() for e in exc.errors())
        parser.error("invalid settings: " + "; ".join(problems))


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe an audio file into text with start and end times per segment.",
    )
    parser.add_argument("audio", type=Path, help="audio file (WAV, MP3, FLAC, Ogg, M4A, ...)")
    parser.add_argument("--format", choices=get_args(Format), default="json")
    parser.add_argument("--output", type=Path, help="write the transcript here, not to stdout")
    parser.add_argument("--model", help="Whisper model name (default: base.en)")
    parser.add_argument("--workers", type=_positive_int, help="chunks transcribed in parallel")
    parser.add_argument("--language", help="language code; by default English-only models use en")
    parser.add_argument("--mock", action="store_true", help="fake transcriber, no model needed")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser
