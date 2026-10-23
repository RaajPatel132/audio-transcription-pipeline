import pytest
from pydantic import ValidationError

from transcription.schema import Segment, Transcript
from transcripts import make_transcript


def test_times_are_rounded_to_milliseconds() -> None:
    segment = Segment(id=0, start=1.00049, end=2.99951, text="x")

    assert (segment.start, segment.end) == (1.0, 3.0)


def test_json_round_trip_keeps_schema_version() -> None:
    transcript = make_transcript()

    restored = Transcript.model_validate_json(transcript.model_dump_json())

    assert restored == transcript
    assert restored.schema_version == "1.0"


def test_models_are_immutable() -> None:
    segment = Segment(id=0, start=0, end=1, text="x")

    with pytest.raises(ValidationError):
        segment.text = "y"  # type: ignore[misc]
