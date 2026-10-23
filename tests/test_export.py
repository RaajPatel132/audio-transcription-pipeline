import json

import pytest

from transcription.export import format_timestamp, render, to_srt, to_txt, to_vtt
from transcription.schema import Segment
from transcripts import make_transcript

SEGMENTS = [
    Segment(id=0, start=0, end=2.5, text="Hello there."),
    Segment(id=1, start=2.6, end=5.25, text="How are you?"),
    Segment(id=2, start=9, end=3661.007, text="Fine & well."),
]


def test_srt() -> None:
    assert to_srt(make_transcript(SEGMENTS)) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHello there.\n\n"
        "2\n00:00:02,600 --> 00:00:05,250\nHow are you?\n\n"
        "3\n00:00:09,000 --> 01:01:01,007\nFine & well.\n"
    )


def test_vtt() -> None:
    assert to_vtt(make_transcript(SEGMENTS)) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.500\nHello there.\n\n"
        "00:00:02.600 --> 00:00:05.250\nHow are you?\n\n"
        "00:00:09.000 --> 01:01:01.007\nFine &amp; well.\n"
    )


def test_txt_starts_a_paragraph_after_a_long_gap() -> None:
    assert to_txt(make_transcript(SEGMENTS)) == "Hello there. How are you?\n\nFine & well.\n"


def test_json_carries_segments_and_schema_version() -> None:
    data = json.loads(render(make_transcript(SEGMENTS), "json"))

    assert data["schema_version"] == "1.0"
    assert data["segments"][1] == {"id": 1, "start": 2.6, "end": 5.25, "text": "How are you?"}


def test_empty_transcript_renders_without_cues() -> None:
    empty = make_transcript([])

    assert to_srt(empty) == ""
    assert to_txt(empty) == ""
    assert to_vtt(empty) == "WEBVTT\n"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00:00,000"),
        (59.9996, "00:01:00,000"),
        (59.999, "00:00:59,999"),
        (3599.9999, "01:00:00,000"),
        (3661.007, "01:01:01,007"),
        (36000, "10:00:00,000"),
    ],
)
def test_timestamp_rounding(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds, ",") == expected
