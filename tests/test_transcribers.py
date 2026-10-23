import numpy as np
import pytest

from audio_factory import silence, tone, tones_and_gaps
from transcription.config import Settings
from transcription.transcribers import build_transcriber
from transcription.transcribers.fake import FakeTranscriber


def fake() -> FakeTranscriber:
    return FakeTranscriber(Settings())


def test_segments_line_up_with_the_sound() -> None:
    samples, spans = tones_and_gaps(tone_s=3, gap_s=1, count=4)

    result = fake().transcribe(samples, None)

    assert len(result.segments) == 4
    for segment, (start, end) in zip(result.segments, spans, strict=True):
        assert segment.start == pytest.approx(start, abs=0.03)
        assert segment.end == pytest.approx(end, abs=0.03)


def test_silence_has_no_segments() -> None:
    assert fake().transcribe(silence(5), None).segments == []


def test_sounds_closer_than_the_quiet_window_are_one_segment() -> None:
    samples = np.concatenate([tone(1), silence(0.1), tone(1)])

    assert len(fake().transcribe(samples, None).segments) == 1


def test_language_defaults_to_english_and_follows_the_request() -> None:
    assert fake().transcribe(tone(1), None).language == "en"
    assert fake().transcribe(tone(1), "de").language == "de"


def test_the_same_audio_gives_the_same_result() -> None:
    samples, _ = tones_and_gaps(2, 1, 3)

    assert fake().transcribe(samples, None) == fake().transcribe(samples, None)


def test_factory_builds_the_fake_for_mock_runs() -> None:
    assert build_transcriber(Settings(), mock=True).name == "fake"
