import itertools
from pathlib import Path

import numpy as np
import pytest

from audio_factory import RATE, silence, tone, tones_and_gaps, write_wav
from transcription.chunker import frame_levels_db, plan_chunks, read_chunk
from transcription.config import Settings
from transcription.schema import ChunkInfo

SETTINGS = Settings(chunk_min_s=20, chunk_max_s=30)


def test_frame_levels_in_dbfs() -> None:
    samples = np.concatenate([tone(0.03), silence(0.03), tone(0.01)])

    levels = frame_levels_db(samples, frame_len=480)

    assert len(levels) == 2  # the trailing partial frame is ignored
    assert levels[0] == pytest.approx(-9.03, abs=0.1)
    assert levels[1] == -100.0


def test_short_file_is_one_chunk(tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "a.wav", tone(12))

    assert plan_chunks(wav, SETTINGS) == [ChunkInfo(index=0, start=0, end=12)]


def test_cuts_land_inside_the_silences(tmp_path: Path) -> None:
    samples, spans = tones_and_gaps(tone_s=8, gap_s=1, count=12)
    gaps = [(a[1], b[0]) for a, b in itertools.pairwise(spans)]

    chunks = plan_chunks(write_wav(tmp_path / "a.wav", samples), SETTINGS)

    assert len(chunks) > 3
    for chunk in chunks[:-1]:
        assert any(start < chunk.end < end for start, end in gaps)
        assert not chunk.forced_cut


def test_continuous_sound_forces_cuts_within_the_maximum(tmp_path: Path) -> None:
    chunks = plan_chunks(write_wav(tmp_path / "a.wav", tone(100)), SETTINGS)

    assert len(chunks) > 3
    for chunk in chunks[:-1]:
        assert chunk.forced_cut
        assert chunk.end - chunk.start <= SETTINGS.chunk_max_s


@pytest.mark.parametrize("seconds", [30.5, 61, 100.25, 199])
def test_chunks_cover_the_file_without_gaps_or_overlap(tmp_path: Path, seconds: float) -> None:
    samples, _ = tones_and_gaps(tone_s=4, gap_s=0.6, count=int(seconds / 4.6) + 1)
    samples = samples[: round(seconds * RATE)]
    chunks = plan_chunks(write_wav(tmp_path / "a.wav", samples), SETTINGS)

    assert chunks[0].start == 0
    assert chunks[-1].end == pytest.approx(seconds, abs=0.001)
    for before, after in itertools.pairwise(chunks):
        assert before.end == after.start
        assert SETTINGS.chunk_min_s - 0.001 <= before.end - before.start <= SETTINGS.chunk_max_s
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_read_chunk_returns_only_that_slice(tmp_path: Path) -> None:
    wav = write_wav(tmp_path / "a.wav", np.concatenate([tone(1), silence(1), tone(1)]))

    middle = read_chunk(wav, ChunkInfo(index=1, start=1, end=2))

    assert len(middle) == RATE
    assert not middle.any()
    assert middle.dtype == np.float32
