import threading
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from audio_factory import silence, tone, tones_and_gaps, write_wav
from transcription.chunker import plan_chunks
from transcription.config import Settings
from transcription.errors import ErrorCode, PipelineError
from transcription.log import job_id
from transcription.pipeline import ChunkStore, MemoryChunkStore, ProgressCallback, transcribe_file
from transcription.schema import ChunkInfo, Transcript
from transcription.transcribers import ChunkResult
from transcription.transcribers.fake import FakeTranscriber

SETTINGS = Settings(chunk_min_s=30, chunk_max_s=60, retry_base_s=0, retry_max_s=0)


class Spy:
    """Wraps a transcriber to count calls, and optionally to fail or to record the job ID."""

    name = "spy"

    def __init__(self, fail_first: int = 0, error: Exception | None = None) -> None:
        self._inner = FakeTranscriber(SETTINGS)
        self._fail_first = fail_first
        self._error = error or RuntimeError("temporary glitch")
        self._lock = threading.Lock()
        self.calls = 0
        self.job_ids: set[str] = set()

    def transcribe(self, samples: NDArray[np.float32], language: str | None) -> ChunkResult:
        with self._lock:
            self.calls += 1
            failing = self.calls <= self._fail_first
        self.job_ids.add(job_id.get())
        if failing:
            raise self._error
        return self._inner.transcribe(samples, language)


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture(scope="module")
def long_audio(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[tuple[float, float]]]:
    """Six minutes: forty 8-second tones, each followed by a 1.2-second gap."""
    samples, spans = tones_and_gaps(tone_s=8, gap_s=1.2, count=40)
    return write_wav(tmp_path_factory.mktemp("audio") / "long.wav", samples), spans


def run(
    path: Path,
    work_dir: Path,
    transcriber: Spy | None = None,
    store: ChunkStore | None = None,
    on_progress: ProgressCallback | None = None,
) -> Transcript:
    return transcribe_file(path, SETTINGS, transcriber or Spy(), work_dir, store, on_progress)


def test_every_tone_appears_once_at_its_true_time(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    path, spans = long_audio

    transcript = run(path, work_dir)

    assert len(transcript.chunks) > 4
    assert len(transcript.segments) == len(spans)
    for segment, (start, end) in zip(transcript.segments, spans, strict=True):
        assert segment.start == pytest.approx(start, abs=0.05)
        assert segment.end == pytest.approx(end, abs=0.05)
    assert transcript.duration_s == pytest.approx(spans[-1][1] + 1.2, abs=0.01)
    assert transcript.language == "en"
    assert transcript.model == "spy"


def test_a_transient_failure_is_retried(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    path, spans = long_audio
    spy = Spy(fail_first=1)

    transcript = run(path, work_dir, spy)

    assert len(transcript.segments) == len(spans)
    assert spy.calls == len(transcript.chunks) + 1


def test_a_permanent_failure_fails_the_run(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    spy = Spy(fail_first=1000)

    with pytest.raises(PipelineError) as excinfo:
        run(long_audio[0], work_dir, spy)

    chunks = plan_chunks(work_dir / "audio.wav", SETTINGS)
    assert excinfo.value.code == ErrorCode.TRANSCRIPTION_FAILED
    assert spy.calls < SETTINGS.retry_attempts * len(chunks)  # chunks not yet started were skipped


def test_a_non_retryable_error_passes_through_unchanged(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    spy = Spy(fail_first=1000, error=PipelineError(ErrorCode.MODEL_UNAVAILABLE, "no model"))

    with pytest.raises(PipelineError) as excinfo:
        run(long_audio[0], work_dir, spy)

    assert excinfo.value.code == ErrorCode.MODEL_UNAVAILABLE


def test_saved_chunks_never_reach_the_transcriber(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    path, _ = long_audio
    store = MemoryChunkStore()
    first = run(path, work_dir, store=store)
    spy = Spy()

    second = run(path, work_dir, spy, store=store)

    assert spy.calls == 0
    assert second.segments == first.segments


def test_only_missing_chunks_are_redone(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    path, _ = long_audio

    class Missing(MemoryChunkStore):
        def __init__(self, hidden_index: int) -> None:
            super().__init__()
            self.hidden_index = hidden_index

        def get(self, chunk: ChunkInfo) -> ChunkResult | None:
            return None if chunk.index == self.hidden_index else super().get(chunk)

    store: ChunkStore = Missing(hidden_index=2)
    run(path, work_dir, store=store)
    spy = Spy()

    run(path, work_dir, spy, store=store)

    assert spy.calls == 1


def test_progress_reaches_the_total(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    updates: list[tuple[int, int]] = []

    transcript = run(long_audio[0], work_dir, on_progress=lambda d, t: updates.append((d, t)))

    total = len(transcript.chunks)
    assert updates == [(done, total) for done in range(1, total + 1)]


def test_worker_threads_keep_the_job_id(
    long_audio: tuple[Path, list[tuple[float, float]]], work_dir: Path
) -> None:
    spy = Spy()
    token = job_id.set("job-42")
    try:
        run(long_audio[0], work_dir, spy)
    finally:
        job_id.reset(token)

    assert spy.job_ids == {"job-42"}


def test_short_file_is_a_single_chunk(tmp_path: Path, work_dir: Path) -> None:
    path = write_wav(tmp_path / "short.wav", np.concatenate([silence(0.5), tone(2), silence(0.5)]))

    transcript = run(path, work_dir)

    assert len(transcript.chunks) == 1
    assert [(s.start, s.end) for s in transcript.segments] == [pytest.approx((0.5, 2.5), abs=0.05)]


def test_bad_input_is_reported_before_any_work(tmp_path: Path, work_dir: Path) -> None:
    text = tmp_path / "notes.mp3"
    text.write_text("not audio " * 100)
    spy = Spy()

    with pytest.raises(PipelineError) as excinfo:
        run(text, work_dir, spy)

    assert excinfo.value.code == ErrorCode.UNSUPPORTED_FORMAT
    assert spy.calls == 0
