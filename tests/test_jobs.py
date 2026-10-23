import secrets
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audio_factory import silence, tone, write_wav
from fakes import Spy, wait_for
from transcription.chunker import plan_chunks, read_chunk
from transcription.config import Settings
from transcription.errors import ErrorCode, PipelineError
from transcription.files import JobFiles
from transcription.ingest import normalise
from transcription.jobs import JobRunner
from transcription.log import configure_logging
from transcription.schema import Transcript
from transcription.store import Job, JobState, JobStore
from transcription.transcribers.fake import FakeTranscriber

Audio = tuple[Path, list[tuple[float, float]]]


class Harness:
    """A store, a folder and whatever runners a test starts, all cleaned up at the end."""

    def __init__(self, root: Path, **overrides: Any) -> None:
        self.settings = Settings(
            data_dir=root,
            chunk_min_s=30,
            chunk_max_s=60,
            retry_base_s=0,
            retry_max_s=0,
            **overrides,
        )
        self.files = JobFiles(root)
        self.store = JobStore(root / "jobs.db")
        self.store.init_schema()
        self._runners: list[JobRunner] = []

    def runner(self, transcriber: Spy) -> JobRunner:
        runner = JobRunner(self.store, self.files, self.settings, transcriber)
        self._runners.append(runner)
        return runner

    def run(self, transcriber: Spy) -> JobRunner:
        runner = self.runner(transcriber)
        runner.start()
        runner.notify()
        return runner

    def submit(self, audio: Path, **fields: Any) -> Job:
        job = Job(
            dedupe_key=secrets.token_hex(4),
            filename=audio.name,
            size_bytes=audio.stat().st_size,
            sha256="0" * 64,
            **fields,
        )
        received = self.files.incoming(audio.name)
        shutil.copy(audio, received)
        self.files.adopt(received, job.id)
        self.store.create(job)
        return job

    def job(self, job: Job) -> Job:
        found = self.store.get(job.id)
        assert found is not None
        return found

    def wait_for_state(self, job: Job, state: JobState) -> Job:
        wait_for(lambda: self.job(job).state == state)
        return self.job(job)

    def job_dir(self, job: Job) -> Path:
        return self.files.root / "jobs" / job.id

    def stop(self) -> None:
        for runner in self._runners:
            runner.stop(10)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    harness = Harness(tmp_path / "service")
    yield harness
    harness.stop()


@pytest.fixture(scope="module")
def short_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    samples = np.concatenate([tone(2), silence(1), tone(2)])
    return write_wav(tmp_path_factory.mktemp("short") / "clip.wav", samples)


def test_a_queued_job_completes(harness: Harness, long_audio: Audio) -> None:
    path, spans = long_audio
    job = harness.submit(path)

    harness.run(Spy())
    done = harness.wait_for_state(job, JobState.COMPLETED)

    assert done.attempts == 1
    assert done.chunks_total is not None
    assert done.chunks_total > 1
    assert done.chunks_done == done.chunks_total
    assert done.finished_at is not None
    transcript = Transcript.model_validate_json(harness.files.result_path(job.id).read_text())
    assert len(transcript.segments) == len(spans)
    assert transcript.source.filename == "long.wav"
    assert not (harness.job_dir(job) / "work").exists()


def test_a_transcriber_that_always_fails_fails_the_job(harness: Harness, short_audio: Path) -> None:
    job = harness.submit(short_audio)

    harness.run(Spy(fail_first=1000))
    failed = harness.wait_for_state(job, JobState.FAILED)

    assert failed.error_code == ErrorCode.TRANSCRIPTION_FAILED
    assert failed.error_message
    assert not harness.files.result_path(job.id).exists()
    assert not (harness.job_dir(job) / "work").exists()


def test_bad_input_found_while_running_keeps_its_own_code(harness: Harness, tmp_path: Path) -> None:
    text = tmp_path / "notes.mp3"
    text.write_text("this is not audio\n" * 50)
    job = harness.submit(text)

    harness.run(Spy())
    failed = harness.wait_for_state(job, JobState.FAILED)

    assert failed.error_code == ErrorCode.UNSUPPORTED_FORMAT


def test_a_missing_model_fails_the_job_with_its_code(harness: Harness, short_audio: Path) -> None:
    error = PipelineError(ErrorCode.MODEL_UNAVAILABLE, "no model")
    job = harness.submit(short_audio)

    harness.run(Spy(fail_first=1000, error=error))
    failed = harness.wait_for_state(job, JobState.FAILED)

    assert failed.error_code == ErrorCode.MODEL_UNAVAILABLE


def test_an_unexpected_error_fails_that_job_and_the_worker_carries_on(
    harness: Harness, short_audio: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_result = harness.files.write_result
    attempts: list[str] = []

    def flaky(job_id: str, text: str) -> None:
        attempts.append(job_id)
        if len(attempts) == 1:
            raise OSError("disk full")
        write_result(job_id, text)

    monkeypatch.setattr(harness.files, "write_result", flaky)
    first = harness.submit(short_audio)
    second = harness.submit(short_audio)

    harness.run(Spy())
    failed = harness.wait_for_state(first, JobState.FAILED)
    harness.wait_for_state(second, JobState.COMPLETED)

    assert failed.error_code == ErrorCode.INTERNAL_ERROR
    assert "disk full" not in (failed.error_message or "")


def test_a_recovered_job_only_transcribes_the_missing_chunks(
    harness: Harness, long_audio: Audio, tmp_path: Path
) -> None:
    path, _ = long_audio
    job = harness.submit(path)
    harness.store.claim_next()  # the process dies while this job is running
    wav = tmp_path / "audio.wav"
    normalise(path, wav, harness.settings)
    chunks = plan_chunks(wav, harness.settings)
    fake = FakeTranscriber(harness.settings)
    for chunk in chunks[:3]:
        harness.store.save_chunk(job.id, chunk, fake.transcribe(read_chunk(wav, chunk), None))
    spy = Spy()

    runner = harness.runner(spy)
    runner.recover()
    runner.start()
    runner.notify()
    done = harness.wait_for_state(job, JobState.COMPLETED)

    assert spy.calls == len(chunks) - 3
    assert done.attempts == 2
    transcript = Transcript.model_validate_json(harness.files.result_path(job.id).read_text())
    starts = [segment.start for segment in transcript.segments]
    assert starts == sorted(starts)


def test_recovery_fails_a_job_that_used_all_its_attempts(
    harness: Harness, short_audio: Path
) -> None:
    worn = harness.submit(short_audio, state=JobState.PROCESSING, attempts=3)
    fresh = harness.submit(short_audio, state=JobState.PROCESSING, attempts=1)
    harness.files.work_dir(worn.id)
    harness.files.incoming("half-received.mp3").write_bytes(b"partial")

    harness.runner(Spy()).recover()

    assert harness.job(worn).state == JobState.FAILED
    assert harness.job(worn).error_code == ErrorCode.INTERNAL_ERROR
    assert not (harness.job_dir(worn) / "work").exists()
    assert harness.job(fresh).state == JobState.QUEUED
    assert list((harness.files.root / "incoming").iterdir()) == []


def test_a_stop_during_a_job_puts_it_back_in_the_queue(
    harness: Harness, long_audio: Audio, tmp_path: Path
) -> None:
    path, _ = long_audio
    job = harness.submit(path)
    gate = threading.Event()
    spy = Spy(gate=gate)
    runner = harness.run(spy)
    wait_for(spy.started.is_set)

    stopper = threading.Thread(target=runner.stop, args=(10,))
    stopper.start()
    wait_for(lambda: runner.stopping)
    gate.set()
    stopper.join(15)

    wav = tmp_path / "audio.wav"
    normalise(path, wav, harness.settings)
    chunks = plan_chunks(wav, harness.settings)
    saved = [c for c in chunks if harness.store.load_chunk(job.id, c) is not None]
    stopped = harness.job(job)
    assert not stopper.is_alive()
    assert stopped.state == JobState.QUEUED
    assert stopped.attempts == 0
    assert 1 <= len(saved) < len(chunks)
    assert spy.calls == len(saved)


def test_a_job_resumes_after_a_clean_stop(harness: Harness, long_audio: Audio) -> None:
    path, spans = long_audio
    job = harness.submit(path)
    gate = threading.Event()
    first = Spy(gate=gate)
    runner = harness.run(first)
    wait_for(first.started.is_set)
    stopper = threading.Thread(target=runner.stop, args=(10,))
    stopper.start()
    wait_for(lambda: runner.stopping)
    gate.set()
    stopper.join(15)
    second = Spy()

    harness.run(second)
    done = harness.wait_for_state(job, JobState.COMPLETED)

    transcript = Transcript.model_validate_json(harness.files.result_path(job.id).read_text())
    assert len(transcript.segments) == len(spans)
    assert first.calls + second.calls == done.chunks_total
    assert done.attempts == 1


def test_the_transcriber_runs_under_the_jobs_own_id(harness: Harness, long_audio: Audio) -> None:
    job = harness.submit(long_audio[0])
    spy = Spy()

    harness.run(spy)
    harness.wait_for_state(job, JobState.COMPLETED)

    assert spy.job_ids == {job.id}


def test_every_log_line_of_a_job_carries_its_id(
    harness: Harness, long_audio: Audio, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")
    job = harness.submit(long_audio[0])

    runner = harness.run(Spy())
    harness.wait_for_state(job, JobState.COMPLETED)
    runner.stop(10)

    lines = capsys.readouterr().err.splitlines()
    assert any("chunk 1/" in line for line in lines)
    assert [line for line in lines if f"job={job.id} " not in line] == []


def test_two_workers_never_run_the_same_job_twice(short_audio: Path, tmp_path: Path) -> None:
    pair = Harness(tmp_path / "pair", job_workers=2)
    try:
        jobs = [pair.submit(short_audio) for _ in range(6)]
        spy = Spy()

        pair.run(spy)
        finished = [pair.wait_for_state(job, JobState.COMPLETED) for job in jobs]

        assert [job.attempts for job in finished] == [1] * 6
        assert spy.calls == 6
    finally:
        pair.stop()


def test_an_idle_runner_stops_at_once(harness: Harness) -> None:
    runner = harness.run(Spy())
    started = time.monotonic()

    runner.stop(10)

    assert time.monotonic() - started < 2
